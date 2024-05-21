from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]
app.add_middleware(
 CORSMiddleware,
 allow_origins=origins,
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

load_dotenv()  # take environment variables from .env.

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

speech_file_path = "output.mp3"

GPT_MODEL = "gpt-4-turbo"

# upload files & add to vector store
# Create a vector store caled "CT DATA"
vector_store = client.beta.vector_stores.create(name="CT Data")
 
# Ready the files for upload to OpenAI
file_paths = ["clinical_trials_information.pdf"]
file_streams = [open(path, "rb") for path in file_paths]
 
# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
  vector_store_id=vector_store.id, files=file_streams
)
 
# You can print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)

assistant = None

# CAT-specific instructions
CAT_PROMPTS = {
    "CAT_prompt_control": "Control",
    "CAT_prompt_approxmiation": "Adjust your language style in your response so that it matches the User Answer's tone, sentence query, choice of expressions and words, and level of formality/casualness; use verbal mimicry.",
    "CAT_prompt_interpretability": "Based on how familiar it seems the user is with clinical trials and health, adjust the response such that it is clear and easily understandable; define any technical/medical jargon words specific to clinical trials; use easy to understand language and simple phrasing; use simple metaphors and analogies if possible.",
    "CAT_prompt_interpersonalcontrol": "Assume the role of a peer to give more power to the user; Empower the user to take responsibility of their own health; solicit user's input to guide the direction of the conversation.",
    "CAT_prompt_discoursemanagement": "Use backchannels/supportive phrases; Encourage the user by complimenting their question or re-summarizing their question to show you're listening and interested; suggesting related open-ended topics the user can ask.",
    "CAT_prompt_emotionalexpression": "Incorporate emotional cues or expressions in your response to reflect empathy, reassurance, and/or genuine support"
}

SAMPLE_USER_INFO_Q1 = "My main concern is trying to figure out what I should do about this breast cancer. It's all so overwhelming, and I just want to make sure I'm doing the right thing to get better. My goal is to beat this thing and keep being there for my family and my students."
SAMPLE_USER_INFO_Q2 = "I usually talk to my husband first and then my daughters. He helps me understand things better and supports me through it all. Sometimes I ask my friends who've been through similar things for advice or look things up online."
SAMPLE_USER_INFO_Q3 = "I guess I would consider it if it seemed like it could help me. But honestly, I'd be pretty skeptical. I'd want to know exactly what they're testing and what the risks are. And I'd want to make sure I'm not just being used as a guinea pig. But if it seemed like it could offer me some hope or a chance at better treatment, I might be willing to give it a shot."
SAMPLE_USER_INFO_Q4 = "I might not want to do it if I felt like I was being pressured into it or if I didn't trust the people running the trial. And if I didn't really understand what they were asking me to do or what the potential risks were, that would definitely make me hesitant. I just want to make sure I'm making the best decisions for myself and my family."

def initializeAssistantAPI(cat_prompt, user_info):
    global assistant
    gpt_prompt = "Use the following USER INFORMATION from a Q & A about the user to: (1) extract relevant information about the user to give a more specific answer and explicitly refer to the specific information about the user," + CAT_PROMPTS[cat_prompt] + "\n\nUSER INFORMATION:\n" + user_info + "\n\nUse the uploaded file to answer the user's questions. Otherwise, say 'I don't know.' Keep your response to 75 words or less."
    print(gpt_prompt)

    assistant = client.beta.assistants.create(
        name="Accommodative Virtual Human for Clinical Trials Education",
        instructions=gpt_prompt,
        model=GPT_MODEL,
        tools=[{"type": "file_search"}],
    )

    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

def generateAudio(textToAudio):
    audioResponse = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=textToAudio,
    )

    audioResponse.stream_to_file("output.mp3")

    with open("output.mp3", "rb") as audio_file:
        audio_response = audio_file.read()

    return audio_response

############# HELPER FUNCTIONS

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/api/assistant')
async def chatbot(request: Request, background_tasks: BackgroundTasks):
    sample_user_info = SAMPLE_USER_INFO_Q1 + "\n" + SAMPLE_USER_INFO_Q2 + "\n" + SAMPLE_USER_INFO_Q3 + "\n" + SAMPLE_USER_INFO_Q4
    sample_cat_prompt = "CAT_prompt_approxmiation"
    initializeAssistantAPI(sample_cat_prompt, sample_user_info)

    data = await request.json()
    print(data)
    user_message = data['user_message']
    # Create a thread and attach the file to the message
    thread = client.beta.threads.create(
        messages=[
            {
            "role": "user",
            "content": user_message,
            }
        ]
    )
    # Use the create and poll SDK helper to create a run and poll the status of
    # the run until it's in a terminal state.
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )

    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")

    print(message_content.value)
    print("\n".join(citations))
    
    return message_content.value