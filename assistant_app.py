from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# origins = ["*"]
# app.add_middleware(
#  CORSMiddleware,
#  allow_origins=origins,
#  allow_credentials=True,
#  allow_methods=["*"],
#  allow_headers=["*"],
# )

load_dotenv()  # take environment variables from .env.

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

speech_file_path = "output.mp3"

GPT_MODEL = "gpt-3.5-turbo"

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

def initializeAssistantAPI():
    global assistant
    assistant = client.beta.assistants.create(
        name="Financial Analyst Assistant",
        instructions="Use the PDF to answer the user's questions. Otherwise, let them know you can't answer.",
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

@app.post('/api/chatbot')
async def chatbot(request: Request, background_tasks: BackgroundTasks):
    initializeAssistantAPI()
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