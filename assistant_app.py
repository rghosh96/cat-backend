from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pydantic import BaseModel
import json
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

# Path to store the assistant ID
ASSISTANT_ID_FILE = "assistant_ids.json"

# Path to store user thread IDs
USER_THREADS_FILE = "user_threads.json"

SAMPLE_USER_INFO_Q1 = "My main concern is trying to figure out what I should do about this breast cancer. It's all so overwhelming, and I just want to make sure I'm doing the right thing to get better. My goal is to beat this thing and keep being there for my family and my students."
SAMPLE_USER_INFO_Q2 = "I usually talk to my husband first and then my daughters. He helps me understand things better and supports me through it all. Sometimes I ask my friends who've been through similar things for advice or look things up online."
SAMPLE_USER_INFO_Q3 = "I guess I would consider it if it seemed like it could help me. But honestly, I'd be pretty skeptical. I'd want to know exactly what they're testing and what the risks are. And I'd want to make sure I'm not just being used as a guinea pig. But if it seemed like it could offer me some hope or a chance at better treatment, I might be willing to give it a shot."
SAMPLE_USER_INFO_Q4 = "I might not want to do it if I felt like I was being pressured into it or if I didn't trust the people running the trial. And if I didn't really understand what they were asking me to do or what the potential risks were, that would definitely make me hesitant. I just want to make sure I'm making the best decisions for myself and my family."

# Data model for the request
class InteractionRequest(BaseModel):
    message: str
    language_style: str
    user_info: str

# Function to get the saved assistant ID
def get_assistant_id(cat_bot_id):
    try:
        with open(ASSISTANT_ID_FILE, "r") as f:
            data = json.load(f)
            return data[cat_bot_id]
    except FileNotFoundError:
        return None

# Function to get the user's thread ID
def get_user_thread_id(user_id):
    try:
        with open(USER_THREADS_FILE, "r") as f:
            data = json.load(f)
            return data.get(user_id)
    except FileNotFoundError:
        return None

# Function to save the user's thread ID
def save_user_thread_id(user_id, thread_id):
    try:
        with open(USER_THREADS_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    data[user_id] = thread_id

    with open(USER_THREADS_FILE, "w") as f:
        json.dump(data, f)

# Function to create a thread and interact with the assistant
def interact_with_assistant(user_id, cat_bot_id, user_message, user_info):
    assistant_id = get_assistant_id(cat_bot_id)
    print(assistant_id)
    if not assistant_id:
        raise HTTPException(status_code=500, detail="Assistant not initialized")

    combined_prompt = f"User: {user_message}\nBackground Information About Me: {user_info}"

    print("COMBINED PROMPT:", combined_prompt)
    
    thread_id = get_user_thread_id(user_id)
    if thread_id:
        # Retrieve the existing thread and append the new message
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=combined_prompt
        )
    else:
        # Create a new thread
        thread = client.beta.threads.create(
            messages=[
                 {
                    "role": "user",
                    "content": combined_prompt,
                }
            ]
        )
        save_user_thread_id(user_id, thread.id)
    
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id if thread_id else thread.id, assistant_id=assistant_id
    )
    messages = list(client.beta.threads.messages.list(thread_id=thread_id if thread_id else thread.id, run_id=run.id))
    print("MESSAGES:", messages)
    message_content = messages[-1].content[0].text.value
    print("MESSAGE_CONTENT:", message_content)
    parsed_value = json.loads(message_content)
    # Access the "Topic" and "Response"
    topic = parsed_value.get("Topic")
    response = parsed_value.get("Response")
    highlights = parsed_value.get("Highlights")

    return topic, response, highlights

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

@app.post("/api/assistant")
async def interact(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    user_id = data['user_id']
    cat_bot_id = data['cat_bot_id']
    user_message = data['user_message']
    user_info = SAMPLE_USER_INFO_Q1 + " " + SAMPLE_USER_INFO_Q2 + " " + SAMPLE_USER_INFO_Q3 + " " + SAMPLE_USER_INFO_Q4

    topic, response, highlights = interact_with_assistant(
        user_id, cat_bot_id, user_message, user_info
    )
    return {"topic": topic, "response": response, "highlights": highlights}

@app.post("/api/intro")
async def interact(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    user_id = data['user_id']
    cat_bot_id = data['cat_bot_id']
    user_info = SAMPLE_USER_INFO_Q1 + " " + SAMPLE_USER_INFO_Q2 + " " + SAMPLE_USER_INFO_Q3 + " " + SAMPLE_USER_INFO_Q4
    user_message = "Greet the user based on the follwoing Background Information:"

    topic, response, highlights = interact_with_assistant(
        user_id, cat_bot_id, user_message, user_info
    )
    return {"topic": topic, "response": response, "highlights": highlights}