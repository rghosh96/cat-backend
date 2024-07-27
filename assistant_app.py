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
import base64
from collections import defaultdict
import re

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

client_rashi = OpenAI(api_key=os.environ.get("RASHI_OPENAI_KEY"))
client_verg = OpenAI(api_key=os.environ.get("VERG_OPENAI_KEY"))

speech_file_path = "output.mp3"

# Path to store the assistant ID
ASSISTANT_ID_FILE = "assistant_ids_cat.json"

# Path to store user thread IDs
USER_THREADS_FILE_INFORMATION = "user_threads_information.json"
# Path to store user thread IDs
USER_THREADS_FILE_BROWSE = "user_threads_browse.json"

# Data model for the request
class InteractionRequest(BaseModel):
    message: str
    language_style: str
    user_info: str


# Function to fetch file content
def fetch_file_content(file_id):
    try:
        file_content = client_rashi.files.retrieve(file_id)
        return file_content.filename
    except Exception as e:
        print(f"Error fetching file {file_id}: {e}")
        return None

# Function to get the saved assistant ID
def get_assistant_id(cat_bot_id):
    print(cat_bot_id)
    try:
        with open(ASSISTANT_ID_FILE, "r") as f:
            data = json.load(f)
            return data[cat_bot_id]
    except FileNotFoundError:
        return None

# Function to get the user's thread ID
def get_user_thread_id(user_id, type):
    if (type == 0):
        user_threads_file = USER_THREADS_FILE_INFORMATION
    else:
        user_threads_file = USER_THREADS_FILE_BROWSE
    try:
        with open(user_threads_file, "r") as f:
            data = json.load(f)
            return data.get(user_id)
    except FileNotFoundError:
        return None

# Function to save the user's thread ID
def save_user_thread_id(user_id, thread_id, type):
    if (type == 0):
        user_threads_file = USER_THREADS_FILE_INFORMATION
    else:
        user_threads_file = USER_THREADS_FILE_BROWSE
    try:
        with open(user_threads_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    data[user_id] = thread_id

    with open(user_threads_file, "w") as f:
        json.dump(data, f)

# Function to create a thread and interact with the assistant
def interact_with_assistant(user_id, cat_bot_id, user_message, health_literacy):
    assistant_id = get_assistant_id(cat_bot_id)
    if not assistant_id:
        raise HTTPException(status_code=500, detail="Assistant not initialized")

    if cat_bot_id == "control_assistant_id":
        combined_prompt = f"User: {user_message}"
    else:
        combined_prompt = f"User: {user_message}\n BRIEF SCORE: {health_literacy}\n"
    
    thread_id = get_user_thread_id(user_id, 0)
    if thread_id:
        # Retrieve the existing thread and append the new message
        client_rashi.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=combined_prompt
        )
    else:
        # Create a new thread
        thread = client_rashi.beta.threads.create(
            messages=[
                 {
                    "role": "user",
                    "content": combined_prompt,
                }
            ]
        )
        save_user_thread_id(user_id, thread.id, 0)
    
    run = client_rashi.beta.threads.runs.create_and_poll(
        thread_id=thread_id if thread_id else thread.id, assistant_id=assistant_id
    )

    messages = list(client_rashi.beta.threads.messages.list(thread_id=thread_id if thread_id else thread.id, run_id=run.id))
    message_content = messages[-1].content[0].text.value
    json_string = re.search(r'\{.*\}', message_content, re.DOTALL).group()

    parsed_value = json.loads(json_string)
    # Access the "Topic" and "Response"
    topic = parsed_value.get("Topic")
    response = parsed_value.get("Response")

    cleaned_response = re.sub(r'【.*?】', '', response)
    cleaned_response.strip()  # Remove any leading/trailing whitespace

    return topic, cleaned_response

# Function to create a thread and search with the assistant
def browse_with_assistant(user_id, cat_bot_id, user_message, health_literacy):
    assistant_id = get_assistant_id(cat_bot_id)
    if not assistant_id:
        raise HTTPException(status_code=500, detail="Assistant not initialized")

    if cat_bot_id == "ct_control_assistant_id":
        combined_prompt = f"User: {user_message}"
    else:
        combined_prompt = f"User: {user_message}\n BRIEF SCORE: {health_literacy}\n"
    
    thread_id = get_user_thread_id(user_id, 1)
    if thread_id:
        # Retrieve the existing thread and append the new message
        client_rashi.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=combined_prompt
        )
    else:
        # Create a new thread
        thread = client_rashi.beta.threads.create(
            messages=[
                 {
                    "role": "user",
                    "content": combined_prompt,
                }
            ]
        )
        save_user_thread_id(user_id, thread.id, 1)
    
    run = client_rashi.beta.threads.runs.create_and_poll(
        thread_id=thread_id if thread_id else thread.id, assistant_id=assistant_id
    )

    messages = list(client_rashi.beta.threads.messages.list(thread_id=thread_id if thread_id else thread.id, run_id=run.id))
    message_content = messages[-1].content[0].text.value

    cleaned_response_browse = re.sub(r'【.*?】', '', message_content)
    cleaned_response_browse.strip()  # Remove any leading/trailing whitespace

    return cleaned_response_browse

def generateAudio(textToAudio):
    audioResponse = client_verg.audio.speech.create(
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

@app.post("/api/cat/assistant")
async def interact(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    user_id = data['user_id']
    cat_bot_id = data['cat_bot_id']
    user_message = data['user_message']
    health_literacy = data['health_literacy']
    topic, response = interact_with_assistant(
        user_id, cat_bot_id, user_message, health_literacy
    )

    audio_response = generateAudio(response)
    audio_base64 = base64.b64encode(audio_response).decode('utf-8')
    audio_data_url = f"data:audio/wav;base64,{audio_base64}"

    return {"topic": topic, "response": response, "audio": audio_data_url}

@app.post("/api/cat/browse")
async def search(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    user_id = data['user_id']
    cat_bot_id = data['cat_bot_id']
    user_message = data['user_message']
    health_literacy = data['health_literacy']
    response = browse_with_assistant(
        user_id, cat_bot_id, user_message, health_literacy
    )

    audio_response = generateAudio(response)
    audio_base64 = base64.b64encode(audio_response).decode('utf-8')
    audio_data_url = f"data:audio/wav;base64,{audio_base64}"

    return {"response": response, "audio": audio_data_url}