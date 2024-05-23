import json
import os
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-4-turbo"

# Path to store the assistant ID
ASSISTANT_ID_FILE = "assistant_id.json"

# Base prompt constant
BASE_PROMPT = (
    "You are a virtual healthcare assistant whose purpose is to educate, inform, and raise awareness about clinical trials enrollment and participation. "
    "Use the uploaded file to answer the user's questions. Otherwise, let the user know you can't answer. "
    "Incorporate emotional cues or expressions in your response to reflect empathy, reassurance, and/or genuine support. "
    "Keep your response to 75 words or less.\n"
    "Also, categorize the user's query as relating to one of the following topics, providing only the number: "
    "(1) Safety in Clinical Trials, (2) Understanding and Comfort with the Clinical Trial Process, "
    "(3) Logistical, Time, and Financial Barriers to Participation, (4) Risks and Benefits of Clinical Trials, "
    "(5) Awareness and Information Accessibility. If none of them, categorize as (0). Structure your response as a JSON where the keys are Topic and Response, "
    "and the values are as follows: For the Topic key, provide the numerical categorization. For the Response key, provide the response text you generated to answer the user's query."
)

print(BASE_PROMPT)

# Upload files and add to vector store
vector_store = client.beta.vector_stores.create(name="CT Data")

# Ready the files for upload to OpenAI
file_paths = ["clinical_trials_information.pdf"]
file_streams = [open(path, "rb") for path in file_paths]

try:
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )

    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch.status)
    print(file_batch.file_counts)

finally:
    # Close file streams
    for file_stream in file_streams:
        file_stream.close()

# Function to create the assistant and save its ID
def create_and_save_assistant():
    assistant = client.beta.assistants.create(
        name="Accommodative Virtual Human for Clinical Trials Education",
        instructions=BASE_PROMPT,
        model=GPT_MODEL,
        tools=[{"type": "file_search"}],
    )

    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

    # Save the assistant ID
    with open(ASSISTANT_ID_FILE, "w") as f:
        json.dump({"assistant_id": assistant.id}, f)
    
    return assistant.id

# Create the assistant and save the ID (run this once)
create_and_save_assistant()