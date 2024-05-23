# This script helps create various CAT chatbots by prompting the user to select a CAT strategy.
# Based on the selection, the script constructs an assistant with tailored communication prompts.
# The assistant is created using the OpenAI API and its ID is saved for future reference.
# ONLY NEED TO RUN ONCE IF CAT DEFINITIONS/BASE PROMPT CHANGES

import json
import os
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-4-turbo"

# Path to store the assistant ID
ASSISTANT_ID_FILE = "assistant_ids.json"

user_cat_input = 0

# CAT-specific instructions:
# 0 - control
# 1 - approximation
# 2 - interpretability
# 3 - interpersonal control
# 4 - discourse management
# 5 - emotional expression
CAT_PROMPTS = [
    "Control",
    "Adjust your language style in your response so that it matches the user's tone, sentence query, choice of expressions and words, and level of formality/casualness; use verbal mimicry.",
    "Based on how familiar it seems the user is with clinical trials and health, adjust the response such that it is clear and easily understandable; define any technical/medical jargon words specific to clinical trials; use easy to understand language and simple phrasing; use simple metaphors and analogies if possible.",
    "Assume the role of a peer to give more power to the user; Empower the user to take responsibility of their own health; solicit user's input to guide the direction of the conversation.",
    "Use backchannels/supportive phrases; Encourage the user by complimenting their question or re-summarizing their question to show you're listening and interested; suggesting related open-ended topics the user can ask.",
    "Incorporate emotional cues or expressions in your response to reflect empathy, reassurance, and/or genuine support."
]
CAT_STRATEGY = [
    "Control",
    "approximation",
    "interpretability",
    "interpersonal_control",
    "discourse_management",
    "emotional_expression"
]

# Function to create the assistant and save its ID
def create_and_save_assistant():
    # Base prompt constant
    BASE_PROMPT = (
        "You are a virtual healthcare assistant whose purpose is to educate, inform, and raise awareness about clinical trials enrollment and participation. "
        "Only use the uploaded file to answer the user's questions. Otherwise, let the user know you can't answer. "
        + CAT_PROMPTS[user_cat_input] +
        " Keep your response to 75 words or less.\n"
        "Also, categorize the user's query as relating to one of the following topics, providing only the number: "
        "(1) Safety in Clinical Trials, (2) Understanding and Comfort with the Clinical Trial Process, "
        "(3) Logistical, Time, and Financial Barriers to Participation, (4) Risks and Benefits of Clinical Trials, "
        "(5) Awareness and Information Accessibility. If none of them, categorize as (0). Structure your response as a JSON where the keys are Topic and Response, "
        "and the values are as follows: For the Topic key, provide the numerical categorization. For the Response key, provide the response text you generated to answer the user's query."
    )
    print(BASE_PROMPT)
    assistant = client.beta.assistants.create(
        name=CAT_STRATEGY[user_cat_input]+" CAT bot",
        instructions=BASE_PROMPT,
        model=GPT_MODEL,
        tools=[{"type": "file_search"}],
    )

    # Load the saved vector store ID
    with open("vector_store_id.txt", "r") as f:
        vector_store_id = f.read().strip()

    print(vector_store_id)
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
    )

    # Save the assistant ID
    with open(ASSISTANT_ID_FILE, "a") as f:
        assistant_key = CAT_STRATEGY[user_cat_input] + "_assistant_id"
        json.dump({assistant_key: assistant.id}, f)
    
    return assistant.id

# Get user input
user_cat_input = input("Create CAT Assistant. \n0 - Control\n1 - Approximation\n2 - Interpretability\n3 - Interpersonal Control\n4 - Discourse Management\n5 - Emotional Expression\nEnter one of the numbers above: ")
print("You entered:", user_cat_input)
user_cat_input = int(user_cat_input)
print("Generating CAT Assistant:", CAT_STRATEGY[user_cat_input], CAT_PROMPTS[user_cat_input])

# Create the assistant and save the ID (run this once)
create_and_save_assistant()