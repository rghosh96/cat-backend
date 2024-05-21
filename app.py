from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
import pandas as pd
import numpy as np
import tiktoken
import pytz
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import csv
import json

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

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-3.5-turbo"

df  = pd.read_csv('dataset.csv')


# CAT_PROMPTS = {
#     "CAT_prompt_control": "Control",
#     "CAT_prompt_approxmiation": "Given the following USER QUERY and RESPONSE, adjust the RESPONSE (while delivering the same information) to follow these guidelines in 75 words or less: Adjust your language style so that it matches the user query's tone, sentence query, choice of expressions and words, and level of formality/casualness.",
#     "CAT_prompt_interpretability": "Given the following USER QUERY and RESPONSE, adjust the RESPONSE (while delivering the same information) to follow these guidelines in 75 words or less: Ensure the response is clear and easily understandable based on the user’s level of understanding expressed by the user’s query. Define any technical/medical jargon words. Use easy to understand language and simple phrasing.",
#     "CAT_prompt_interpersonalcontrol": "Given the following USER QUERY and RESPONSE, adjust the RESPONSE (while delivering the same information) to follow these guidelines in 75 words or less: Provide resources or references, or other healthcare resources the user can contact for further exploration to give the user more control and agency in the conversation. Empower the user to take responsibility of their own health by encouraging them to make their own decisions. Finally, ensure to solicit the user's input to guide the direction of the conversation, such as asking what to talk about.",
#     "CAT_prompt_discoursemanagement": "Given the following USER QUERY and RESPONSE, adjust the RESPONSE (while delivering the same information) to follow these guidelines in 75 words or less: Express patience in the middle of the response (for example, you can pause and say something like 'OK?'). Express importance of the user's engagement/input or the relevance/validity of the user's query for understanding the current topic better. Finally, suggest open-ended questions the user can ask: for example, 'You can ask me about X or Y next’.",
#     "CAT_prompt_emotionalexpression": "Given the following USER QUERY and RESPONSE, adjust the RESPONSE (while delivering the same information) to follow these guidelines in 75 words or less: Incorporate emotional cues or expressions in your response to reflect empathy, reassurance, and/or genuine support based on the user's query. Respond in a way that makes it clear that the user's question is heard, and that they are cared about."
# }

CAT_PROMPTS = {
    "CAT_prompt_control": "Control",
    "CAT_prompt_approxmiation": "Given the following USER_QUERY and ORIGINAL_RESPONSE: \n(1) Score how casual/formal the USER_QUERY is using the Formality Scale (Biber et al., 1999). \n(2) Adjust the ORIGINAL_RESPONSE (while delivering the same information) in 75 words or less based on the score in (1):\n- Adjust your language style so that it matches the user query's tone, sentence query, choice of expressions and words, and level of formality/casualness; use verbal mimicry. \nStructure your response as a JSON where the keys are Score, Adjusted_Response, and Highlighted_Response, and the values are as follows: For the Score key, provide the Formality Scale result from (1). For the Adjusted_Response key, provide the response text based on the Formality Scale score provided from (2). For the Highlighted_Response key, list the specific text from the Adjusted_Response key that was added/changed from the ORIGINAL_RESPONSE based on the Formality Scale in array form.",
    "CAT_prompt_interpretability": "Given the following USER_QUERY and ORIGINAL_RESPONSE: \n(1) Score the health literacy level of the USER_QUERY using the Flesch-Kincaid Grade Level. \n(2) Adjust the ORIGINAL_RESPONSE (while delivering the same information) in 75 words or less based on the health literacy score in (1):\n- Adjust the response such that it is clear and easily understandable, and concise and to the point; define any technical/medical jargon words specific to clinical trials if the score is low; use easy to understand language and simple phrasing; use simple metaphors and analogies if health literacy score is low. \nStructure your response as a JSON where the keys are Score, Adjusted_Response, and Highlighted_Response, and the values are as follows: For the Score key, provide the Flesch-Kincaid Grade Level result from (1). For the Adjusted_Response key, provide the response text based on the Flesch-Kincaid Grade Level score provided from (2). For the Highlighted_Response key, list the 3 most simplified phrases directly from the Adjusted_Response key that was used to simplify the ORIGINAL_RESPONSE based on the Flesch-Kincaid Grade Level in array form.",
    "CAT_prompt_interpersonalcontrol": "Given the following USER_QUERY and ORIGINAL_RESPONSE: \n(1) Score the level of control/power the user has in the conversation and making health decisions based on the USER_QUERY using the Conversational Control Scale (CCS). \n(2) Adjust the ORIGINAL_RESPONSE (while delivering the same information) in 75 words or less based on the control/power score in (1):\n  If the score is low: \n  1. Assume the role of a peer to give more power to the user.\n    2. Empower the user to take responsibility of their own health.\n    3. End by soliciting user's input to guide the direction of the conversation, such as asking what to talk about or asking for more relevant information from the user. \nStructure your response as a JSON where the keys are Score, Adjusted_Response, and Highlighted_Response, and the values are as follows: For the Score key, provide the Conversational Control Scale result from (1). For the Adjusted_Response key, provide the response text based on the Conversational Control Scale score provided from (2). For the Highlighted_Response key, list the specific text from the Adjusted_Response key that was added/changed from the ORIGINAL_RESPONSE based on the Conversational Control Scale in array form.",
    "CAT_prompt_discoursemanagement": "Given the following USER_QUERY and ORIGINAL_RESPONSE: \n(1) Score the conversational needs of the USER_QUERY using the Health Information Needs Scale. \n(2) Adjust the ORIGINAL_RESPONSE (while delivering the same information) in 75 words or less based on the conversational needs score in (1):\n- If the score is low:\n    1. Use backchannels/supportive phrases such as “I see” or “Right”.\n    2. Encourage the user by complimenting their question or re-summarizing their question to show you're listening and interested. \n   3. End by suggesting related open-ended topics the user can ask: for example, 'You can ask me about X or Y next'. \nStructure your response as a JSON where the keys are Score, Adjusted_Response, and Highlighted_Response, and the values are as follows: For the Score key, provide the Health Information Needs Scale result from (1). For the Adjusted_Response key, provide the response text based on the Health Information Needs Scale score provided from (2). For the Highlighted_Response key, list the specific text from the Adjusted_Response key that was added/changed from the ORIGINAL_RESPONSE based on the Health Information Needs Scale in array form.",
    "CAT_prompt_emotionalexpression": "Given the following USER_QUERY and ORIGINAL_RESPONSE:\n(1) Perform a sentiment analysis of the USER_QUERY.\n(2) Adjust the ORIGINAL_RESPONSE (while delivering the same information) in 75 words or less based on the sentiment analysis of the user's overall sentiment:\n- If the overall sentiment of the user's queries is negative or mixed:\n  - Incorporate emotional cues or expressions in your response to reflect empathy, reassurance, and/or genuine support.\n  - Respond in a way that makes it clear that the user's questions are heard and that they are cared about.\n- If the overall sentiment of the user's queries is positive or neutral:\n  - Use positive statements that commend the user’s interest and encourage the user's engagement.\n- Regardless of sentiment rating, provide a phrase/statement letting the user know you are there to support them. \n\nStructure your response as a JSON where the keys are Score, Adjusted_Response, and Highlighted_Response, and the values are as follows: For the Score key, provide the sentiment analysis result from (1). For the Adjusted_Response key, provide the response text based on the sentiment analysis provided from (2). For the Highlighted_Response key, list the specific text from the Adjusted_Response that reflects the additions/changes you made based on the sentiment analysis score that makes the Adjusted_Response different from the ORIGINAL_RESPONSE in array form."
}


# Check if "tokens" column exists in the DataFrame
if 'tokens' not in df.columns:
    # Initialize the encoder for the GPT-4 model
    enc = tiktoken.encoding_for_model("gpt-4")

    # List to store the number of tokens per section
    tokens_per_section = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Encode the combined text of "title" and "heading" columns
        tokens = enc.encode(row['title'] + ' ' + row['heading'])
        # Append the number of tokens to the list
        tokens_per_section.append(len(tokens))

    # Add a new column with the number of tokens per section
    df['tokens'] = tokens_per_section
    # Save the updated DataFrame back to a new CSV file
    df.to_csv('dataset.csv', index=False)

df.head()
df = df.set_index(["title", "heading"])

## This code was written by OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = client.embeddings.create(
      model=model,
      input=text
    )
    return result.data[0].embedding

# def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
#     """
#     Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
#     Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
#     """
#     return {
#         idx: get_embedding(r.heading) for idx, r in df.iterrows()
#     }


### Function to compute embeddings for your entire CSV
def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }


# document_embeddings  = compute_doc_embeddings(df)
# entries = list(document_embeddings.items())

# ### Prints out an example entry for sanity check
# example_entry = list(document_embeddings.items())[0]
# print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

# ## The embeddings calculation creates a vector of length 1536 (0 - 1535)
# ## So our CSV will be 1538 columns, one for title, one for heading, and 1536 for the vector's we generate
# headers = ["title", "heading"]
# for x in range(1536):
#     headers.append(x)

# ### CHANGE: Name of the file you want to save it to
# csv_file = "cat_faq_embeddings.csv"

# ### This writes your embeddings into a new CSV. You need both this CSV and the CSV you generated in create_dataset.py
# ### In order to do the FAQ Bot.
# with open(csv_file, 'w', encoding = 'utf-8', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(headers)

#     for items in entries:
#         title, heading = items[0]
#         values = items[1]
#         row = [title,heading]
#         row = row + values
#         writer.writerow(row)

document_embeddings = {}
with open('cat_faq_embeddings.csv', 'r') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)
        
        # Skip the header row if it exists
        next(reader, None)
        
        # Iterate through each row in the CSV
        for row in reader:
            # Extract the key tuple from the row
            key = (row[0], row[1])
            
            # Extract the list of floats from the row
            values = [float(x) for x in row[2:]]
            
            # Add the key-value pair to the dictionary
            document_embeddings[key] = values


# An example embedding:
# example_entry = list(document_embeddings.items())[0]
# print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

## This code was written by OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    return np.dot(np.array(x), np.array(y))

def order_by_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"
    

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_by_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
        
    return chosen_sections

def answer_with_gpt(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    lastThreeTurns,
    first: bool = False,
    show_prompt: bool = False,
) -> str:
    print("IN ANSWER W GPT")
    print(query)
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    if show_prompt:
        print(prompt)


    print("FIRST API CALL TO GET INFO")
    context= ""
    for article in prompt:
        context = context + article 
    messages = []
    if first == False:
        messages = [
            {
                "role": "system",
                "content": "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only respond by using the provided context and conversation history. If a user says something out of the provided context, say 'I don't know'. If the user says something along the lines of 'yes' or 'no', refer to the conversation history to answer. Keep your response to 75 words or less. Also, categorize the user's query as relating to one of the following topics, providing only the number: (1) Safety in Clinical Trials, (2) Understanding and Comfort with the Clinical Trial Process, (3) Logistical, Time, and Financial Barriers to Participation, (4) Risks and Benefits of Clinical Trials, (5) Awareness and Information Accessibility. Structure your response as a JSON where the keys are Topic and Response, and the values are as follows: For the Topic key, provide the numerical categorization. For the Response key, provide the response text you generated to answer the user's query."
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only respond by using the provided context and conversation history. If a user says something out of the provided context, say 'I don't know'. If the user says something along the lines of 'yes' or 'no', refer to the conversation history to answer. Keep your response to 75 words or less."
            }
        ]

    messages = messages + lastThreeTurns

    context = context + '\n\n --- \n\n + ' + query
    messages.append({"role" : "user", "content":context})
   

    # print("AB TO ANSWER WITH GPT, MESSAGES IS: ", messages)

    response = client.chat.completions.create(
        model=COMPLETIONS_MODEL,
        messages=messages,
        max_tokens=200,
        temperature=0
        )
    
    # print("AFTER FIRST CALL:", response.choices[0].message.content)

    if first == False:
        fullResponse = response.choices[0].message.content
        # print(fullResponse)
        response_dict = json.loads(fullResponse)
        # print(type(response_dict))
        return response_dict['Response'], response_dict['Topic']
    else:
        return response.choices[0].message.content, 0

def tailor_with_gpt(
    userQuery: str,
    agentResponse: str,
    lastThreeTurns,
    selected_prompt: str,  # Pass selected_prompt as an argument
) -> str:

    finalPrompt = "{}\nUSER_QUERY: {}\nORIGINAL_RESPONSE: {}".format(CAT_PROMPTS[selected_prompt],userQuery, agentResponse)


    messages = [{"role" : "system", "content": finalPrompt}]

    print("AB TO ANSWER WITH GPT, MESSAGES IS: ", messages)

    tailoredResponse = client.chat.completions.create(
        model=COMPLETIONS_MODEL,
        messages=messages,
        max_tokens=200,
        temperature=0
        )
    
    # print(agentResponse)
    print("HERE IS RESPONSE")
    print(tailoredResponse.choices[0].message.content)
    response_dict = json.loads(tailoredResponse.choices[0].message.content)
    # print(response_dict)
    print("AB TO TRY TO RESPOND")
    return response_dict['Adjusted_Response'], response_dict['Highlighted_Response']
   

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
    data = await request.json()
    print(data)
    message = data['message']
    condition = data['condition']
    lastThreeTurns = data['lastThreeTurns']
    first = data['first']
    returnControl = data['returnControl']
    firstItem = False
    if first == 'true':
        firstItem = True

    selected_value = CAT_PROMPTS[list(CAT_PROMPTS.keys())[int(condition)]]
    selected_prompt = next(key for key, value in CAT_PROMPTS.items() if value == selected_value)


    # print("IN CHAT API, SELECTED PROMPT IS: " + selected_prompt)
    lastThreeTurns.reverse()
    # print("LAST THREE TURNS ARE", lastThreeTurns)

    text_response_info, topic = answer_with_gpt(message, df, document_embeddings, lastThreeTurns, firstItem)
    if selected_prompt == "CAT_prompt_control":
        text_response_CAT = text_response_info
        highlights = []
    else:
        print("AB TO GET TAILOR WITH GPT")
        text_response_CAT, highlights = tailor_with_gpt(message, text_response_info, lastThreeTurns, selected_prompt)


    audio_response = generateAudio(text_response_CAT)
    audio_base64 = base64.b64encode(audio_response).decode('utf-8')
    audio_data_url = f"data:audio/wav;base64,{audio_base64}"
    # audio_data_url = "boop"
    print("SENDING INFO BACK TO FRONT END")

    if returnControl == 'false':
        return JSONResponse({'message': text_response_CAT, 'highlights': highlights, 'audio': audio_data_url, 'topic': topic})
    else:
        return JSONResponse({'message': text_response_CAT, 'highlights': highlights, 'audio': audio_data_url, 'topic': topic, 'controlMessage': text_response_info})