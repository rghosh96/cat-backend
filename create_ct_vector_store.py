# The create_ct_vector_store.py script is designed to upload a PDF file containing clinical trials information to OpenAI's vector store, 
# save the vector store ID for future reuse, and update an OpenAI assistant's tool resources with this vector store ID.
# This script effectively automates the process of setting up and maintaining a vector store for clinical trials data, 
# making it easily reusable and integrable with OpenAI's assistant functionalities.
# RUN ONLY ONCE TO GET ID

import os
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

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

    # Save the vector store ID for reuse
    vector_store_id = vector_store.id
    with open("vector_store_id.txt", "w") as f:
        f.write(vector_store_id)

    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch.status)
    print(file_batch.file_counts)

finally:
    # Close file streams
    for file_stream in file_streams:
        file_stream.close()