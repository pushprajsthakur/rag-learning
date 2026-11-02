import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()

client = genai.Client()

# response = client.models.generate_content(
#     model="gemini-3.8-flash",
#     contents="What is the capital of France?",
#     config=types.GenerateContentConfig(
#         system_instruction="You are a helpful assistant.",
#        # Turn off automatic function calling to silence the warning
#         automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
#     )
# )
# print(response.text)

emb_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="your text embedding goes here",
    )
print("Embedding Response: ",emb_response)

embedding_vector = emb_response.embeddings[0].values
print("Dimensions: ",len(embedding_vector))

# print(list(embedding_vector))
