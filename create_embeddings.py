from openai import OpenAI
from sentence_transformers import SentenceTransformer
import sys
import numpy as np
import chromadb
from chromadb.utils import embedding_functions


with open('api-key.txt') as f:
    key = f.read()
client = OpenAI(api_key=key)

documents = []
s = ""
with open('hadestwo.txt') as f:
    for line in f:
        s += line
documents.append(s)

s = ""
with open('mcpvp.txt') as f:
    for line in f:
        s += line
documents.append(s)

s = ""
with open('fantasie.txt', errors = 'ignore') as f:
    for line in f:
        s += line
documents.append(s)



def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

try:
    document_embeddings = np.load('document_embeddings.npy', allow_pickle=True)
except FileNotFoundError:
    document_embeddings = [get_embedding(doc) for doc in documents]
    np.save('document_embeddings.npy', document_embeddings)




chroma_client = chromadb.PersistentClient()

# Create or get a collection in ChromaDB
collection_name = "basic_collection"

#chroma_client.delete_collection(name = collection_name)

collection = chroma_client.get_or_create_collection(name = collection_name)



# Add documents and their embeddings to the collection
for index, (doc, embedding) in enumerate(zip(documents, document_embeddings)):
    collection.add(
        ids=[f"doc_{index}"],
        documents=[doc],
        embeddings=[embedding]
    )

