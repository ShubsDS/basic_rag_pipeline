from openai import OpenAI
from sentence_transformers import SentenceTransformer
import sys
import numpy as np
import chromadb
from chromadb.utils import embedding_functions




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


model = SentenceTransformer('all-MiniLM-L6-v2')

# def get_embedding(text, model="text-embedding-3-small"):
#    text = text.replace("\n", " ")
#    return model.encode

for document in documents:
    document.replace("\n", " ")

try:
    document_embeddings = np.load('document_embeddings.npy', allow_pickle=True)
except FileNotFoundError:
    document_embeddings = model.encode(documents)
    np.save('document_embeddings.npy', document_embeddings)




chroma_client = chromadb.PersistentClient()

# Create or get a collection in ChromaDB
collection_name = "basic_collection"

#chroma_client.delete_collection(name = collection_name)

collection = chroma_client.get_or_create_collection(name = collection_name)



# Add documents and their embeddings to the collection
collection.add(
    documents=documents,               # Original documents
    embeddings=document_embeddings.tolist(),    # Embeddings as list
    metadatas=[{"id": i} for i in range(len(documents))],  # Metadata (optional)
    ids=[f"doc_{i}" for i in range(len(documents))]        # Unique IDs for each document
)
print('complete')

