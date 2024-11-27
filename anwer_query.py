from openai import OpenAI
import chromadb

with open('api-key.txt') as f:
    key = f.read()

client = OpenAI(api_key=key)


chroma_client = chromadb.PersistentClient()

# Create or get a collection in ChromaDB
collection_name = "basic_collection"

collection = chroma_client.get_or_create_collection(name = collection_name)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

query = input("Enter your query: ")

query_embedding = get_embedding(query)

k = 1

# Search for similar documents
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=k
)

retrieved_docs = results['documents'][0]

total_context = "" 
for doc in retrieved_docs:
    total_context += doc
    total_context += "\n"



system_message = """ you are an expert in the fields of music and video games. You have especially studied the content and development of hades 2. You are additionally a minecraft pvp expert. Finally, you are a chopin specialist with great knowledge of his works.
"""
user_message = f""" 
Based on the following context, provide a detailed answer to this question

Context:
{total_context}

Question:
{query}

Answer:
"""

# user_message = f"""Question: {query}
# Answer: 
# """

# Print the entire prompt
print("\nPrompt:")
print(system_message)
print(user_message)

# Generate a response using the language model
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [{"role" : "system", "content": system_message},
    {"role" : "user", "content" : user_message}
    ],
    max_tokens=300,
    temperature=0.7
)

# Extract and print the answer
answer = response.choices[0].message.content
print("\nAnswer:")
print(answer)