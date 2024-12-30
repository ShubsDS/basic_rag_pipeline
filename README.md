# basic_rag_pipeline
This project is a basic RAG pipeline built for the LMI onboarding project

# Setup instructions
Aside from all of the code in this repository, you will need a separate file, named api-key.txt, containing only your openai api key
Additionally, you will need to have numpy, opanai, and chromadb installed 

# Process explanation

### Reading in the data and processing
The first step to the process is simply reading in the data that we have decided to use as our context.
In this case, that would be the 3 text documents regarding hades 2, minecraft pvp, and chopin's fantasie impromptu.

We then use OpenAI's text-embedding-3-small model to create vector representations of all of the tokens in the 3 documents, and then save those embeddings using numpy.

### Storing Data in ChromaDB
The ChromaDB library lets us store our embeddings in clients, allowing us to work with them easier. We create a new chromaDB persistent client and store our embeddings in that client

### Querying 

We then make a query, find which documents in our collection that query is most similar to, and we add those documents to the context of our prompt. Finally returning an answer to the original question.
