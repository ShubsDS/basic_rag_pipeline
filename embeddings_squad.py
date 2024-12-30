import json
from sentence_transformers import SentenceTransformer
import chromadb


with open('train-v2.0.json', 'r') as f:
    squad_data = json.load(f)

dataset = []
for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            dataset.append({
                'question': qa['question'],
                'context': context,
                'is_impossible': qa['is_impossible'],
                'answers': qa['answers']
            })
print("completed reading dataset")


model = SentenceTransformer('all-MiniLM-L6-v2')

questions = [item['question'] for item in dataset]
contexts = [item['context'] for item in dataset]

question_embeddings = model.encode(questions, convert_to_tensor=True)
context_embeddings = model.encode(contexts, convert_to_tensor=True)


chroma_client = chromadb.PersistentClient()
collection_name = "squad_collection"

collection = chroma_client.get_or_create_collection(name = collection_name)


collection.add(
    documents = contexts,
    embeddings = context_embeddings.tolist(),
    ids=[f"{i}" for i in range(len(dataset))]
)