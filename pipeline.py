import openai
from sentence_transformers import SentenceTransformer
import sys

def in_venv():
    return sys.prefix != sys.base_prefix

with open('api-key.txt') as f:
    openai.api_key = f.read()

