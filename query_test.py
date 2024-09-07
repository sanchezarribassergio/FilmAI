import pymongo
import requests
from typing import List
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import heapq

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# Cargamos el modelo generador de embeddings
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
model = AutoModel.from_pretrained('BAAI/bge-m3')

# Conexión a MongoDB Atlas
client = pymongo.MongoClient("URI_MONGODB")
db = client.TFM
collection = db.IMDB

def generate_embedding(text: str) -> list[float]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings


# Búsqueda de prueba
query = "Película que trate sobre un club de poetas"

results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": generate_embedding(query),
            "path": "embedding",
            "index": "vector_index",
            "numCandidates": 100,
            "limit": 4
        }
    }
])

#print(f"Search: {query}\n")
for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["overview"]},\nReleased Year: {document["release_date"]},\nDocument: {document["Director"]} \n')