import openai
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Usamos la clave de OpenAI
openai.api_key = "OPENAI_API_KEY"

# Conexión a MongoDB Atlas
client = MongoClient("URI_MONGODB")
db = client.TFM
collection = db.IMDB

# Carga el modelo para generar embeddings
modelo = SentenceTransformer("BAAI/bge-m3")

# Función para generar un embedding de una consulta del usuario
def generar_embedding_query(query):
    return modelo.encode(query).tolist()

# Función para realizar la búsqueda en MongoDB usando Vector Search
def buscar_documentos(query):
    query_embedding = generar_embedding_query(query)

    resultados = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": 4
            }
        }
    ])
    
    return list(resultados)

# Función para generar un contexto a partir de toda la información disponible en los documentos
def generar_contexto_dinamico(documentos):
    contextos = []
    for doc in documentos:
        contexto = []
        for key, value in doc.items():
            if key != '_id' and key != 'embedding':
                contexto.append(f"{key}: {value}")
        contextos.append("\n".join(contexto))
    return "\n\n".join(contextos)

# Función para generar la respuesta del chatbot utilizando OpenAI
def generar_respuesta(query, documentos_relevantes, historial):
    if documentos_relevantes:
        context = generar_contexto_dinamico(documentos_relevantes)

        historial.append({"role": "user", "content": f"Información relevante:\n{context}\n Responde a la consulta: {query}"})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=historial,
            max_tokens=150
        )

        respuesta = completion.choices[0].message['content'].strip()
    else:
        respuesta = "Sintiéndolo mucho, no puedo darte esa información porque estoy aprendiendo y aún no sé la respuesta."

    historial.append({"role": "assistant", "content": respuesta})

    return respuesta
