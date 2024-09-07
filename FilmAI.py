import openai
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Usamos la clave de OpenAI
openai.api_key = "OPENAI_API_KEY"

# Conexión a MongoDB Atlas
client = MongoClient("URI_MONGODB")
db = client.TFM
collection = db.IMDB

# Cargamos el modelo para generar embeddings
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

# Función para iniciar la conversación con el chatbot
def iniciar_chatbot():
    print("¡Hola! Soy FilmAI, tu asistente personal. ¿En qué puedo ayudarte hoy?\n")

    historial = [{"role": "system", "content": "Responde basándote única y exclusivamente en la información proporcionada. Si no tienes suficiente información, indica que no puedes responder porque estás aprendiendo. Si te preguntan sobre Pokémon o cine, cíñete a la información que tienes disponible."}]

    while True:
        consulta = input("Tú: ")

        if consulta.lower() in ["salir", "exit", "adiós", "adios"]:
            print("FilmAI: ¡Gracias por tu compañía! ¡Hasta la próxima!")
            break

        documentos_relevantes = buscar_documentos(consulta)
        respuesta = generar_respuesta(consulta, documentos_relevantes, historial)
        print(f"FilmAI: {respuesta}")
        print("FilmAI: ¿Hay algo más en lo que te pueda ayudar? (si deseas terminar la conversación, di 'adiós')\n")

if __name__ == "__main__":
    iniciar_chatbot()