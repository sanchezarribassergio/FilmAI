from flask import Flask, render_template, request, jsonify
import openai
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Usamos la clave de OpenAI
openai.api_key = "OPENAI_API_KEY"

# Conexión a MongoDB Atlas
client = MongoClient("URI_MONGODB")
db = client.TFM
collection = db.IMDB

# Cargamos el modelo para generar embeddings
modelo = SentenceTransformer("BAAI/bge-m3")

historial = [{"role": "system", "content": "Responde solo basándote en la información proporcionada. Si no tienes suficiente información, indica que no puedes responder porque estás en fase beta"}]

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
                "numCandidates": 1000,
                "limit": 5
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
def generar_respuesta(query, documentos_relevantes):
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_input = request.form['msg']
    documentos_relevantes = buscar_documentos(user_input)
    respuesta = generar_respuesta(user_input, documentos_relevantes)
    return jsonify(respuesta=respuesta)

if __name__ == "__main__":
    app.run(debug=True)
