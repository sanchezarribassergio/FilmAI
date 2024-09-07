from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Conexión a MongoDB Atlas
client = MongoClient("URI_MONGODB")

# Seleccionamos la base de datos y la colección
db = client.TFM
collection = db.IMDB

# Extraemos los documentos de la colección
documentos = list(collection.find({"embedding": {"$exists": False}}))

# Visualizamos algunos documentos para asegurarnos de que la conexión funciona
for doc in documentos[:5]:
    print(doc)

# Carga del modelo de procesamiento del lenguaje para generar embeddings
modelo = SentenceTransformer("BAAI/bge-m3")

# Función para crear una sola cadena de texto
def combinar_caracteristicas(doc):
    return " ".join([
        doc.get('Name', ''),
        doc.get('Type', ''),
        doc.get('Species', ''),
        doc.get('Height', ''),
        doc.get('Weight', ''),
        doc.get('Abilities', ''),
        doc.get('Catch rate', ''),
        doc.get('Base Friendship', ''),
        doc.get('Base Exp.', ''),
        doc.get('Growth Rate', ''),
        doc.get('Gender', ''),
        doc.get('HP', ''),
        doc.get('Attack', ''),
        doc.get('Defense', ''),
        doc.get('Sp. Atk', ''),
        doc.get('Sp. Def', ''),
        doc.get('Speed', '')
    ])

# Bucle para generar embeddings y almacenarlos en MongoDB Atlas
for doc in tqdm(documentos, desc="Generando embeddings", unit="documento"):
    texto_combinado = combinar_caracteristicas(doc)
    embedding = modelo.encode(texto_combinado).tolist()
    collection.update_one({'_id': doc['_id']}, {'$set': {'embedding': embedding}})