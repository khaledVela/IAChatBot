import os
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# ========================
# CONFIGURACIÓN
# ========================

os.environ["AZURE_OPENAI_API_KEY"] = "TU_API_KEY"
os.environ["AZURE_OPENAI_API_BASE"] = "https://TU-RESOURCE.openai.azure.com"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "text-embedding-ada-002"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"
os.environ["PINECONE_API_KEY"] = "TU_PINECONE_KEY"

# ========================
# CARGA DE DOCUMENTOS (TODOS LOS PDFs DE ./docs)
# ========================
print("📂 Cargando documentos de ./docs ...")
loader = DirectoryLoader("./docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"✔️ Se cargaron {len(documents)} documentos en total")

# Dividir en chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"📑 Divididos en {len(docs)} chunks")

# ========================
# EMBEDDINGS + PINECONE
# ========================
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "langchain-demo2"

# Crear índice si no existe
if not pc.has_index(index_name):
    print("🛠️ Creando índice en Pinecone ...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # embeddings de ada-002
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Conectar vectorstore
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Subir documentos
print("⬆️ Subiendo embeddings a Pinecone ...")
vectorstore.add_documents(docs)
print("✔️ Embeddings almacenados en Pinecone")

# ========================
# MODELO DE QA
# ========================
llm = AzureChatOpenAI(
    temperature=0,
    azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    deployment_name="gpt-4o"  # ⚠️ ajusta a tu deployment
)

# Definir prompt y chain
prompt = ChatPromptTemplate.from_template(
    "Usa el contexto para responder la pregunta de forma clara.\n\nContexto:\n{context}\n\nPregunta:\n{input}"
)
chain = create_stuff_documents_chain(llm, prompt)

# ========================
# FLASK SERVER
# ========================
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question", "")

    if not query:
        return jsonify({"error": "No question provided"}), 400

    # Buscar documentos similares
    retrieved_docs = vectorstore.similarity_search(query, k=3)

    # Ejecutar cadena de QA
    result = chain.invoke({"input": query, "context": retrieved_docs})

    return jsonify({"question": query, "answer": result})

from flask import render_template

@app.route("/chat", methods=["GET"])
def chat():
    return render_template("chat.html")
from flask import render_template

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ========================
# MAIN
# ========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
