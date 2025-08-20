from flask import Flask, request, jsonify
import os
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import AzureChatOpenAI
from pinecone import Pinecone, ServerlessSpec

# ========================
# CONFIGURACIÓN
# ========================


openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
openai.api_base = os.environ["AZURE_OPENAI_API_BASE"]
openai.api_type = "azure"
openai.api_version = os.environ["AZURE_OPENAI_API_VERSION"]

# ========================
# PREPARAR VECTORSTORE
# ========================
pdf_path = "./docs/corollacross_brochure.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "langchain-demo2"

if pc.has_index(index_name):
    pc.delete_index(index_name)

# Crear de nuevo con la dimensión correcta (1536)
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
vectorstore.add_documents(docs)

llm = AzureChatOpenAI(
    temperature=0,
    openai_api_base=os.environ["AZURE_OPENAI_API_BASE"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    deployment_name=os.environ.get("AZURE_OPENAI_GPT4_MODEL_NAME", "gpt-4o")
)

chain = load_qa_chain(llm, chain_type="stuff")

# ========================
# SERVIDOR FLASK
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
    result = chain.run(input_documents=retrieved_docs, question=query)

    return jsonify({"question": query, "answer": result})

from flask import render_template

@app.route("/chat", methods=["GET"])
def chat():
    return render_template("chat.html")
from flask import render_template

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

