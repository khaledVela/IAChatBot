# Catálogo + Mini-Chat IA (Flask · LangChain · Azure OpenAI · Pinecone)

Proyecto web con:

* **Catálogo** de productos en `index.html`.
* **Mini-chat** embebido que habla con tu backend `/ask`.
* Flujo especial de **“soporte”**: cuando el usuario escribe `soporte`, el widget pide el **número del cliente** y envía **todo el chat como `.txt`** a `http://localhost:21465/api/enviar-archivo/cuenta1` vía **form-data** con autenticación por **API Key**.
* Al volver a hacer clic en el botón flotante verde, el chat **se reinicia desde cero** (historial, estados y UI).

---

## 1) Arquitectura rápida

```
Navegador
  ├─ index.html  (catálogo + mini-chat + flujo soporte)
  │   └─ POST http://localhost:21465/api/enviar-archivo/cuenta1  (FormData: numero, texto, documento)
  │         Headers: api-key: <tu_api_key>   ← (o x-api-key, según tu servicio)
  └─ POST /ask  (Flask) → LangChain (Azure OpenAI + Pinecone)
        ├─ Carga PDFs ./docs
        ├─ Embeddings en Pinecone
        └─ Respuesta: { answer }
```

---

## 2) Requisitos

* **Python 3.10+**
* Cuenta/recursos de **Azure OpenAI** (deployments para *embeddings* y *chat*).
* Cuenta de **Pinecone** (índice serverless).
* Servicio externo de soporte corriendo en `http://localhost:21465` que acepte:

  * `POST /api/enviar-archivo/cuenta1`
  * **form-data**: `numero` (string), `texto` (string), `documento` (file)
  * Header de API key (por ejemplo `api-key` o `x-api-key`)
  * **CORS** habilitado para tu origen y header personalizado.

### requirements.txt (sugerido)

```txt
Flask>=3.0
langchain>=0.2
langchain-community>=0.2
langchain-openai>=0.1
langchain-pinecone>=0.1
pinecone-client>=3.0
pypdf>=4.2
tiktoken>=0.7
python-dotenv>=1.0
```

---

## 3) Estructura de carpetas (ejemplo)

```
tu_proyecto/
├─ app.py
├─ docs/                      # tus PDFs para el RAG
│  └─ ...pdf
├─ templates/
│  └─ index.html              # catálogo + mini-chat (tu archivo)
└─ static/
   └─ images/
      ├─ redes.jpg
      ├─ sistemas.jpg
      └─ financiera.jpg
```

> Si prefieres rutas tipo `/servidor-rag/images/...`, usa `static_url_path="/servidor-rag"` o mapea esa carpeta como estático (ver §7).

---

## 4) Variables de entorno

Crea un `.env` (o exporta en tu shell):

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=xxxxxxxxxxxxxxxx
AZURE_OPENAI_API_BASE=https://<tu-recurso>.openai.azure.com
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002   # dim=1536
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o                         # tu deployment

# Pinecone
PINECONE_API_KEY=xxxxxxxxxxxxxxxx
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# (Opcional) SMTP si alguna vez activas envíos por correo
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
SMTP_FROM=
SUPPORT_EMAIL_TO=
```

> Asegúrate de que la **dimensión del índice** Pinecone coincida con tu modelo de embeddings. Para `text-embedding-ada-002` es **1536**.

---

## 5) Backend Flask (`app.py`)

Tu servidor expone:

* `POST /ask` — recibe `{ "question": "..." }`, busca en Pinecone y responde con `{ "answer": "..." }`.
* `GET /` — sirve `templates/index.html`.

Pseudocódigo (coincide con tu app real):

```python
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

# 1) Carga PDFs ./docs
loader = DirectoryLoader("./docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# 2) Embeddings + Pinecone
emb = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "langchain-demo2"
if not pc.has_index(index_name):
    pc.create_index(name=index_name, dimension=1536, metric="cosine",
                    spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD","aws"),
                                        region=os.getenv("PINECONE_REGION","us-east-1")))
store = PineconeVectorStore(index_name=index_name, embedding=emb)
store.add_documents(docs)

# 3) LLM de QA
llm = AzureChatOpenAI(
    temperature=0,
    azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
)
prompt = ChatPromptTemplate.from_template(
    "Usa el contexto para responder la pregunta de forma clara.\n\nContexto:\n{context}\n\nPregunta:\n{input}"
)
chain = create_stuff_documents_chain(llm, prompt)

@app.route("/ask", methods=["POST"])
def ask():
    q = (request.get_json() or {}).get("question", "").strip()
    if not q:
        return jsonify({"error":"No question provided"}), 400
    ctx = store.similarity_search(q, k=3)
    ans = chain.invoke({"input": q, "context": ctx})
    text = str(ans).strip() or "Lo siento, no encontré información sobre eso."
    return jsonify({"question": q, "answer": text})

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
```

---

## 6) Frontend (`templates/index.html`)

* Catálogo + widget.
* Botón flotante **reinicia** el chat cada vez que se pulsa.
* Palabra clave `soporte` activa el flujo humano:

  1. el bot pide **número del cliente**;
  2. envía **FormData** con `numero` (fijo), `texto` (número del cliente) y `documento` (chat en `.txt`) a `SUPPORT_URL`;
  3. añade el header con tu **API Key**.

Personaliza estas constantes en el `<script>`:

```js
const SUPPORT_URL  = 'http://localhost:21465/api/enviar-archivo/cuenta1';
const SUPPORT_NUM  = '59176392492';                    // fijo
const API_KEY      = '<TU_API_KEY>';                   // la que te dieron
const HEADER_NAME  = 'api-key';                        // o 'x-api-key'
```

**Importante (CORS):** tu servicio en `:21465` debe permitir

* el **origen** de tu web (p. ej. `http://localhost:8000`),
* el header **`api-key`** (o `x-api-key`) en `Access-Control-Allow-Headers`.

---

## 7) Imágenes estáticas del catálogo

### Opción recomendada (Flask static):

* Pon tus imágenes en `static/images/`.
* En `index.html` (Jinja), usa:

  ```html
  <img src="{{ url_for('static', filename='images/redes.jpg') }}" alt="Producto 1">
  ```

### O bien conservar `/servidor-rag/images/...`:

* Inicia Flask con `static_folder="servidor-rag"`, `static_url_path="/servidor-rag"`.
* Mantén `<img src="/servidor-rag/images/redes.jpg">`.

---

## 8) Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Crea/rellena .env
python app.py
```

Abre: **[http://localhost:8000](http://localhost:8000)**

---

## 9) API de soporte (lo que envía el front)

**URL:** `POST http://localhost:21465/api/enviar-archivo/cuenta1`
**Headers:** `api-key: <tu_api_key>` (o `x-api-key`, según tu servicio)
**Body (multipart/form-data):**

* `numero`: `59176392492` (constante)
* `texto`: `<número_del_cliente>` (el usuario lo escribe en el chat)
* `documento`: archivo `.txt` con el historial del chat (un archivo por petición)

**Ejemplo de líneas del `.txt`:**

```
[2025-08-23T21:15:02.123Z] USUARIO: soporte
[2025-08-23T21:15:03.456Z] BOT: Perfecto. Indícame tu número (formato internacional, p. ej. 5917XXXXXXXX)...
[2025-08-23T21:15:08.001Z] USUARIO: 5917XXXXXXX
...
```

---

## 10) Personalización útil

* **Reiniciar también al cerrar con “X”**: llama a `resetChat()` dentro del handler del botón cerrar antes de ocultar.
* **Cambiar la palabra clave** del flujo humano: en el JS busca `text.toLowerCase() === 'soporte'`.
* **Validación del número**: por defecto acepta 8–15 dígitos. Para forzar prefijo `591`, ajusta la función `looksLikePhone`.

---

## 11) Problemas típicos y soluciones

* **CORS bloquea la petición de soporte**
  Asegúrate de que tu servicio de `:21465`:

  * Responde al preflight `OPTIONS`.
  * Incluye `Access-Control-Allow-Origin: http://localhost:8000` (o `*`).
  * Incluye `Access-Control-Allow-Headers: api-key, content-type` (o `x-api-key`).
* **`401/403` en soporte**
  Header mal escrito (`api-key` vs `x-api-key`) o API key incorrecta.
* **`OpenAI 401/404/429`**
  Revisa `AZURE_OPENAI_*`, deployments y cuota de tu recurso.
* **`Pinecone dimension mismatch`**
  Si cambias a otro embedding (p. ej. `text-embedding-3-large`), también cambia la **dimensión** del índice.
* **Imágenes no cargan**
  Verifica rutas, `static_url_path` y que el servidor sirva los estáticos (no abras `index.html` directamente con `file://`).

---

## 12) Licencia

Este README asume uso educativo/demostrativo. Revisa licencias de terceros (FontAwesome CDN, Tailwind CDN si lo usas, etc.) y términos de Azure OpenAI y Pinecone antes de ir a producción.

---

¿Quieres que te genere también el `requirements.txt`, un `.env.example` y un `Procfile` (Gunicorn) para desplegarlo? Puedo dejarlos listos para copiar-pegar.
