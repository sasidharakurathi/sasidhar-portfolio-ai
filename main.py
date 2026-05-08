import os
import logging
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.memory import ChatMemoryBuffer
import qdrant_client

logging.getLogger("httpx").setLevel(logging.WARNING)
load_dotenv()

app = FastAPI(title="Sasidhar AI Clone API", description="Production RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sasidharakurathi-portfolio.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
global_index = None
user_sessions = {}


class ChatRequest(BaseModel):
    query: str
    is_mini_widget: bool = False
    session_id: str


CLONE_SYSTEM_PROMPT = """
You are Akurathi Sasidhar, a 22-year-old AI Engineer and Backend Developer. 
You are speaking directly to a user visiting your portfolio website. 

RULES:
1. Speak in the first person ("I", "my", "mine").
2. Answer questions strictly based on your provided knowledge base, projects, skills, and experience. 
3. If the user asks a conversational question ("Hi", "How are you?"), respond naturally as yourself.
4. IF THE USER ASKS A QUESTION UNRELATED TO YOU, STRICTLY REFUSE. 
5. Be professional, confident, and concise. Do not hallucinate skills or projects you do not have.
"""


@app.on_event("startup")
def startup_event():
    global global_index
    print("\n--- BOOTING UP AI CLONE SERVER ---")

    Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
    Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")

    client = qdrant_client.QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(
        client=client, collection_name="sasi_kb_hybrid", enable_hybrid=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if client.collection_exists("sasi_kb_hybrid"):
        print("-> Existing Hybrid database found!")
        global_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    else:
        print("-> Building Hybrid database from scratch...")
        documents = SimpleDirectoryReader("data").load_data()
        nodes = MarkdownNodeParser().get_nodes_from_documents(documents)
        global_index = VectorStoreIndex(nodes, storage_context=storage_context)

    print("--- SERVER IS READY TO RECEIVE REQUESTS ---\n")


@app.get("/api/ping")
def ping():
    return {"status": "awake", "message": "Server is running"}


@app.post("/api/chat")
def chat_with_clone(request: ChatRequest):
    if not global_index:
        raise HTTPException(status_code=500, detail="Database is still initializing.")

    session_id = request.session_id

    if session_id not in user_sessions:
        print(f"Creating new memory buffer for session: {session_id}")
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        user_sessions[session_id] = global_index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            system_prompt=CLONE_SYSTEM_PROMPT,
            similarity_top_k=5,
            sparse_top_k=5,
            vector_store_query_mode="hybrid",
        )

    final_query = request.query

    if request.is_mini_widget:
        mini_instructions = (
            "\n\n[SYSTEM INSTRUCTION: The user is using a tiny floating widget. "
            "Keep your answer extremely brief (1 to 2 short sentences max). "
            "End exactly with: 'Switch to the full chat view for a detailed breakdown.']"
        )
        final_query += mini_instructions

    engine = user_sessions[session_id]
    response = engine.chat(final_query)

    return {
        "query": request.query,
        "is_mini_widget": request.is_mini_widget,
        "session_id": session_id,
        "response": str(response),
    }
