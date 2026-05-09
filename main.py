import logging
from contextlib import asynccontextmanager
from threading import Lock
from time import monotonic
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
logger = logging.getLogger(__name__)
load_dotenv()

SESSION_TTL_SECONDS = 1800
MAX_ACTIVE_SESSIONS = 500

global_index: Optional[VectorStoreIndex] = None
user_sessions: Dict[str, Dict[str, Any]] = {}
session_lock = Lock()


def _prune_expired_sessions(now: float) -> None:
    expired_session_ids = [
        session_id
        for session_id, session_data in user_sessions.items()
        if now - session_data["last_access"] > SESSION_TTL_SECONDS
    ]
    for session_id in expired_session_ids:
        user_sessions.pop(session_id, None)


def _enforce_session_limit() -> None:
    overflow_count = len(user_sessions) - MAX_ACTIVE_SESSIONS
    if overflow_count <= 0:
        return

    # Evict least recently used sessions first.
    sorted_sessions = sorted(
        user_sessions.items(), key=lambda item: item[1]["last_access"]
    )
    for session_id, _ in sorted_sessions[:overflow_count]:
        user_sessions.pop(session_id, None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global global_index
    logger.info("--- BOOTING UP AI CLONE SERVER ---")

    Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
    Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")

    client = qdrant_client.QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(
        client=client, collection_name="sasi_kb_v2", enable_hybrid=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if client.collection_exists("sasi_kb_v2"):
        logger.info("Existing Hybrid database found")
        global_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    else:
        logger.info("Building Hybrid database from scratch")
        documents = SimpleDirectoryReader("data").load_data()
        nodes = MarkdownNodeParser().get_nodes_from_documents(documents)
        global_index = VectorStoreIndex(nodes, storage_context=storage_context)

    logger.info("--- SERVER IS READY TO RECEIVE REQUESTS ---")
    yield


app = FastAPI(
    title="Sasidhar AI Clone API",
    description="Production RAG Backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["https://sasidharakurathi-portfolio.vercel.app", "https://portfolio-644959644449.us-central1.run.app"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    is_mini_widget: bool = False
    session_id: str = Field(min_length=1, max_length=128)


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


@app.get("/api/ping")
def ping():
    return {"status": "awake", "message": "Server is running"}


@app.post("/api/chat")
def chat_with_clone(request: ChatRequest):
    if global_index is None:
        raise HTTPException(status_code=500, detail="Database is still initializing.")

    session_id = request.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id cannot be empty.")

    now = monotonic()

    with session_lock:
        _prune_expired_sessions(now)
        session_data = user_sessions.get(session_id)

        if session_data is None:
            logger.info("Creating new memory buffer for session: %s", session_id)
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            session_data = {
                "engine": global_index.as_chat_engine(
                    chat_mode="condense_plus_context",
                    memory=memory,
                    system_prompt=CLONE_SYSTEM_PROMPT,
                    similarity_top_k=5,
                    sparse_top_k=5,
                    vector_store_query_mode="hybrid",
                ),
                "last_access": now,
            }
            user_sessions[session_id] = session_data
            _enforce_session_limit()
        else:
            session_data["last_access"] = now

    final_query = request.query

    if request.is_mini_widget:
        mini_instructions = (
            "\n\n[SYSTEM INSTRUCTION: The user is using a tiny floating widget. "
            "Keep your answer extremely brief (1 to 2 short sentences max). "
            "End exactly with: '\n\nSwitch to the full chat view for a detailed breakdown.']"
        )
        final_query += mini_instructions

    engine = session_data["engine"]
    try:
        response = engine.chat(final_query)
    except Exception:
        logger.exception("Chat generation failed for session: %s", session_id)
        raise HTTPException(
            status_code=503,
            detail="Chat service is temporarily unavailable. Please try again.",
        )

    return {
        "query": request.query,
        "is_mini_widget": request.is_mini_widget,
        "session_id": session_id,
        "response": str(response),
    }
