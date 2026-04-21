import os
import logging
import sys
from dotenv import load_dotenv

# LlamaIndex Core
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import MarkdownNodeParser  # <-- THE NEW BRAIN UPGRADE

# Gemini Integrations
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Qdrant Integration
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

print("1. Loading Environment Variables...")
load_dotenv()

print("2. Configuring Google GenAI Settings...")
Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")

print("3. Initializing in-memory Qdrant database...")
client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="sasi_kb_markdown")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("4. Reading the knowledge base...")
documents = SimpleDirectoryReader("data").load_data()

print("5. Parsing Markdown Structure (The Memory Upgrade)...")
# Initialize the Markdown Parser
parser = MarkdownNodeParser()
# This extracts the document and splits it intelligently by ## and ### headers
nodes = parser.get_nodes_from_documents(documents)

print(f"-> Extracted {len(nodes)} distinct sections based on headers.")

print("6. Embedding and Storing in Qdrant...")
# Notice we pass `nodes` here instead of `documents`
index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

print("\n--- Setup Complete. Testing Query Engine ---\n")

query_engine = index.as_query_engine()

# Let's test a query that requires understanding context grouped under a specific header
test_query = "What is the architecture and tech stack of RoboSwift? Explain how it parses stdout."
print(f"USER: {test_query}\n")

response = query_engine.query(test_query)

print(f"AI CLONE: {response}")
