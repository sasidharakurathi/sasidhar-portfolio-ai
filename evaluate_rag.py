import json
import nest_asyncio

nest_asyncio.apply()  # This magically fixes the Gemini event loop issue!

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
import qdrant_client

load_dotenv()


def setup_pipeline():
    print("Setting up LLM and Database...")
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
    Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")

    # We only need the single synchronous client!
    client = qdrant_client.QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(
        client=client, collection_name="sasi_kb_hybrid", enable_hybrid=True
    )
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def run_evaluation():
    index = setup_pipeline()
    query_engine = index.as_query_engine(
        similarity_top_k=3, vector_store_query_mode="hybrid"
    )

    # Initialize Evaluators
    llm = GoogleGenAI(model="gemini-2.5-flash")
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevance_evaluator = RelevancyEvaluator(llm=llm)

    # Load Golden Dataset
    with open("eval_dataset.json", "r") as f:
        dataset = json.load(f)

    print(f"\nStarting Evaluation for {len(dataset)} queries...\n" + "-" * 40)

    total_faithfulness = 0
    total_relevance = 0

    for item in dataset:
        query = item["query"]
        print(f"Q: {query}")

        # 1. Generate Response (Synchronous)
        response = query_engine.query(query)

        # 2. Evaluate Faithfulness (Synchronous)
        faith_eval = faithfulness_evaluator.evaluate_response(response=response)
        if faith_eval.passing:
            total_faithfulness += 1

        # 3. Evaluate Relevance (Synchronous)
        rel_eval = relevance_evaluator.evaluate_response(query=query, response=response)
        if rel_eval.passing:
            total_relevance += 1

        print(f"Faithfulness Pass: {'✅' if faith_eval.passing else '❌'}")
        print(f"Relevance Pass:    {'✅' if rel_eval.passing else '❌'}")
        print("-" * 40)

    print("\n--- FINAL EVALUATION SCORE ---")
    print(f"Faithfulness (No Hallucinations): {total_faithfulness}/{len(dataset)}")
    print(f"Relevance (Answered Prompt):      {total_relevance}/{len(dataset)}")


if __name__ == "__main__":
    run_evaluation()
