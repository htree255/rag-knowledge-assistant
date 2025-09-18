# This script demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline
# using the LlamaIndex library. It shows BM25 retrieval method and a unified function 
# to run a query against a Google Gemini large language model.

# ==============================================================================
# 1. Setup and Data Loading
# ==============================================================================

import os
from dotenv import load_dotenv
import pandas as pd
from mock_data import data

from llama_index.core import Document, StorageContext, VectorStoreIndex

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from google import genai

# ==============================================================================
# 2. Config and Setup
# ==============================================================================

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")
TOP_K = int(os.getenv("TOP_K", 3))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Convert to pandas for pre processing
kb_df = pd.DataFrame(data)

# print(kb_df.head)
# print(kb_df.columns)

# ==============================================================================
# 3. Document Preparation
# ==============================================================================

# We convert the loaded dataset items into LlamaIndex's Document objects.
# This format is required for LlamaIndex to process the data.
# The `text` field contains the core content (knowledge base text), and the
# `metadata` field stores additional information, like the query category.

docs = [Document(text = row["text"], metadata = {"category":row["category"],
                                                 "restaurant_id":row["restaurant_id"]}) 
        for _, row in kb_df.iterrows() ]

# ==============================================================================
# 4. Retriever Initialization with two methods: BM25 and embeddings
# ==============================================================================

# -----------------
# BM25 Retriever
# -----------------

bm25_retriever = BM25Retriever.from_defaults(nodes=docs, similarity_top_k=TOP_K)

# -----------------
# embedding Retriever
# -----------------

# Create the embedding model
embed_client = GoogleGenAIEmbedding(
    model_name=EMBED_MODEL_NAME,
    api_key=GEMINI_API_KEY
)

# Create a vector store using the embeddings
storage_context = StorageContext.from_defaults()
vector_index = VectorStoreIndex.from_documents(docs, storage_context, embed_model=embed_client)

# Then make a retriever out of the vector index
embedding_retriever = vector_index.as_retriever(similarity_top_k=TOP_K)

# Create a retriever dictionary
retrievers = {
    "bm25": bm25_retriever,
    "embedding": embedding_retriever
}

# ==============================================================================
# 4. LLM Initialization
# ==============================================================================

client = genai.Client()

# ==============================================================================
# 5. RAG Pipeline Function
# ==============================================================================

def rag_pipeline(query: str, method: str):
    """
    Runs the RAG pipeline with BM25 retrieval method.

    Args:
        query (str): The question to be answered.
        method (str): The retrieval method to use ('bm25' or 'embedding').
    """
    
    retriever = retrievers[method]
    context_list = retriever.retrieve(query)
    context = ""

    # The retrieved documents are formatted into a single string to be passed
    # to the LLM as context. 
    for node_with_score in context_list[:1]:
        context+=f"Node Text: {node_with_score.node.get_content()[:250]}...\n"
        #context+=f"BM25 Score: {node_with_score.score}\n"
        #print(context)

    # A simple prompt template is used to instruct the LLM. It clearly separates
    # the provided context from the user's question.
    prompt = f"""You are a helpful Eatsy customer support assistant.
             Use the following knowledge base context to answer the query.

            ### User Query
            {query}

            ### Context
            {context}
            """
    
    try:
        response = client.models.generate_content(
        model=MODEL_NAME, contents=prompt
        )
        answer = response.candidates[0].content.parts[0].text
    except Exception as e:
        answer = f"Error: {str(e)}"

    #print(response.text)
    return query, answer

# ==============================================================================
# 6. Main function (Example Usage)
# ==============================================================================

if __name__ == "__main__":

    queries = [
        "How can I get a discount?",
        "I want to cancel my order and get a refund.",
        "I want to customize my order as I have a dietary restriction"
    ]
    # Example usage:
    for q in queries:
        result = rag_pipeline(q, "embedding")
        print(f"\n[User] {result[0]}")
        print(f"[Eatsy Assistant] {result[1]}")
