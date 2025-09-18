# 🍃 Eatsy RAG Knowledge Assistant  

**A Retrieval-Augmented Generation (RAG) knowledge assistant for customer support in a vegan/vegetarian food delivery business.**  

---

## 🔑 Highlights  
- **Knowledge Base**: Mock dataset with FAQs, refund policies, delivery info, and support transcripts.  
- **Retriever**: BM25 and Gemini embedding-based retrieval (switchable).  
- **LLM**: Google Gemini integration.  
- **Pipeline**: Unified `rag_pipeline()` function for query → retrieve → context → generate.  
- **Portfolio Value**: Demonstrates applied AI engineering (retrieval, LLM integration, RAG design).  

---

## ⚙️ Tech Stack  
- **Python** (pandas, dotenv)  
- **LlamaIndex** (BM25Retriever, VectorStoreIndex, GeminiEmbedding)  
- **LLMs** (Gemini for text generation + embeddings)  
- **Custom Knowledge Base**  

---

## 🚀 Quickstart  

```bash
git clone https://github.com/htree255/rag-knowledge-assistant.git
cd rag-knowledge-assistant
pip install -r requirements.txt
```

Add a `.env` file with your API key and defaults:  

```ini
GOOGLE_API_KEY=your_google_key_here

# Default LLM model
MODEL_NAME=gemini-2.5-flash

# Default Embedding model
EMBED_MODEL_NAME=gemini-embedding-001

# Retrieval settings
TOP_K=3
```

Run the pipeline:  

```bash
python app.py
```

**Example Run**:  

```
BM25-based retrieval
---------------------------------
[User] How can I get a discount?
[Eatsy Assistant] You can get discounts by referring a friend.

[User] I want to cancel my order and get a refund.
[Eatsy Assistant] You can cancel your order within 5 minutes of placing it.

[User] I want to customize my order as I have a dietary restriction.
[Eatsy Assistant] Yes, you can absolutely customize your order due to dietary restrictions!


Embedding-based retrieval
---------------------------------
[User] How can I get a discount?
[Eatsy Assistant] You can get a discount by referring a friend.

[User] I want to cancel my order and get a refund.
[Eatsy Assistant] You can cancel an order within 5 minutes of placing it. If you cancel within this timeframe, you will be eligible for a refund.

[User] I want to customize my order as I have a dietary restriction.
[Eatsy Assistant] Yes, you can customize your order due to dietary restrictions.
```

---

## 🔀 Switching Retrieval Methods  

The pipeline supports both **BM25** (keyword-based) and **embedding-based** retrieval using the Gemini embedding model.  

```python
# BM25 retriever
result = rag_pipeline("How can I get a discount?", method="bm25")

# Embedding retriever
result = rag_pipeline("How can I get a discount?", method="embedding")
```

This makes it easy to compare **keyword search** vs **semantic search** in customer support scenarios.  

---

## 📓 Observations / Learnings  

We compared **two approaches** for working with Gemini embeddings:  

### 🔹 Direct Google SDK (`genai.Client`)  

```python
from google import genai
client = genai.Client()

res = client.models.embed_content(
    model="gemini-embedding-001",
    contents="What is the meaning of life?"
)
vector = res.embedding.values
```

**✅ Pros**  
- Direct access to Gemini API (no extra layer).  
- Full control over embedding calls and storage.  
- Great for testing and custom pipelines.  

**⚠️ Cons**  
- You must manage document chunking, vector storage, and retrieval logic manually.  
- More boilerplate code for a complete RAG system.  

---

### 🔹 LlamaIndex `GeminiEmbedding`  

```python
from llama_index.embeddings.gemini import GeminiEmbedding

embed_model = GeminiEmbedding(model_name="gemini-embedding-001")
vector = embed_model.get_text_embedding("What is the meaning of life?")
```

**✅ Pros**  
- Integrates seamlessly with LlamaIndex retrievers (`VectorStoreIndex`, `as_retriever`).  
- Handles chunking, embedding, and retrieval automatically.  
- Easy to switch between BM25, embeddings, or hybrid retrieval.  
- Less boilerplate for building full RAG pipelines.  

**⚠️ Cons**  
- Adds dependency on LlamaIndex.  
- Less direct control over low-level API calls.  

---

### 💡 Takeaway  

- Use **Google SDK directly** if you want **full control** and are building a pipeline from scratch.  
- Use **LlamaIndex GeminiEmbedding** if you want **faster development** and easy integration with BM25 + hybrid retrieval.  

---

## 🔮 Roadmap  
- Add **hybrid retriever** (BM25 + embeddings together).  
- Add **LLM Guardrails** (to prevent hallucinations & off-brand responses).   
- Implement an **LLM Judge** for automated evaluation.  
- Add **evaluation metrics** (e.g., retrieval precision/recall, grounding score).  
- Build a **UI chatbot** with retrieval toggle.  

---
