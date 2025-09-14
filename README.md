# 🍃 Eatsy RAG Assistant  

**A Retrieval-Augmented Generation (RAG) knowledge assistant for customer support in a vegan/vegetarian food delivery business.**  

---

## 🔑 Highlights  
- **Knowledge Base**: Mock dataset with FAQs, refund policies, delivery info, and support transcripts.  
- **Retriever**: BM25-based search (embedding retriever coming soon).  
- **LLM**: Google Gemini integration.  
- **Pipeline**: Unified `rag_pipeline()` function for query → retrieve → context → generate.  
- **Portfolio Value**: Demonstrates applied AI engineering (retrieval, LLM integration, RAG design).  

---

## ⚙️ Tech Stack  
- **Python** (pandas, dotenv)  
- **LlamaIndex** (BM25Retriever)  
- **LLMs** (Gemini)  
- **Custom Knowledge Base**  

---

## 🚀 Quickstart  

```bash
git clone https://github.com/your-username/rag-knowledge-assistant.git
cd rag-knowledge-assistant
pip install -r requirements.txt
```

Add `.env` with your API key:  
```
GEMINI_API_KEY=your_google_key_here

```

Run the pipeline:  
```bash
python app.py
```

Example:  
```
Query: I want a refund
Answer: Refunds can be requested in-app within 24h and process in 3–5 days.
```

---

## 🔮 Roadmap  
- Add embedding-based retriever.  
- Add **LLM Guardrails** (to prevent hallucinations & off-brand responses).   
- Implement an **LLM Judge** for automated evaluation.  
- Add **evaluation metrics** (e.g., retrieval precision/recall, answer grounding score). 
- Build UI chatbot

---
