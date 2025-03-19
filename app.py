import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import time

app = FastAPI(
    title="LLM-Powered Booking Analytics & QA System",
    description="A simple FastAPI service providing booking analytics and retrieval-augmented Q&A.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

DATA_CSV_PATH = "hotel_bookings_preprocessed.csv"

if not os.path.exists(DATA_CSV_PATH):
    raise FileNotFoundError(
        f"Preprocessed data file '{DATA_CSV_PATH}' not found. "
        "Please export your cleaned DataFrame from Jupyter."
    )

dtype_dict = {
    "is_canceled": "int8",
    "lead_time": "int16",
    "adr": "float32",
    "total_stays": "int16",
    "revenue": "float32"
}

chunk_size = 10000
df_chunks = pd.read_csv(DATA_CSV_PATH, dtype=dtype_dict, low_memory=True, chunksize=chunk_size)
df = pd.concat(df_chunks, ignore_index=True)
print("Data loaded. Shape:", df.shape)

total_bookings = len(df)
total_cancellations = df['is_canceled'].sum()
cancellation_rate = (total_cancellations / total_bookings) * 100
df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
monthly_revenue = df.groupby(pd.Grouper(key='arrival_date', freq='ME'))['revenue'].sum()
country_counts = df['country'].value_counts().head(5).to_dict()

analytics_data = {
    "cancellation_rate": round(cancellation_rate, 2),
    "monthly_revenue_trend": monthly_revenue.to_dict(),
    "top_countries": country_counts,
}

def record_to_text(row):
    return (
        f"Hotel: {row.get('hotel', 'N/A')}, "
        f"Country: {row.get('country', 'Unknown')}, "
        f"Canceled: {row.get('is_canceled', '0')}, "
        f"Lead Time: {row.get('lead_time', 'N/A')} days, "
        f"ADR: {row.get('adr', 'N/A')}, "
        f"Total Stays: {row.get('total_stays', 0)}, "
        f"Revenue: {row.get('revenue', 0.0)}"
    )

records_text = df.apply(record_to_text, axis=1).tolist()

FAISS_INDEX_PATH = "faiss_index.bin"
EMBEDDINGS_NPY_PATH = "embeddings.npy"

print("Loading SentenceTransformer model for embeddings...")
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_NPY_PATH):
    print("Loading existing FAISS index...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    embeddings = np.load(EMBEDDINGS_NPY_PATH)
else:
    print("Computing embeddings for each record (this may take time)...")
    embeddings = embedding_model.encode(records_text, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(EMBEDDINGS_NPY_PATH, embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

print(f"FAISS index loaded with {index.ntotal} vectors.")

print("Loading DistilGPT2 (this may take a moment)...")
model_name = "distilgpt2"
tokenizer_llm = AutoTokenizer.from_pretrained(model_name)
model_llm = AutoModelForCausalLM.from_pretrained(model_name)

def answer_question(question: str, k: int = 3) -> str:
    query_emb = embedding_model.encode([question])
    query_emb = np.array(query_emb, dtype=np.float32)
    distances, indices = index.search(query_emb, k)
    retrieved_contexts = [records_text[i] for i in indices[0]]
    context_str = "\n".join(retrieved_contexts)
    prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer_llm(prompt, return_tensors="pt")
    outputs = model_llm.generate(
        **inputs,
        max_length=256,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )
    answer = tokenizer_llm.decode(outputs[0], skip_special_tokens=True)
    return answer

class AnalyticsRequest(BaseModel):
    pass

class AskRequest(BaseModel):
    question: str

@app.post("/analytics")
def get_analytics(request: AnalyticsRequest):
    try:
        return analytics_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question_endpoint(request: AskRequest):
    try:
        user_question = request.question
        answer = answer_question(user_question, k=3)
        return {"question": user_question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "faiss_index_size": index.ntotal,
        "model_loaded": True
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
