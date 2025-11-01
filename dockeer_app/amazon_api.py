# -----------------------------
# 1️⃣ Imports
# -----------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import torch
import os
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# -----------------------------
# 2️⃣ Define paths relative to container working dir (/app)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_discount_predictor.pkl")
EMBEDDING_PATH = os.path.join(BASE_DIR, "embeddings", "amazon_embeddings.pt")

# -----------------------------
# 3️⃣ Load discount prediction model
# -----------------------------
try:
    loaded_model = joblib.load(MODEL_PATH)
    print("✅ Discount prediction model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load discount model: {MODEL_PATH}\nError: {e}")

# -----------------------------
# 4️⃣ Load embeddings and SentenceTransformer for RAG
# -----------------------------
try:
    rag_data = torch.load(EMBEDDING_PATH, map_location="cpu")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = rag_data["embeddings"]
    corpus_texts = rag_data["texts"]
    print("✅ Embeddings loaded successfully for RAG!")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load embeddings: {EMBEDDING_PATH}\nError: {e}")

# -----------------------------
# 5️⃣ Load text generation model (Qwen)
# -----------------------------
try:
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        device_map="auto"
    )
    print("✅ Text-generation model (Qwen2.5-1.5B-Instruct) loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load generator model: {e}")

# -----------------------------
# 6️⃣ FastAPI app
# -----------------------------
app = FastAPI(title="Amazon API with Discount Prediction & RAG")

# -----------------------------
# 7️⃣ Request schemas
# -----------------------------
class ProductItem(BaseModel):
    product_name: str
    category: str
    actual_price: float
    rating: float
    rating_count: int
    about_product: str
    review_content: str

class ProductList(BaseModel):
    products: List[ProductItem]

class QueryItem(BaseModel):
    query: str
    top_k: int = 5

# -----------------------------
# 8️⃣ Discount Prediction Endpoint
# -----------------------------
@app.post("/predict_discount")
def predict_discount_api(request: ProductList):
    try:
        input_data = pd.DataFrame([item.dict() for item in request.products])
        expected_cols = ['actual_price', 'rating', 'rating_count',
                         'category', 'product_name', 'about_product', 'review_content']
        for col in expected_cols:
            if col not in input_data.columns:
                raise HTTPException(status_code=400, detail=f"Missing column: {col}")

        preds = loaded_model.predict(input_data)
        input_data['predicted_discount'] = preds
        response = input_data[['product_name', 'predicted_discount']].to_dict(orient="records")

        return {"predictions": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during discount prediction: {e}")

# -----------------------------
# 9️⃣ RAG Answer Endpoint
# -----------------------------
@app.post("/rag_answer")
def rag_answer_api(request: QueryItem):
    try:
        query = request.query
        top_k = request.top_k

        query_emb = embed_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, corpus_embeddings)[0]
        top_results = torch.topk(scores, k=top_k)
        top_chunks = [corpus_texts[idx] for idx in top_results[1]]

        context = "\n\n".join(top_chunks)
        prompt = f"""You are an expert Amazon product assistant.
Based on the following information and review about the product, answer the user's question in a very optimal way.
Context:
{context}
User question:
{query}
Answer:"""

        response = generator(
            prompt,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            return_full_text=False
        )

        return {"answer": response[0]["generated_text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RAG generation: {e}")

# -----------------------------
# 🔟 Health Check
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Amazon API running with Discount Prediction and RAG"}
