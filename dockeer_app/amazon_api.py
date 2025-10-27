# -----------------------------
# 1️⃣ Imports
# -----------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# -----------------------------
# 2️⃣ Load trained model for discount prediction
# -----------------------------
model_path = "/home/ubuntu/experiments/forecasting model/xgb_discount_predictor.pkl"
loaded_model = joblib.load(model_path)
print("✅ Discount model loaded successfully!")

# -----------------------------
# 3️⃣ Load embeddings for RAG
# -----------------------------
embeddings_path = "/home/ubuntu/experiments/amazon_embeddings.pt"
rag_data = torch.load(embeddings_path)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = rag_data['embeddings']
corpus_texts = rag_data['texts']
print("✅ RAG embeddings loaded successfully!")

# -----------------------------
# 4️⃣ Load text-generation model
# -----------------------------
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",  # replace with local path if needed
    device_map="auto"
)
print("✅ Generator model loaded successfully!")

# -----------------------------
# 5️⃣ FastAPI app
# -----------------------------
app = FastAPI(title="Amazon API with Discount Prediction & RAG")

# -----------------------------
# 6️⃣ Pydantic models
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
# 7️⃣ Discount prediction endpoint
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
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 8️⃣ RAG answer endpoint
# -----------------------------
@app.post("/rag_answer")
def rag_answer_api(request: QueryItem):
    try:
        query = request.query
        top_k = request.top_k

        # -----------------------------
        # Retrieve top-k chunks
        # -----------------------------
        query_emb = embed_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, corpus_embeddings)[0]
        top_results = torch.topk(scores, k=top_k)
        top_chunks = [corpus_texts[idx] for idx in top_results[1]]

        # -----------------------------
        # Generate answer
        # -----------------------------
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
            return_full_text=False  # only the generated tokens
        )

        return {"answer": response[0]["generated_text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 9️⃣ Health check
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Amazon API running with Discount Prediction and RAG"}
