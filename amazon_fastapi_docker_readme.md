# Amazon Product Assistant API

A **FastAPI** application that provides two main functionalities for Amazon product data:

1. **Discount Prediction**: Predict the discount percentage of a product based on its features using a trained XGBoost model.
2. **QA (Retrieve & Generate based)**: Answer user queries about products using product metadata, reviews, and descriptions with a Retrieval-Augmented Generation pipeline.

The application is fully **Dockerized** for easy deployment.

---

## Table of Contents

1. [Project Structure]
2. [Setup & Installation]
3. [FastAPI Endpoints]
4. [RAG Pipeline]  
5. [Discount Prediction Pipeline]
6. [Docker Deployment]
7. [Testing API] 
8. [GPU Support]  
9. [Notes & Best Practices]
10. [Secret recepi]
11. [Analysis and Better options]
---

## Project Structure

```
amazon_app/
├── amazon_api.py          # Main FastAPI application
├── requirements.txt       # Python dependencies
├── models/
│   └── xgb_model.pkl      # Saved XGBoost discount prediction model
├── embeddings/
│   └── amazon_embeddings.pt  # Precomputed embeddings for RAG QA
└── Dockerfile             # Docker configuration
```

### File Details:

- **amazon_api.py**: Contains FastAPI app, routes for:
  - `/predict_discount` → Predict product discount
  - `/query_product` → RAG-based question answering
- **requirements.txt**: Python dependencies including FastAPI, scikit-learn, transformers, torch, xgboost, sentence-transformers.
- **models/xgb_model.pkl**: Pretrained XGBoost model for discount prediction.
- **embeddings/amazon_embeddings.pt**: Serialized embeddings for RAG retrieval.
- **Dockerfile**: Instructions for building a deployable Docker image.

---

## Setup & Installation

### Prerequisites

- Python 3.10+  
- Docker 20+  
- NVIDIA GPU (optional, for RAG model acceleration)

### Local Setup

1. Clone the repo:

```bash
git clone <repo_url>
cd amazon_app
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Ensure `models/xgb_model.pkl` and `embeddings/amazon_embeddings.pt` are in the correct directories.

---

## FastAPI Endpoints

### 1️⃣ `/predict_discount`  

Predicts discount percentage given product details.

**Method:** POST  
**Input JSON Example:**

```json
{
  "product_name": "Wayona Nylon Braided USB Cable for iPhone",
  "category": "Computers&Accessories|Accessories&Peripherals|Cables&Accessories|Cables|USBCables",
  "actual_price": 1099.0,
  "rating": 4.2,
  "rating_count": 24269,
  "about_product": "Durable nylon braided design, fast charging and sync, high compatibility",
  "review_content": "Good quality and works fine so far."
}
```

**Response Example:**

```json
{
  "product_name": "Wayona Nylon Braided USB Cable for iPhone",
  "predicted_discount": 55.23
}
```

---

### 2️⃣ `/query_product`  

RAG-based QA endpoint. Returns answers for user questions based on product data.

**Method:** POST  
**Input JSON Example:**

```json
{
  "query": "I need a durable iPhone charging cable with fast charging.",
  "top_k": 5
}
```

**Response Example:**

```json
{
  "answer": "You should consider the Wecool Unbreakable 3-in-1 charging cable, which supports fast charging and has a durable nylon braided design."
}
```

**Notes:**  
- The pipeline retrieves top-k most relevant product chunks from embeddings.  
- Uses `SentenceTransformer` for embedding and `transformers` for text generation.  

---

## RAG Pipeline (Technical)

1. **Embedding generation**: Each product row is converted into a text “chunk” containing product metadata, description, and reviews.  
2. **Sentence Embeddings**: Generated using `all-MiniLM-L6-v2` (384-dim).  
3. **Query Retrieval**: Query encoded → cosine similarity → top-k chunks selected.  
4. **Answer Generation**: Using a `transformers` text generation model (`Qwen2.5`) based on retrieved chunks.  

---

## Discount Prediction Pipeline

1. Preprocessing:
   - Numerical features scaled (`StandardScaler`)  
   - Categorical features one-hot encoded (`OneHotEncoder`)  
   - Text features converted to TF-IDF (`TfidfVectorizer`)  

2. Model:
   - XGBoost Regressor (`XGBRegressor`) trained on historical Amazon product data.  
3. Output:
   - Returns predicted discount percentage for each product.

---

## Docker Deployment

### Dockerfile Explanation

- **Base Image**: `python:3.10-slim`  
- **Dependencies**: `pip install` from `requirements.txt`  
- **Working Directory**: `/app`  
- **Expose Port**: 8000 for FastAPI  
- **CMD**: Runs Uvicorn to serve the FastAPI app

### Build Docker Image

```bash
docker build -t amazon_fastapi_app:latest .
```

### Run Docker Container

```bash
docker run -d -p 8000:8000 --name amazon_app amazon_fastapi_app:latest
```


- App URL: `http://localhost:8000`  
- Check Swagger docs: `http://localhost:8000/docs`

---

## Testing API

### 1. Discount Prediction:

```bash
curl -X POST "http://localhost:8000/predict_discount" \
-H "Content-Type: application/json" \
-d '{
  "product_name": "Wayona Nylon Braided USB Cable for iPhone",
  "category": "Computers&Accessories|Accessories&Peripherals|Cables&Accessories|Cables|USBCables",
  "actual_price": 1099.0,
  "rating": 4.2,
  "rating_count": 24269,
  "about_product": "Durable nylon braided design, fast charging and sync, high compatibility",
  "review_content": "Good quality and works fine so far."
}'
```

### 2. RAG Query:

```bash
curl -X POST "http://localhost:8000/rag_answer" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "I need a durable iPhone charging cable with fast charging.",
           "top_k": 5
         }'
```

---

## GPU Support

- The RAG model can leverage GPU for embedding generation and inference.  
- When running Docker:

```bash
docker run --gpus all -d -p 8000:8000 amazon_fastapi_app:latest
```

- Ensure `nvidia-docker2` and NVIDIA drivers are installed.

---

## Notes & Best Practices

- Keep `models/` and `embeddings/` outside container for large datasets (mount as volume).  
- Update `requirements.txt` if adding new dependencies.  
- For production, consider using **Uvicorn + Gunicorn** for better concurrency.  
- Limit `max_new_tokens` in RAG pipeline to avoid memory spikes.  
- Use batching for embedding queries if dealing with large data.

---

✅ **With this setup, you have a deployable container that can predict Amazon product discounts and answer product questions using a RAG-based QA pipeline.**

## Secret recepie
 - RAG framework
 Using cosine similarity since data is limited allowing better matching result

 - Discount prediction
 Used XGB based regression techineque 
 Outcome RMSE: ~14 on validation set

 ## Analysis 

 ### RAG

 - *Bottlenecks
        
    1. Using cosine similarity may lead to latency overhead

 - Improvements
    1. Mult stage retrieval strategy
        - Bucketize the elements based on category 
        - Based on input query use only selective category
        - Retrieve only from those category
    2. Inclusion of reranking strategy
    3. Agentic RAG- based on the 
    4. Answer alignemnet enhancement using RHFL 

 ### Discount Prediction 
 - Agenda- to predict the discount to be given on the specific item based on other parameters

 - Bottleneck
    1. Non increamental approach - As data grows but model remain stale
     2. scalability

 - Improvements
    1. Can use Reinforcement Bandit algorithms (UCB/ Thompson smapling)
    2. Incremantal approach.




# LoRA Fine-Tuning of TinyLlama LLM

This project demonstrates **LoRA-based fine-tuning** of a causal language model (TinyLlama) using custom text data. It includes dataset preparation, tokenization, model setup, training, saving, and inference testing.

---

## Table of Contents

1. [Overview] 
2. [Project Structure] 

---

## Overview

This script allows you to fine-tune a language model in causal way using LoRA (Low-Rank Adaptation). LoRA introduces a small number of trainable parameters to large pre-trained models, making fine-tuning faster and memory-efficient.

Key features:

- Handles tokenization and padding automatically.
- Uses PEFT (`peft` library) for LoRA integration.
- Supports FP16 and GPU acceleration.
- Includes a quick text generation test after training.

---

## Project Structure


---

## Setup & Installation

### Prerequisites

- Python 3.10+  
- PyTorch 2.x with CUDA (optional but recommended for GPU acceleration)  
- `transformers` >= 4.33  
- `datasets`  
- `peft`  





## Testing 

.local/sumit/experiment/hgincurl -X POST "http://localhost:8000/rag_answer" \ost:8000/rag_answer" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "I need a durable iPhone charging cable with fast charging.",
           "top_k": 5
         }'
{"answer":" Based on the provided reviews, the Amazon Basics New Release Nylon USB-A to Lightning Cable Cord appears to be a suitable option for your requirements. Here's why:\n\n- It's rated as \"good\" multiple times and praised for its durability, making it more reliable than some other alternatives.\n- The review mentions that it's \"superab,\" suggesting it performs exceptionally well.\n- The reviewer notes that they were satisfied enough to recommend it despite it being \"expensive.\"\n- Despite mentioning potential issues such as slow charging speed or pins that come out easily, these seem to be rare occurrences rather than common problems.\n\nWhile Belkin and Wecool products offer similar features and quick charging speeds, their pricing makes them potentially more costly. However, based on customer satisfaction and performance reports, the Amazon Basics cable stands out as providing a balance between cost-effectiveness and reliability. \n\nTherefore, if budget is a concern but durability and fast charging are essential, the Amazon Basics cable would be a strong contender for meeting your needs. Always consider testing it before committing to make sure it meets your specific charging needs."}



hginsight_new/docker_app# curl -X POST "http://localhost:8000/predict_discount" \
-H "Content-Type: application/json" \
-d '{
  "products": [
    {
      "product_name": "Wayona Nylon Braided USB Cable for iPhone",
      "category": "Computers&Accessories|Accessories&Peripherals|Cables&Accessories|Cables|USBCables",
      "actual_price": 1099.0,
      "rating": 4.2,
      "rating_count": 24269,
}'] } "review_content": "Good quality and works fine so far."arging and sync, high compatibility",
{"predictions":[{"product_name":"Wayona Nylon Braided USB Cable for iPhone","predicted_discount":49.99294662475586}]}