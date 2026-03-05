#!/bin/bash
set -e

# Force Python 3.11
export PYTHON_VERSION=3.11.9

pip install --upgrade pip

# Install all dependencies (sentence-transformers pulls torch which is large;
# Render's free tier may need extra memory — upgrade plan if build times out)
pip install -r requirements.txt

# Pre-download the HuggingFace embedding model so first cold-start is instant
python -c "
from langchain_huggingface import HuggingFaceEmbeddings
import os
model = os.getenv('embedding_model_name', 'BAAI/bge-small-en')
print(f'Pre-loading embedding model: {model}')
HuggingFaceEmbeddings(model_name=model, encode_kwargs={'normalize_embeddings': True})
print('Embedding model cached.')
"
