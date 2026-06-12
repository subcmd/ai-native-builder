# 2. 예시 (Python 실전 코드)

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

import chroma_db
from chroma_db.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# 1) 인덱싱
chroma = chromadb.PersistentClient(path="./naive_rag_db")