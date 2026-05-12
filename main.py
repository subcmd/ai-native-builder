# 2. 예시 (Python 실전 코드)

# prompts/classify_feedback/v2.0.yaml 파일 가정

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

