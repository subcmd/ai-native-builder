# 2. 예시 (Python 실전 코드)

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

from sentence_transformers import SentenceTransformer
import numpy as np

models = {
    "BGE-m3": SentenceTransformer("BAAI/bge-m3"),
    "multilingual-e5": SentenceTransformer("intfloat/multilingual-e5-large"),
    "ko-sroberta": SentenceTransformer("jhgan/ko-sroberta-multitask"),
}

# 평가 데이터: 의미가 비슷한 쌍
pairs_similar = [
    ("DAU 가 감소했다", "일간 활성 사용자 수가 줄었다"),
    ("결제 전환율 개선", "구매 전환률 향상"),
    ("이탈 유저 분석", "churn user analysis"),
]
# 의미가 다른 쌍
pairs_different = [
    ("DAU 가 감소했다", "오늘 점심 메뉴는 무엇인가"),
    ("결제 전환율", "축구 경기 결과"),
]

for name, model in models.items():
    print(f"\n=== {name} ===")
    print("[비슷한 쌍]")
    for s1, s2 in pairs_similar:
        e1, e2 = model.encode([s1, s2], normalize_embeddings=True)
        print(f"  {np.dot(e1, e2):.3f}  | {s1}  ↔  {s2}")
    print("[다른 쌍]")
    for s1, s2 in pairs_different:
        e1, e2 = model.encode([s1, s2], normalize_embeddings=True)
        print(f"  {np.dot(e1, e2):.3f}  | {s1}  ↔  {s2}")

"""
=== BGE-m3 ===
# 다국어 환경까지 고려시 강함

[비슷한 쌍]
  0.675  | DAU 가 감소했다  ↔  일간 활성 사용자 수가 줄었다
  0.882  | 결제 전환율 개선  ↔  구매 전환률 향상
  0.600  | 이탈 유저 분석  ↔  churn user analysis
[다른 쌍]
  0.392  | DAU 가 감소했다  ↔  오늘 점심 메뉴는 무엇인가
  0.429  | 결제 전환율  ↔  축구 경기 결과

=== multilingual-e5 ===

[비슷한 쌍]
  0.887  | DAU 가 감소했다  ↔  일간 활성 사용자 수가 줄었다
  0.956  | 결제 전환율 개선  ↔  구매 전환률 향상
  0.863  | 이탈 유저 분석  ↔  churn user analysis
[다른 쌍]
  0.787  | DAU 가 감소했다  ↔  오늘 점심 메뉴는 무엇인가
  0.820  | 결제 전환율  ↔  축구 경기 결과

=== ko-sroberta ===
# 범용적 한국어 유사도 측정에 효율적

[비슷한 쌍]
  0.598  | DAU 가 감소했다  ↔  일간 활성 사용자 수가 줄었다
  0.733  | 결제 전환율 개선  ↔  구매 전환률 향상
  0.577  | 이탈 유저 분석  ↔  churn user analysis
[다른 쌍]
  -0.114  | DAU 가 감소했다  ↔  오늘 점심 메뉴는 무엇인가
  0.178  | 결제 전환율  ↔  축구 경기 결과
"""