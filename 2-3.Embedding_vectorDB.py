# 2. 예시 (Python 실전 코드)

# 예시 1. OpenAI 임베딩 + 코사인 유사도

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

def embed(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# 단어들 임베딩
words = ["강아지", "고양이", "반려동물", "자동차", "비행기", "교통수단"]
embs = {w: embed(w) for w in words}

# 모든 쌍의 유사도
for i, w1 in enumerate(words):
    for w2 in words[i+1:]:
        s = cosine_sim(embs[w1], embs[w2])
        print(f"{w1:5s} ↔ {w2:5s}: {s:.3f}")

'''
강아지   ↔ 고양이  : 0.336
강아지   ↔ 반려동물 : 0.409
강아지   ↔ 자동차  : 0.284
강아지   ↔ 비행기  : 0.241
강아지   ↔ 교통수단 : 0.221
고양이   ↔ 반려동물 : 0.284
고양이   ↔ 자동차  : 0.253
고양이   ↔ 비행기  : 0.162
고양이   ↔ 교통수단 : 0.152
반려동물  ↔ 자동차  : 0.317
반려동물  ↔ 비행기  : 0.292
반려동물  ↔ 교통수단 : 0.285
자동차   ↔ 비행기  : 0.362
자동차   ↔ 교통수단 : 0.399
비행기   ↔ 교통수단 : 0.344
'''

# =============================================================================
# 예시 2. 로컹 임베딩(BGE-m3)

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
# pip install sentence-transformers torch

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# 처음 다운로드 시 약 2GB
model = SentenceTransformer("BAAI/bge-m3")

texts = [
    "DAU는 일간 활성 사용자 수를 의미한다.",
    "월간 활성 사용자는 MAU라고 부른다",
    "오늘 점심은 김치찌개를 먹었다",
]

embeddings = model.encode(texts, normalize_embeddings=True)
print(embeddings.shape) # (3, 1024)

# 1번 문장과 다른 문장들의 유사도
sims = embeddings @ embeddings[0]   # 정규화됐으니 내적 = 코사인
for text, sim in zip(texts, sims):
    print(f"{sim:.3f} | {text}")

## 장점 : 호출 무제한, 데이터 외부로 안 나감 (보안/규제)
## 단점 : 처음 모델 로딩 시간, 메모리 ~3GB

'''
modules.json: 100%|██████████████████████████████████████████| 349/349 [00:00<?, ?B/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config_sentence_transformers.json: 100%|█████████████████████| 123/123 [00:00<?, ?B/s]
README.md: 15.8kB [00:00, ?B/s]
sentence_bert_config.json: 100%|███████████████████████████| 54.0/54.0 [00:00<?, ?B/s]
config.json: 100%|███████████████████████████████████████████| 687/687 [00:00<?, ?B/s]
pytorch_model.bin: 100%|█████████████████████████| 2.27G/2.27G [00:22<00:00, 99.8MB/s]
Loading weights: 100%|███████████████████████████| 391/391 [00:00<00:00, 38650.35it/s]
tokenizer_config.json: 100%|█████████████████████████████████| 444/444 [00:00<?, ?B/s]
sentencepiece.bpe.model: 100%|███████████████████| 5.07M/5.07M [00:02<00:00, 1.98MB/s]
tokenizer.json: 100%|████████████████████████████| 17.1M/17.1M [00:00<00:00, 21.2MB/s] 
special_tokens_map.json: 100%|███████████████████████████████| 964/964 [00:00<?, ?B/s] 
config.json: 100%|███████████████████████████████████████████| 191/191 [00:00<?, ?B/s] 
(3, 1024)on:   0%|                                          | 0.00/191 [00:00<?, ?B/s] 
1.000 | DAU는 일간 활성 사용자 수를 의미한다.
0.707 | 월간 활성 사용자는 MAU라고 부른다
0.355 | 오늘 점심은 김치찌개를 먹었다
model.safetensors: 100%|██████████████████████████| 2.27G/2.27G [00:15<00:00, 146MB/s]
'''

# =============================================================================
# 예시 3. Chroma 기본 사용

import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
# pip install chromadb

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

client = chromadb.PersistentClient(path="./chroma_db")

# OpenAI 임베딩 사용
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="growth_docs",
    embedding_function=ef
)

# 문서 추가 (임베딩은 자동 계산됨)
collection.add(
    documents=[
        "DAU와 WAU의 차이는 측정 기간이다",
        "Funnel 분석은 단계별 전환율을 추적한다",
        "RFM 세그멘테이션은 최근성·빈도·금액으로 유저를 분류한다",
        "A/B 테스트는 통계적 유의성과 표본 크기 계산이 핵심이다",
    ],
    metadatas=[
        {"category": "metric"},
        {"category": "analysis"},
        {"category": "segmentation"},
        {"category": "experimentation"},
    ],
    ids=["doc1", "doc2", "doc3", "doc4"]
)

# 검색
results = collection.query(
    query_texts=["사용자 그룹 나누는 방법"],
    n_results=2
)

for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"distance={dist:.3f} | {doc}")


# distance=0.617 | RFM 세그멘테이션은 최근성·빈도·금액으로 유저를 분류한다
# distance=0.780 | A/B 테스트는 통계적 유의성과 표본 크기 계산이 핵심이다 


# =============================================================================
# 예시 4. pgvector 기본 사용



# =============================================================================
# 예시 5. 미니 시맨틱 검색 시스템



# =============================================================================
# 예시 6. 임베딩 시각화 (UMAP)



# =============================================================================
# 예시 7. 한국어 모델 vs. 다국어 모델 비교

