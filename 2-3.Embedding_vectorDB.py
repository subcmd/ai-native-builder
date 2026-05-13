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
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# Docker로 pgvector 띄우기
# docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17

# pip install psycopg pgvector
import psycopg
from pgvector.psycopg import register_vector
import numpy as np


def embed(text: str) -> np.ndarray:
    return np.array(oai.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding)

# 연결
db_host = os.getenv("DB_HOST", "localhost")
conn = psycopg.connect(f"postgresql://postgres:postgres@{db_host}:5432/postgres", autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)

# 테이블
conn.execute("""
CREATE TABLE IF NOT EXISTS docs (
    id SERIAL PRIMARY KEY,
    content TEXT,
    category TEXT,
    embedding vector(1536)
)""")

# 적재
docs = [
    ("DAU와 WAU의 차이", "metric"),
    ("Funnel 분석 방법", "analysis"),
    ("RFM 세그멘테이션", "segmentation"),
]
for content, category in docs:
    emb = embed(content)
    conn.execute(
        "INSERT INTO docs (content, category, embedding) VALUES (%s, %s, %s)",
        (content, category, emb)
    )

# HNSW 인덱스 생성
conn.execute("""
CREATE INDEX IF NOT EXISTS docs_embedding_idx 
ON docs USING hnsw (embedding vector_cosine_ops)
""")

# 검색 (메타데이터 필터 + 벡터 유사도)
query_emb = embed("유저 그룹화")
results = conn.execute("""
SELECT content, category, embedding <=> %s AS distance
FROM docs
WHERE category IN ('segmentation', 'analysis')
ORDER BY distance
LIMIT 3
""", (query_emb,)).fetchall()

for content, category, dist in results:
    print(f"distance={dist:.3f} [{category}] {content}")


# =============================================================================
# 예시 5. 미니 시맨틱 검색 시스템

import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"



# 디렉터리의 .md 파일을 모두 인덱싱하는 미니 RAG 사전작업
client = chromadb.PersistentClient(path="./mini_search_db")  # PersistentClient: 로컬시스템에서 데이터 저장 및 불러옴
ef = embedding_functions.OpenAIEmbeddingFunction(  # 임베딩 함수: 텍스트를 컴퓨터가 계산할 수 있는 숫자 리스트(벡터)로 변환 (번역기)
    api_key=os.environ["OPENAI_API_KEY"]
)
col = client.get_or_create_collection("knowledge_base", embedding_function=ef)  # 컬렉션: 데이터를 담는 하나의 폴더 또는 테이블

def index_specific_file(file_path: str):
    """
    1. 로드(load): 지정된 경로에서 마크다운(.md) 파일 읽기
    2. 청킹(chunking): 긴 글을 한 번에 저장하면 검색 효율이 떨어지므로, 800 - 1000자 단위로 조각(chunk)
    3. 저장(Upsert): 각 조각에 ID와 파일명 등의 메타데이터를 붙여 ChromaDB에 저장
    -> OpenAI가 각 조각의 의미(벡터)를 계산해서 함께 저장
    """
    path_obj = Path(file_path)  # 파일 탐색
    
    if not path_obj.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    docs, metas, ids = [], [], []   # 실제 텍스트 내용(청크) / 데이터 부가 정보(출처 등) / 각 조각의 고유 이름표(중복 불가)
    text = path_obj.read_text(encoding="utf-8").strip()
    
    if not text:
        print(f"⚠️ 파일이 비어 있습니다: {path_obj.name}")
        return

    # 단순 청킹
    for i in range(0, len(text), 800):
        chunk = text[i:i+1000]
        docs.append(chunk)
        metas.append({"file": path_obj.name, "offset": i})
        ids.append(f"{path_obj.name}::{i}")
    
    col.add(documents=docs, metadatas=metas, ids=ids)
    print(f"✅ 성공: '{path_obj.name}' 파일에서 {len(docs)}개 청크 생성 및 인덱싱 완료")

def search(query: str, k: int = 3):
    """
    1. 사용자 질문을 벡터로 변환
    2. DB에 저장된 chunk 중 가장 가까운 조각 k개를 찾음.
    """
    res = col.query(query_texts=[query], n_results=k)
    if not res["documents"][0]:
        print("🔍 검색 결과가 없습니다.")
        return

    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"\n📄 {meta['file']} (offset={meta['offset']}, dist={dist:.3f})")
        print(doc[:200] + "...")

# --- 실행부 ---
# 요청하신 경로를 원시 문자열(r)로 처리하여 경로 인식을 확실하게 합니다.
target_file = r"C:\Users\ST-USER\AI-Native-Builder\examples.md\ab_test_guide.md"

index_specific_file(target_file)
search("A/B 테스트 표본 크기 계산법")

'''
✅ 성공: 'ab_test_guide.md' 파일에서 1개 청크 생성 및 인덱싱 완료

📄 ab_test_guide.md (offset=0, dist=0.107)
# A/B 테스트와 표본 크기(Sample Size) 계산법

A/B 테스트에서 가장 빈번하게 발생하는 실수 중 하나는 결과가 나올 때까지 무작정 기다리거나, 너무 적은 표본으로 성급하게 결론을 내리는 것입니다. 이를 방지하기 위해서는 테스트 시작 전 **최소 표본 크기**를 설정해야 합니다.

## 1. 표본 크기 결정의 3가지 핵심 요소

1. **기본...
'''

# =============================================================================
# 예시 6. 임베딩 시각화 (UMAP)

import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv

# 1. 환경 설정 및 클라이언트 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Matplotlib 한글 깨짐 방지 (로컬 시각화용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 2. 데이터 준비
sentences = [
    *[f"DAU 지표 및 사용자 분석 보고서 {i}" for i in range(20)],
    *[f"분기 매출 실적 및 재무 통계 {i}" for i in range(20)],
    *[f"불만 사항 및 고객 서비스 피드백 {i}" for i in range(20)],
    *[f"API 연동 및 시스템 아키텍처 기술 문서 {i}" for i in range(20)],
    *[f"SNS 광고 집행 및 마케팅 전략 {i}" for i in range(20)],
]
labels = ["DAU"]*20 + ["매출"]*20 + ["피드백"]*20 + ["기술"]*20 + ["마케팅"]*20

# 3. OpenAI 임베딩 추출 함수
def get_embeddings(text_list, model="text-embedding-3-small"):
    """
    텍스트 리스트를 입력받아 임베딩 벡터 리스트를 반환합니다.
    """
    print(f"{len(text_list)}개 문장 임베딩 생성 중...")
    response = client.embeddings.create(input=text_list, model=model)
    return np.array([data.embedding for data in response.data])

# 실제 API 호출 (비용 발생 주의)
try:
    embs = get_embeddings(sentences)
except Exception as e:
    print(f"API 호출 오류: {e}")
    # 오류 발생 시 테스트용 랜덤 데이터 생성
    embs = np.random.randn(100, 1536)

# 4. UMAP을 이용한 차원 축소 (1536D -> 2D)
print("차원 축소 진행 중...")
reducer = umap.UMAP(
    n_neighbors=15, 
    min_dist=0.1, 
    n_components=2, 
    random_state=42  # 결과 고정 (Warning은 무시하셔도 됩니다)
)
emb_2d = reducer.fit_transform(embs)

# 5. 시각화 (선택 1: Plotly 인터랙티브 시각화 - 추천)
# 마우스를 점 위에 올리면(hover) 실제 문장이 보입니다.
df = pd.DataFrame({
    'x': emb_2d[:, 0],
    'y': emb_2d[:, 1],
    'label': labels,
    'text': sentences
})

fig = px.scatter(
    df, x='x', y='y', color='label', hover_name='text',
    title="문서 임베딩 2D 시각화 (Interactive)",
    labels={'x': 'UMAP 1', 'y': 'UMAP 2'}
)
fig.show()

# 6. 시각화 (선택 2: Matplotlib 정적 시각화)
plt.figure(figsize=(12, 8))
unique_labels = list(set(labels))
colors = plt.cm.get_cmap('viridis', len(unique_labels))

for i, label in enumerate(unique_labels):
    mask = [l == label for l in labels]
    plt.scatter(
        emb_2d[mask, 0], 
        emb_2d[mask, 1], 
        label=label, 
        s=50, 
        alpha=0.7
    )

plt.legend()
plt.title("문서 임베딩 2D 시각화 (Static)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# → 카테고리별로 클러스터가 형성되는지 확인. \
# 클러스터 안 형성되면 임베딩 모델 부적합 신호.

# =============================================================================
# 예시 7. 한국어 모델 vs. 다국어 모델 비교

