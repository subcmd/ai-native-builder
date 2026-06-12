# 2. 예시 (Python 실전 코드)

# 예시 1. OpenAI 임베딩 + 코사인 유사도

import chromadb
from chromadb.utils import embedding_functions
import anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# 1) 인덱싱
chroma = chromadb.PersistentClient(path="./naive_rag_db")
ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
col = chroma.get_or_create_collection("kb", embedding_function=ef)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

def index_file(path: str):
    text = Path(path).read_text(encoding="utf-8")
    chunks = splitter.split_text(text)
    col.add(
        documents=chunks,
        metadatas=[{"source": path, "chunk_idx": i} for i in range(len(chunks))],
        ids=[f"{path}::{i}" for i in range(len(chunks))]
    )

# 2) 검색 + 생성
client = anthropic.Anthropic()

def rag(query: str, k: int = 3) -> dict:
    # Retrieve
    results = col.query(query_texts=[query], n_results=k)
    chunks = results["documents"][0]
    sources = results["metadatas"][0]
    
    # Augment
    context = "\n\n".join(
        f"[Source: {m['source']}]\n{c}" for c, m in zip(chunks, sources)
    )
    
    prompt = f"""<context>
{context}
</context>

<question>
{query}
</question>

<instructions>
- 컨텍스트만 근거로 답하시오
- 출처를 [Source: 파일명] 형식으로 인용
- 컨텍스트에 없는 내용은 "문서에 해당 정보가 없습니다" 로 답
</instructions>"""
    
    # Generate
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "answer": resp.content[0].text,
        "sources": [m["source"] for m in sources]
    }

# 사용
for f in Path("./docs").glob("*.md"):
    index_file(str(f))

result = rag("DAU 와 MAU 차이는?")
print(result["answer"])
print("출처:", result["sources"])


# =============================================================================
# 예시 2. 로컹 임베딩(BGE-m3)


# =============================================================================
# 예시 3. Chroma 기본 사용



# =============================================================================
# 예시 4. pgvector 기본 사용


# =============================================================================
# 예시 5. 미니 시맨틱 검색 시스템



# =============================================================================
# 예시 6. 임베딩 시각화 (UMAP)



# =============================================================================
# 예시 7. 한국어 모델 vs. 다국어 모델 비교

