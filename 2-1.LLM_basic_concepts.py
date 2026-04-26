import os
from dotenv import load_dotenv

# .env 파일에 정의된 환경 변수를 로드합니다.
load_dotenv()

# os.environ을 통해 키를 가져옵니다.
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# ----------------------------------------

# 토큰 직접 카운팅
# Anthropic
import anthropic
client = anthropic.Anthropic(api_key=anthropic_key)
response = client.messages.count_tokens(
    model="claude-sonnet-4-5",
    messages=[{"role":"user", "content":"안녕하세요"}]
)
print(f"Anthropic 토큰 수 : {response.input_tokens}")

# 토큰 수 : 14
# 이유 : anthropic은 <role> 등 태그들도 포함해서 token을 계산함.
# 한국어 효율이 떨어짐
# count_tokens는 API 포맷 전체 계산


# OpenAI - tiktoken 라이브러리
# tiktoken은 로컹 인코딩 방식이라 OpenAI 키가 없어도 실행 가능
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("안녕하세요")
print(f"OpenAI (gpt-4o) 토큰 수 : {len(tokens)}")

# 토큰 수 : 2
# 이유 : messages 리스트 구조를 포함하지 않음.
# 한국어 효율성이 높음.
# encode()는 텍스트만 계산

# ----------------------------------------

# Anthropic 형식
client.messages.create(
    model="claude-sonnet-4-5",
    system="You are a senior Growth PM assistant. Always respond in Korean.",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=1024,
)

# OpenAI 형식 (system이 messages 안에 들어감)
client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are..."},
        {"role": "user", "content": "..."}
    ]
)

# ----------------------------------------

# 1-7. Streaming (실시간 응답)
# 서버 측 구현 방식 : Server-Sent Events(SSE) / Chunked transfer

# Anthropic Streaminig
with client.messages.stream(
    model = "claude-sonnet-4-5",
    max_tokens = 1024,
    messages = [{"role" : "user", "content" : "긴 답변 부탁"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# end="" : 글자가 강제로 띄어쓰기가 되는데, 그걸 방지
# flush : output buffer를 강제로 비워 화면에 바로 글자 나오도록

# FastAPI에서 LLM 응답 스트리밍
'''
from fastapi.responses import StreamingResponse

@app.post("/chat")
def chat(req: ChatReq):
    def event_generator():
        with client.messages.stream(...) as stream:
            for text in stream.text_stream:
                yield f"data: {text}\n\n"
    return StreamResponse(event_generator(), media_type="text/event-stream")
'''

# ----------------------------------------

# 1-8. Toole Use / Function Calling
tools = [{
    "name" : "get-weather",
    "description" : "도시 날씨를 조회합니다.",
    "input_schema" : {
        "type" : "object",
        "properties" : {
            "city" : {"type" : "string"},
            "date" : {"type" : "string"}
        },
        "required" : ["city"]
    }
}]

response = client.messages.create(
    model = "claude-sonnet-4-5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "내일 서울 날씨"}]
)
# response.stop_reason == "tool_use"면 도구 호출 발생

# ----------------------------------------

# 1-9. Structured Output
from pydantic import BaseModel
class Sentiment(BaseModel):
    label: str
    score: float

response = client.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format=Sentiment
)
sentiment = response.choices[0].message.parsed  # Pydantic 객체!

# ----------------------------------------

# 1-10. Idempotency, Retry, Rate Limit — Production 필수
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import anthropic

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
    ))
)
def call_llm(prompt: str) -> str:
    response = client.messages.create(...)
    return response.content[0].text

'''
python 2_llm.py
'''
