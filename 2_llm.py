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



'''
python 2_llm.py
'''
