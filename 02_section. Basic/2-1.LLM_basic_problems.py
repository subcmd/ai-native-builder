# 연습문제
# 🟢 Lv.1 — 개념 & 기초 호출

import os
import anthropic
from dotenv import load_dotenv

# (a) .env 파일에 API 키 등록 + python-dotenv 로 로드
load_dotenv()

# (b) Claude Sonnet 으로 "Growth PM 의 정의를 50자로" 요청

client = anthropic.Anthropic()

response = client.messages.create(
    model = "claude-sonnet-4-5",
    max_tokens=200,
    messages=[{"role": "user", "content": "Growth PM 의 정의를 50자로"}]
)

print(response.content[0].text)
print(f"input = {response.usage.input_tokens}, output = {response.usage.output_tokens}, stop = {response.stop_reason}")

# (c) 응답 텍스트, input/output 토큰 수, stop_reason 출력
# input = 19, output = 72, stop = end_turn

# (d) 같은 프롬프트로 OpenAI도 호출해 비교
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()   # API 키 호출

openai_key = os.getenv("OPENAI_API_KEY")    # 명시적으로 API 키 호출

client = OpenAI(api_key=openai_key)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    max_tokens=200,
    messages=[{"role": "user", "content": "Growth PM 의 정의를 50자로"}]
)

# Anthropic: response.content[0].text
# OpenAI: response.choices[0].message.content
print(response.choices[0].message.content)

# 사용량 정보 및 종료 사유 출력
# Anthropic: usage.input_tokens / usage.output_tokens / stop_reason
# OpenAI: usage.prompt_tokens / usage.completion_tokens / finish_reason
print(f"input = {response.usage.prompt_tokens}, "
      f"output = {response.usage.completion_tokens}, "
      f"stop = {response.choices[0].finish_reason}")

'''
python 2-1.LLM_basic_problems.py
'''