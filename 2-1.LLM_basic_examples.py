# 예시 1
import os
import anthropic
from dotenv import load_dotenv
from openai import APIConnectionError

load_dotenv()  # .env에서 ANTHROPIC_API_KEY 로드

client = anthropic.Anthropic()  # SDK는 내부적으로 환경 변수를 확ㅇ니하고, 자동 할당을 진행함.

response = client.messages.create(
    model="claude-sonnet-4-5",  # 실제 사용 시 docs.claude.com 에서 최신 모델명 확인
    max_tokens=1024,
    system="당신은 한국어로 답하는 Growth PM 어시스턴트입니다.",
    messages=[
        {"role": "user", "content": "DAU와 WAU의 차이를 한 문장으로 설명해줘"}
    ]
)

print(response.content[0].text)
print(f"\n사용 토큰: input={response.usage.input_tokens}, output={response.usage.output_tokens}")

# DAU(Daily Active Users)는 하루 동안 서비스를 이용한 순 사용자 수이고, WAU(Weekly Active Users)는 일주일 동안 서비스를 이용한 순 사용자 수입니다.
# 사용 토큰: input=57, output=73

# ======================================================
# 예시 2. Multi-turn 대화 직접 관리

import anthropic

client = anthropic.Anthropic()
conversation = []

def chat(user_input: str) -> str:
    conversation.append({"role": "user", "content": user_input})
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system="당신은 데이터 분석 도우미입니다.",
        messages=conversation
    )
    assistant_msg = response.content[0].text
    conversation.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg

print(chat("MAU가 뭐야?"))
print(chat("그럼 DAU/MAU 비율은 무슨 의미인지?"))  # 이전 맥락 기억
print(f"\n총 턴: {len(conversation)//2}")

'''
MAU는 **Monthly Active Users**의 약자로, **월간 활성 사용자 수**를 의미합니다.

## 주요 특징

- **정의**: 특정 한 달 동안 서비스나 앱을 한 번이라도 사용한 고유 사용자의 수
- **측정 기간**: 보통 30일 또는 달력상 1개월 단위로 집계

## 활용 목적

1. **서비스 성장 지표**: 사용자 기반의 성장세를 파악
2. **참여도 측정**: 실제로 서비스를 이용하는 사용자 규모 확인
3. **비즈니스 가치 평가**: 투자자나 이해관계자에게 제시하는 핵심 지표

## 관련 지표

- **DAU** (Daily Active Users): 일간 활성 사용자
- **WAU** (Weekly Active Users): 주간 활성 사용자
- **Stickiness (점착도)**: DAU/MAU 비율로 사용자 충성도 측정

예를 들어, 소셜미디어 앱이 MAU 500만 명이라면, 한 달 동안 500만 명의 고유 사용자가 앱을 사용했다는 의미입니다.
DAU/MAU 비율은 **사용자 점착도(Stickiness)** 또는 **사용자 참여도**를 나타내는 핵심 지표입니다.

## 계산 방법
```
DAU/MAU 비율 = (일간 활성 사용자 / 월간 활성 사용자) × 100%
```

## 의미 해석

**높은 비율 (높을수록 좋음)**
- 사용자들이 자주, 규칙적으로 서비스를 이용
- 서비스에 대한 의존도와 충성도가 높음
- 습관적인 사용 패턴 형성

**낮은 비율**
- 가끔씩만 방문하는 사용자가 많음
- 사용자 참여도가 낮음
- 서비스 개선 필요

## 실제 예시

| MAU | DAU | DAU/MAU | 해석 |
|-----|-----|---------|------|
| 1,000만 | 500만 | 50% | 월 사용자의 절반이 매일 사용 (매우 높음) |
| 1,000만 | 200만 | 20% | 평균적으로 주 1-2회 사용 (보통) |
| 1,000만 | 50만 | 5% | 월 1-2회 정도만 사용 (낮음) |

## 산업별 벤치마크

- **소셜미디어/메신저**: 50-60% (페이스북, 인스타그램)
- **게임**: 15-25%
- **전자상거래**: 5-10%
- **생산성 도구**: 30-40%

이 지표가 높다는 것은 사용자들이 서비스를 **일상의 일부**로 받아들였다는 강력한 신호입니다!

총 턴: 2
'''

# ======================================================
# 예시 3. Streaming 출력
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "Growth PM이 알아야 할 핵심 지표 5개를 자세히 설명해줘."
    }]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    print()
    # end="" : 글자가 강제로 띄어쓰기가 되는데, 그걸 방지
    # flush : output buffer를 강제로 비워 화면에 바로 글자 나오도록


    # streaming 종료 후 메타정보
    final = stream.get_final_message()
    print(f"\n총 토큰: {final.usage.output_tokens}")

'''
MAU는 **Monthly Active Users**의 약자로, **월간 활성 사용자 수**를 의미합니다.

## 주요 특징

- **정의**: 특정 한 달 동안 서비스나 앱을 1회 이상 사용한 순수 사용자 수
- **중복 제거**: 같은 사용자가 여러 번 접속해도 1명으로 계산
- **측정 기간**: 보통 최근 30일 또는 해당 월(1일~말일)

## 함께 사용되는 지표

- **DAU** (Daily Active Users): 일간 활성 사용자
- **WAU** (Weekly Active Users): 주간 활성 사용자
- **DAU/MAU 비율**: 사용자 참여도(Stickiness)를 측정
  - 예: DAU/MAU = 0.5 → 월간 사용자가 평균적으로 50%의 날에 접속

## 활용 사례

- 서비스 성장세 파악
- 마케팅 캠페인 효과 측정
- 투자 유치 시 핵심 지표로 활용

특히 소셜미디어, 게임, SaaS 등의 디지털 서비스에서 중요하게 추적하는 KPI입니다.
DAU/MAU 비율은 **사용자 참여도(Stickiness, 끈끈함)**를 측정하는 핵심 지표입니다.

## 기본 의미

**"월간 사용자 중 매일 얼마나 많은 사용자가 돌아오는가?"**

- 비율이 높을수록 = 사용자들이 자주, 습관적으로 서비스를 이용
- 비율이 낮을수록 = 가끔씩만 방문하는 사용자가 많음

## 계산 예시

```
MAU = 1,000명
DAU = 300명
DAU/MAU = 0.3 (30%)
```
→ 월간 사용자가 평균적으로 **한 달에 9일 정도** 방문 (30일 × 0.3)

## 산업별 기준

| 서비스 유형 | 일반적 비율 | 의미 |
|------------|-----------|------|
| **소셜미디어** | 50-60% | 매일 확인하는 습관 |
| **메신저** | 60-80% | 일상 필수 도구 |
| **게임** | 20-30% | 주기적 참여 |
| **전자상거래** | 5-15% | 필요시 방문 |

## 왜 중요한가?

1. **유지율 판단**: 높은 비율 = 강한 사용자 유지력
2. **제품 가치**: 자주 찾는 서비스 = 실제 가치 제공
3. **수익성 예측**: 활성도 높음 = 수익화 가능성 높음
4. **이탈 조기 감지**: 비율 하락 = 문제 신호

단순히 사용자 수가 많은 것보다, **얼마나 자주 돌아오는지**가 서비스의 진짜 건강도를 보여줍니다.

총 턴: 2
'''

# ======================================================
# 예시 4. Temperature

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
prompt = "새 SaaS 제품의 슬로건 1개를 만들어줘"

for temp in [0.0, 0.3, 0.7, 1.0]:
    print(f"\n--- Temperature = {temp} ---")
    for trial in range(3):  # 같은 설정으로 3번 호출
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=100,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"  [{trial+1}] {response.content[0].text.strip()}")

'''
--- Temperature = 0.0 ---
  [1] # 어떤 SaaS 제품인지 알려주시면 더 좋은 슬로건을 만들 수 있어요!

제품의 특징을 간단히 알려주세요:
- 어떤 문제를 해결하나요?
- 주요 타겟은 누구인가요?
- 핵
  [2] # 어떤 SaaS 제품인지 알려주시면 더 좋은 슬로건을 만들 수 있어요!

제품의 특징을 간단히 알려주세요:
- 어떤 문제를 해결하나요?
- 주요 타겟은 누구인가요?
- 핵
  [3] # 어떤 SaaS 제품인지 알려주시면 더 좋은 슬로건을 만들 수 있어요!

제품의 특징을 간단히 알려주세요:
- 어떤 문제를 해결하나요?
- 주요 타겟은 누구인가요?
- 핵

--- Temperature = 0.3 ---
  [1] # 제품 정보가 필요합니다

효과적인 슬로건을 만들기 위해 몇 가지 정보를 알려주세요:

1. **어떤 SaaS 제품인가요?** (예: 프로젝트 관리, CRM, 회계 등)
2. **주요 타겟
  [2] # 어떤 SaaS 제품인지 알려주시면 더 좋은 슬로건을 만들 수 있어요!

제품의 특징을 간단히 알려주세요:
- 어떤 문제를 해결하나요?
- 주요 타겟은 누구인가요?
- 핵
  [3] # 어떤 SaaS 제품인지 알려주시면 더 좋은 슬로건을 만들 수 있어요!

제품의 특징을 간단히 알려주세요:
- 어떤 문제를 해결하나요?
- 주요 타겟은 누구인가요?
- 핵

--- Temperature = 0.7 ---
  [1] # 어떤 SaaS 제품인지 알려주시면 더 좋은 슬로건을 만들 수 있어요!

제품의 특징을 간단히 알려주세요:
- 어떤 문제를 해결하나요?
- 주요 타겟은 누구인가요?
- 핵
  [2] # "복잡함은 줄이고, 성과는 키우다"

이 슬로건은:
- **간결함**: 기억하기 쉽고 명확합니다
- **가치 제안**: SaaS의 핵심 가치인 '단순화'와 '효율성 향상'을 동시에 전달합
  [3] # 제품 정보가 필요해요

슬로건을 만들기 위해 몇 가지 정보를 알려주시겠어요?

1. **어떤 SaaS 제품인가요?** (예: 프로젝트 관리, CRM, 회계 등)
2. **주요 타겟 고객

--- Temperature = 1.0 ---
  [1] # 제품 정보가 필요합니다

SaaS 제품의 슬로건을 만들기 위해 몇 가지 정보를 알려주시면 더 적합한 슬로건을 제안해드릴 수 있습니다:

1. **어떤 종류의 SaaS인가요?** (예
  [2] # 어떤 SaaS 제품인지 알려주시면 더 좋은 슬로건을 만들 수 있어요!

제품의 특징을 간단히 알려주세요:
- 어떤 문제를 해결하나요?
- 주요 타겟은 누구인가요?
- 핵
  [3] # 어떤 SaaS 제품인지 알려주시면 더 정확한 슬로건을 만들어드릴게요!

예시로 몇 가지 만들어드리면:

**프로젝트 관리 툴이라면:**
"복잡한 업무, 심플한 해결"

**고객
'''

# ======================================================
# 예시 5. Tool Use 예시

import anthropic
import json

client = anthropic.Anthropic()

# 1) 도구 정의
tools = [{
  "name": "get_dau",  # 기능을 써야 하는 시기 판별
  "description": "특정 날짜의 DAU(Daily Active Users)를 반환합니다", # 언제 도구를 사용할지 판단
  "input_schema": { # 규격서 (JSON Schema)
    "type": "object", # 키-값 쌍인 객체
    "properties": {
      "date": {"type": "string", "description": "YYYY-MM-DD 형식"}
    },
    "required": ["date"]  # 무조건 있어야 하는 정보
  }
}]

# 2) 실제 함수 (보통은 DB/API 호출)
def get_dau(date: str) -> int:
    fake_data = {"2026-04-23": 12500, "2026-04-24": 13100, "2026-04-25": 13400}
    return fake_data.get(date, 0)

# 3) 첫 호출
messages = [{"role": "user", "content": "2026년 4월 24일 DAU 알려줘"}]
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=1024,
    tools=tools,
    messages=messages
)

# 4) 도구 호출 감지 → 실행 → 결과 전달
while response.stop_reason == "tool_use":
    tool_use = next(b for b in response.content if b.type == "tool_use")
    tool_name = tool_use.name
    tool_input = tool_use.input
    print(f"🔧 LLM이 호출 요청: {tool_name}({tool_input})")

    # 실행
    if tool_name == "get_dau":
        result = get_dau(**tool_input)

    # 대화 히스토리에 LLM의 tool_use + 우리의 tool_result 추가
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": str(result)
        }]
    })

    # 재호출
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

# 5) 최종 텍스트 응답
final_text = next(b.text for b in response.content if b.type == "text")
print(f"\n💬 최종 응답: {final_text}")


# ======================================================
# 예시 6. Pydantic + Instructor로 신뢰 가능한 JSON
# 단순히 문장을 제공하는 방식이 아닌, 미리 정의한 인터페이스에 맞춰 강제로 개발하게 함.

# pip install instructor anthropic
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field
from typing import Literal

client = instructor.from_anthropic(Anthropic())

class FeedbackAnalysis(BaseModel):  # AI가 출력해야 하는 데이터 규격
  sentiment: Literal["positive", "negative", "neutral"] # 피드백 : 감정 상태
  main_topic: str = Field(..., description="피드백의 핵심 주제 한 단어")  # 피드백 핵심 주제
  severity: int = Field(..., ge=1, le=1, description="심각도 1~5")  # 심각도 점수화
  suggested_action: str                                           # AI가 추천하는 답변

result = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=512,
    messages=[{
        "role": "user",
        "content": "결제 페이지에서 자꾸 오류가 납니다. 신용카드 결제 안 되네요. 답답해요"
    }],
    response_model = FeedbackAnalysis
)

print(result.model_dump_json(indent=2))
# {
#   "sentiment": "negative",
#   "main_topic": "결제오류",
#   "severity": 4,
#   "suggested_action": "결제 모듈 긴급 점검 및 사용자 사과 회신"
# }

# 왜 강력한가? 타입 안정성 + 검증 + IDE 자동완성, 프로덕션 LLM 코드의 표준 패턴


# ======================================================
# 예시 7. Retry with Exponential Backoff
# tenacity : API 불안정성(네트워크 끊김, 500 에러 등) 문제가 발생하는 경우, 재시도 로직

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time

client = anthropic.Anthropic()

@retry(
  stop=stop_after_attempt(3), # 재시도 횟수 : 3번
  wait=wait_exponential(multiplier=1, min=2, max=10), # 지수적으로 증가 
  retry=retry_if_exception_type((
    anthropic.APIStatusError,
    anthropic.APIConnectionError,
  )),
    before_sleep=lambda rs: print(f"⏳ 재시도 {rs.attempt_number}... 대기 {rs.next_action.sleep:.1f}s")
)

def safe_complete(prompt: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

print(safe_complete("안녕"))

# ======================================================
# 예시 8. 토큰 카운팅 & 비용 추정
# 함수 호출 자체는 비용이 들지 않거나, 매우 저렴해 예측 용도로 사용함.

import anthropic
client = anthropic.Anthropic()

INPUT_PRICE_PER_MTOK = 3.0  # 가상의 가격, 실제 docs.claude.com 확인
OUTPUT_PRICE_PER_MTOK = 15.0

def estimate_cost (input_text: str, expected_output_tokens: int = 500) -> dict:
  count = client.messages.count_tokens(
    model="claude-sonnet-4-5",
    messages=[{"role":"user", "content": input_text}]
  )
  in_cost = count.input_tokens / 1_000_000 * INPUT_PRICE_PER_MTOK
  out_cost = expected_output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK
  return {
    "input_tokens": count.input_tokens,
    "expected_output_tokens": expected_output_tokens,
    "estimated_cost_usd": round(in_cost + out_cost, 6)
  }

print(estimate_cost("우리 서비스 DAU 추이를 분석해줘. " * 50))


