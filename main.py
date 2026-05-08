# 2. 예시 (Python 실전 코드)

# prompts/classify_feedback/v2.0.yaml 파일 가정

import yaml
import os
from pathlib import Path
from openai import OpenAI  # 최신 SDK 방식
from dotenv import load_dotenv

load_dotenv()
# OpenAI 클라이언트 인스턴스 생성

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PromptTemplate:
    def __init__(self, yaml_path: str):
        # 한글 깨짐 방지를 위해 encoding='utf-8' 명시
        self.config = yaml.safe_load(Path(yaml_path).read_text(encoding='utf-8'))

    def render(self, **vars) -> dict:
        messages = []
        # 시스템 메시지가 YAML에 있는 경우 추가
        if self.config.get("system"):
            messages.append({"role": "system", "content": self.config["system"].strip()})
        
        # 유저 템플릿 렌더링
        messages.append({
            "role": "user",
            "content": self.config["template"].format(**vars)
        })

        return {
            "model": self.config.get("model", "gpt-4o"),
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 500),
            "messages": messages
        }
    
    def __call__(self, **vars) -> str:
        params = self.render(**vars)
        # OpenAI v1.0+ API 호출 방식
        response = client.chat.completions.create(**params)
        # 응답 텍스트 추출
        return response.choices[0].message.content

if __name__ == "__main__":
    # 윈도우 경로 이스케이프 방지를 위한 r-string
    prompt = PromptTemplate(r"prompts\classify_feedback\v2.0.yaml")
    
    # 실제 호출
    try:
        result = prompt(feedback_text="결제 안 됩니다. 답답해요")
        print("\n=== OpenAI 분석 결과 ===")
        print(result)
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

'''
=== OpenAI 분석 결과 ===
<sentiment>negative</sentiment>
<category>bug</category>
<severity>3</severity>
'''