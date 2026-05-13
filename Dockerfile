# 1. 베이스 이미지 설정
FROM python:3.11-slim

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. 필요한 패키지 설치
RUN apt-get update && apt-get install -y libpq-dev gcc
RUN pip install openai python-dotenv psycopg[binary] pgvector numpy

# 4. 소스 코드 복사
COPY . .

# 5. 실행 명령 (실행 시점에 DB 연결 대기를 위해 바로 실행하지 않거나 래퍼 스크립트 권장)
CMD ["python", "main.py"]