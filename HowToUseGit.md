# How to use github

## .gitignore 사용법
    "자동 생성 + 무거움 + 비밀" 등을 차단하기 위한 리스트
1. venv
2. 캐시
3. 환경변수
4. IDE


## Git 이용하는 방법 (초기화 ~ 업로드)
1. Git 초기화
    git init

2. Git 상태 확인
    git status

3. 파일 스테이징
    - 전체 파일
        git add .
    - 특정 파일만
        git add <파일명>

4. 커밋 (저장)
    git commit -m "함께 넣고 싶은 문장"

5. Github 저장소(repository)
    1) create repository
    2) https://github.com/SubCmd/원하는 리포지토리명/tree/main

6. 원격 저장소 연결
    git remote add origin "https://github.com/아이디/레포이름.git"

7. 브랜치 이름 설정
    git branch -M main

8. Github로 업로드 진행
    git push -u origin main

## Git 가져오는 방법
1. git clone
- 맨처음 가져올때
    git clone https://github.com/아이디/레포.git
    : 코드 + 커밋 기록 + 브랜치 전부 다운로드
    : git init, origin, 코드 다운로드 전부 진행
    (clone = init + remote + pull)

2. git pull
    - 이미 clone한 프로젝트
    - 다른 사람이 수정한 최신 코드 가져올때
    (git pull = fetch + merge)
    fetch : 최신 코드 가져오기
    merge : 내 코드와 합치기

git pull origin main


## Git 아이디 이메일 