
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 예시 데이터 3개
examples = [
    {
        "news": "삼성전자가 AI 가속기를 출시하여 엔비디아와 경쟁한다.",
        "keywords": "삼성전자, AI가속기, 엔비디아"
    },
    {
        "news": "WHO가 건강 위기 대응을 위한 국제 협력을 강조했다.",
        "keywords": "WHO, 건강위기, 국제협력"
    },
    {
        "news": "정부가 전기차 구매 보조금 지원을 확대한다.",
        "keywords": "정부, 전기차, 구매보조금"
    }
]

# 예시 프롬프트
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{news}"),
    ("ai", "키워드: {keywords}")
])

# Few-Shot 구성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# 시스템 메시지와 Few-Shot 결합
final_prompt = ChatPromptTemplate.from_messages([
    ("system", """뉴스 키워드 추출 전문가입니다.
핵심 키워드 3개만 추출하세요.
반드시 '키워드: 단어1, 단어2, 단어3' 형식으로 답하세요."""),
    few_shot_prompt,
    ("human", "{input}")
])

# 모델과 체인
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
chain = final_prompt | llm | StrOutputParser()

# 테스트 뉴스
test_news = """
제미나이 2.0 플래시는 현재 구글 AI 스튜디오 및 버텍스 AI에서
제미나이 API를 통해 개발자에게 실험 모델로 제공됩니다.
모든 개발자는 멀티모달 입력 및 텍스트 출력을 사용할 수 있으며,
텍스트 음성 변환 및 네이티브 이미지 생성은 일부 파트너들을 대상으로
제공됩니다. 내년 1월에는 더 많은 모델 사이즈와 함께
일반에 공개될 예정입니다.
"""

# 실행
result = chain.invoke({"input": test_news})
print(result)

# 다양한 뉴스 분야 테스트
test_news_list = [
    "한국은행이 물가 안정을 위해 기준금리를 동결했다.",
    "정부가 신재생에너지 발전 시설 지원을 확대했다.",
    "한국 축구 국가대표팀이 국제대회에서 승리했다."
]

for news in test_news_list:
    print("\n뉴스:", news)
    print(chain.invoke({"input": news}))
