
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
parser = StrOutputParser()

# 1단계: 관광 명소 추천
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "여행 전문가입니다. 대표 관광 명소 1개의 이름만 답하세요."),
    ("human", "{city}의 대표 관광 명소를 추천해주세요.")
])

chain1 = prompt1 | llm | parser

# 2단계: 관광 명소 상세 정보
prompt2 = ChatPromptTemplate.from_messages([
    ("system", """관광 가이드입니다.
명소의 역사, 특징, 방문 팁을 각각 설명하세요."""),
    ("human", "명소: {place}")
])

chain2 = prompt2 | llm | parser

# LCEL로 두 체인 연결
multi_chain = (
    chain1
    | RunnableLambda(lambda place: {"place": place})
    | chain2
)

# 각 단계 결과 출력
city = input("여행지를 입력하세요: ")

place = chain1.invoke({"city": city})
print("\n1단계 결과:", place)

detail = chain2.invoke({"place": place})
print("\n2단계 결과:")
print(detail)

# 전체 체인만 실행하려면 다음을 사용
# result = multi_chain.invoke({"city": city})
