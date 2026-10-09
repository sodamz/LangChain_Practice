
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 프롬프트
prompt = PromptTemplate.from_template("""
당신은 AI 요리사입니다.
다음 재료로 만들 수 있는 요리를 하나 추천해주세요.

재료: {ingredients}

다음 형식으로 답변하세요.
추천 요리:
재료:
조리법:
1.
2.
3.
""")

# AI 모델
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 체인 구성
chain = prompt | llm | StrOutputParser()

# 실행
ingredients = input("재료를 입력하세요: ")
result = chain.invoke({"ingredients": ingredients})

print("\n", result)
