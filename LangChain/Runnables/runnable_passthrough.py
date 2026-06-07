from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    temperature=0.2
)
parser = StrOutputParser()

# Chain that keeps the original question while generating an answer
prompt = PromptTemplate(
    template="Answer the question: {question}\n\nContext: {context}",
    input_variables=["question", "context"]
)

# Simulate a retriever that returns static context - can be replaced with real DB
def fake_retriever(question):
    return f"Relevant info about '{question}'"

# RunnablePassthrough.assign() adds new keys without removing old ones
chain = RunnablePassthrough.assign(
    context=lambda x: fake_retriever(x["question"])
) | prompt | llm | parser

result = chain.invoke(
    {"question": "What is LangChain?"}
)
print("Answer:", result)