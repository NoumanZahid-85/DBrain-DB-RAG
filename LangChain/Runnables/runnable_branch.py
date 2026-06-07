from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2
)

class Sentiment(BaseModel):
    label: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(description="Sentiment label"
)

parser = PydanticOutputParser(pydantic_object=Sentiment)

classify_prompt = PromptTemplate(
    template="Classify sentiment of: '{text}'\n{format}",
    input_variables=["text"],
    partial_variables={"format": parser.get_format_instructions()}
)

# Branch handlers
positive_chain = PromptTemplate(
    template="Write a cheerful reply to: {original}",
    input_variables=["original"]
    ) | llm | StrOutputParser()
negative_chain = PromptTemplate(
    template="Write an apologetic reply to: {original}",
    input_variables=["original"]
    ) | llm | StrOutputParser()
neutral_chain = PromptTemplate(
    template="Write a neutral acknowledgment to: {original}",
    input_variables=["original"]
    ) | llm | StrOutputParser()

branch = RunnableBranch(
    (lambda x: x.label == "POSITIVE", positive_chain),
    (lambda x: x.label == "NEGATIVE", negative_chain),
    (lambda x: x.label == "NEUTRAL", neutral_chain),
    (lambda x: True, neutral_chain) 
)

full_chain = classify_prompt | llm | parser | branch

result = full_chain.invoke(
    {"text": "I am so happy with your service!"}
)

print("Response:", result)