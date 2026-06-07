from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

langfuse_handler = CallbackHandler()
output_parser = StrOutputParser()

llm1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
)
llm2 = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3,
)

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="Sentiment of the feedback")
    reason: str = Field(description="Reason for the sentiment")

parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template="Classify the sentimat of the following feedback text into Positive or Negative: \n {feedback} \n {format_instruction}", 
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

prompt_positive = PromptTemplate(
    template="Give a short response for the following positive feedback: \n {feedback}",
    input_variables=["feedback"]
)
prompt_negative = PromptTemplate(
    template="Give a short response for the following negative feedback: \n {feedback}",
    input_variables=["feedback"]
)
classifier_chain = prompt1 | llm1 | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt_positive | llm2 | output_parser),
    (lambda x: x.sentiment == "Negative", prompt_negative | llm2 | output_parser),
    RunnableLambda(lambda x: "Could not classify the sentiment.")
)

chain = classifier_chain | branch_chain

result = chain.invoke(
    {"feedback": "I love this product! It's amazing!"},
    config={"callbacks": [langfuse_handler]}
)
print(result)
chain.get_graph().print_ascii()
langfuse_handler._langfuse_client.flush()

