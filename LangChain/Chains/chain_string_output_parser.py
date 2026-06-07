from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler   
import os
load_dotenv()

langfuse_handler = CallbackHandler()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)

parser = StrOutputParser()
prompt = PromptTemplate(
    template="Give me five interesting facts about {place}",
    input_variables=["place"],
)

chain = prompt | llm | parser

final_result = chain.invoke(
    {"place": "Haroonabad, Bahawalnagar, Punjab"},
    config={"callbacks": [langfuse_handler]}   
)
print(final_result)
chain.get_graph().print_ascii()
langfuse_handler._langfuse_client.flush()