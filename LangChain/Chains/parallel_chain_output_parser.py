from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

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
llm3 = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.3,
)

prompt1 = PromptTemplate(
    template="Give me 3 interesting facts about {city} in bullet points", 
    input_variables=["city"]
)
prompt2 = PromptTemplate(
    template="List top 5 tourist attractions in {city}. Return as a numbered list.", 
    input_variables=["city"]
)
prompt3 = PromptTemplate(
    template="Now gave me a short summary about the {facts} and {attractions} and {dishes}?",
    input_variables=["facts", "attractions", "dishes"]
)

# ---- NEW: Add a chain for dishes ----
dishes_prompt = PromptTemplate(
    template="What are 3 must-try local dishes in {city}? List with short descriptions.",
    input_variables=["city"]
)
dishes_chain = dishes_prompt | llm3 | output_parser   

# ---- Parallel chain: now includes 'dishes' ----
parallel_chain = RunnableParallel({
    'facts': prompt1 | llm1 | output_parser,
    'attractions': prompt2 | llm2 | output_parser,
    'dishes': dishes_chain,               # <-- added this line
})

merge_chain = prompt3 | llm3 | output_parser
chain = parallel_chain | merge_chain

result = chain.invoke(
    {"city": "Multan"},
    config={"callbacks": [langfuse_handler]}
)
print(result)
langfuse_handler._langfuse_client.flush()
chain.get_graph().print_ascii()