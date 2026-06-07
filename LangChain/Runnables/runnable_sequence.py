from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

langfuse_handler = CallbackHandler()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2
)
parser = StrOutputParser()

# Step 1: Generate a short story idea
prompt1 = PromptTemplate(
    template="Generate a one-sentence story idea about {topic}",
    input_variables=["topic"]
)

# Step 2: Expand that idea into a 3-sentence story
prompt2 = PromptTemplate(
    template="Expand the following story idea into a 3-sentence story:\n{idea}",
    input_variables=["idea"]
)

# RunnableSequence automatically chains: prompt1 -> llm -> parser -> prompt2 -> llm -> parser
chain = RunnableSequence(prompt1 | llm | parser | prompt2 | llm | parser)

result = chain.invoke(
    {"topic": "a lost cat in a cyberpunk city"},
    config={"callbacks": [langfuse_handler]}
)
print("Final Story:\n", result)
langfuse_handler._langfuse_client.flush()