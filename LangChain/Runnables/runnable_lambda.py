from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    temperature=0.3
)
parser = StrOutputParser()

# Custom cleaning function
def clean_text(text: str) -> str:
    # Remove extra whitespace and convert to uppercase for demonstration
    return " ".join(text.split()).upper()

def word_count(text: str) -> str:
    words = len(text.split())
    return f"\n[Word count: {words}]"

# Chain: prompt -> LLM -> parser -> clean -> word_count
prompt = PromptTemplate(
    template="Write a short tweet about {topic}", 
    input_variables=["topic"]
) 

chain = (
    prompt 
    | llm 
    | parser 
    | RunnableLambda(clean_text) 
    | RunnableLambda(word_count)
)
result = chain.invoke({"topic": "artificial intelligence"})