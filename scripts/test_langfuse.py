import os
from dotenv import load_dotenv

# Load env variables first, before importing Langfuse / LangChain
load_dotenv()

from langfuse.langchain import CallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def test_tracing():
    print("Initializing Langfuse CallbackHandler...")
    # CallbackHandler automatically reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST/BASE_URL from env
    langfuse_handler = CallbackHandler()

    print("Initializing Gemini LLM (gemini-2.5-flash)...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,
    )

    print("Sending a simple message to Gemini and tracing it via Langfuse...")
    message = "Hello, I am testing the RAG setup and Langfuse tracing. Say hello back!"
    
    response = llm.invoke(
        [HumanMessage(content=message)],
        config={"callbacks": [langfuse_handler]}
    )
    
    print("\nResponse from Gemini:")
    print(response.content)
    
    print("\nFlushing Langfuse traces to ensure they are fully sent...")
    # Best practice for short-lived scripts: flush via a Langfuse client instance
    from langfuse import Langfuse
    Langfuse().flush()
    print("Trace successfully sent to Langfuse!")

if __name__ == "__main__":
    test_tracing()
