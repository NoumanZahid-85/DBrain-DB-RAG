import os
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load env variables
load_dotenv()

# Initialize CallbackHandler
langfuse_handler = CallbackHandler()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1,
)

# Run LLM with callback handler
response = llm.invoke(
    [HumanMessage(content="Hello, Who is the best athlete in the world?")],
    config={"callbacks": [langfuse_handler]}
)

print(response.content)

# Flush the tracing events
langfuse_handler._langfuse_client.flush()