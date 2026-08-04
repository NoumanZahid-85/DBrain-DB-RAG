import os
import time
from dotenv import load_dotenv
from google.genai import Client

load_dotenv()

client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    print("Testing single batch direct embed...")
    texts = [f"This is test sentence number {i}" for i in range(100)]
    start = time.time()
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=texts
    )
    print(f"Success! Got {len(response.embeddings)} embeddings in {time.time() - start:.2f}s")
    print(f"Dimension: {len(response.embeddings[0].values)}")
except Exception as e:
    print(f"Failed: {e}")
