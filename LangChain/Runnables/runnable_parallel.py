from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    temperature=0.3
)
parser = StrOutputParser()

# Three different analyses on the same movie title
summary_chain = PromptTemplate(
    template="Summarize the plot of the movie '{movie}' in one sentence",
    input_variables=["movie"]
)

rating_chain = PromptTemplate(
    template="Predict the IMDb rating out of 10 for '{movie}' (just give a number)",
    input_variables=["movie"]
)

genre_chain = PromptTemplate(
    template="List the top 3 genres for the movie '{movie}' as comma-separated",
    input_variables=["movie"]
)

parallel_chain = RunnableParallel({
    "summary": summary_chain | llm | parser,
    "rating": rating_chain | llm | parser,
    "genres": genre_chain | llm | parser
})
# Here the same input is going to the three chains in parallel and independently but they produce different results/outputs which then combines in the form of a dictionary
result = parallel_chain.invoke(
    {"movie": "Inception"},
)

print("Summary:", result["summary"])
print("Rating:", result["rating"])
print("Genres:", result["genres"])