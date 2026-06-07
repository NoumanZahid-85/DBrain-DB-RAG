from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2
)
parser = StrOutputParser()

# Step 1: Parallel generation of two different descriptions
parallel_gen = RunnableParallel(
    short=PromptTemplate(
        template="Describe {product} in 5 words",
        input_variables=["product"]
        ) | llm | parser,
    long=PromptTemplate(
        template="Describe {product} in 20 words",
        input_variables=["product"]
        ) | llm | parser
)

# Step 2: Decide which description to use based on length (branch)
def pick_longer(x):
    return len(x["long"]) > len(x["short"])  

branch = RunnableBranch(
    (lambda x: pick_longer(x), RunnableLambda(lambda x: x["long"])),
    RunnableLambda(lambda x: x["short"])
)

# Step 3: Combine original input with selected description
final_prompt = PromptTemplate(
    template="Write a catchy slogan for {product} based on this description: {description}",
    input_variables=["product", "description"]
)

chain = (
    RunnablePassthrough.assign(
        descriptions=parallel_gen
    )
    .assign(
        description=branch
    )
    | final_prompt
    | llm
    | parser
)

result = chain.invoke(
    {"product": "smartwatch"}
)

print("Slogan:", result)