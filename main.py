# run 100% locally on your machine - computer

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import logging
# declare the LLM model 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model = OllamaLLM(model="llama3.2")

template = """
You are an expert in Formula 1 racing and statistical analysis. You have deep knowledge of F1 tournaments, driver performance, team strategies, 
and track characteristics. Use the provided data to answer questions accurately and perform statistical analysis when required.

Here are the relevant F1 data points:
{reviews}

Here is the question to answer: {question}

Instructions:
1. Base your answer strictly on the provided data points unless they are insufficient to answer the question.
2. If predicting the next race winner, analyze the data (e.g., past driver performance, team form, track suitability) and provide a reasoned prediction with probabilities or confidence levels where possible.
3. If the data is insufficient, clearly state so and provide a general analysis based on F1 trends, but prioritize the provided data.
4. Format your answer concisely and include relevant details from the data (e.g., driver, team, year, track).
"""

# pass in the review and question to the template
prompt = ChatPromptTemplate.from_template(template)

# invoke the model with the prompt and get the response
chain = prompt | model

while True:
    print("\n\n---------------------------------")
    question = input("As a question (or 'q' to quit): ")
    print("\n\n")
    if question.lower() == 'q':
        break
    
    #This is RAG
    # Retrieve the relevant reviews from the vector store upon the question
    reviews = retriever.invoke(question)
    logger.info(f"Retrieved {len(reviews)} documents for query: {question}")

    print("Retrieved documents:")
    if not reviews:
        print("No documents retrieved.")
    for i, doc in enumerate(reviews, 1):
        print(f"{i}. {doc.page_content} (Metadata: {doc.metadata})")

    formatted_reviews = "\n".join([f"Data point {i}: {doc.page_content} (Metadata: {doc.metadata})" for i, doc in enumerate(reviews, 1)])
    if not formatted_reviews:
        formatted_reviews = "No relevant data points retrieved."
    # Invoke the model with the reviews and question
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)