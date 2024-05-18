from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import uvicorn
import logging

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load embeddings and vector database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
try:
    db = FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
    logger.info("Vector database loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load vector database: {e}")
    raise e

# Load LLM
try:
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens=128,
        temperature=0.5,
    )
    logger.info("LLM model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load LLM model: {e}")
    raise e

# Define custom prompt template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
qa_prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Set up RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt},
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

def clean_answer(answer):
    # Remove unnecessary characters and symbols
    cleaned_answer = re.sub(r'[^\w\s.,-]', '', answer)

    # Remove repetitive phrases by identifying repeated words or sequences
    cleaned_answer = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned_answer)

    # Remove any trailing or leading spaces
    cleaned_answer = cleaned_answer.strip()

    # Replace multiple spaces with a single space
    cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer)

    # Replace \n with newline character in markdown
    cleaned_answer = re.sub(r'\\n', '\n', cleaned_answer)

    # Check for bullet points and replace with markdown syntax
    cleaned_answer = re.sub(r'^\s*-\s+(.*)$', r'* \1', cleaned_answer, flags=re.MULTILINE)

    # Check for numbered lists and replace with markdown syntax
    cleaned_answer = re.sub(r'^\s*\d+\.\s+(.*)$', r'1. \1', cleaned_answer, flags=re.MULTILINE)

    # Check for headings and replace with markdown syntax
    cleaned_answer = re.sub(r'^\s*(#+)\s+(.*)$', r'\1 \2', cleaned_answer, flags=re.MULTILINE)

    return cleaned_answer

def format_sources(sources):
    formatted_sources = []
    for source in sources:
        metadata = source.metadata
        page = metadata.get('page', 'Unknown page')
        source_str = f"{metadata.get('source', 'Unknown source')}, page {page}"
        formatted_sources.append(source_str)
    return "\n".join(formatted_sources)

@app.post("/query", response_model=AnswerResponse)
async def query(question_request: QuestionRequest):
    try:
        question = question_request.question
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        result = qa_chain({"query": question})
        answer = result.get("result")
        sources = result.get("source_documents")

        if sources:
            formatted_sources = format_sources(sources)
            answer += "\nSources:\n" + formatted_sources
        else:
            answer += "\nNo sources found"

        # Clean up the answer
        cleaned_answer = clean_answer(answer)

        # Return cleaned_answer wrapped in a dictionary
        return {"answer": cleaned_answer}

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
