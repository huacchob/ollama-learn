# main.py

import logging
import os
from typing import Any

import ollama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./data/BOI.pdf"
MODEL_NAME = "CodeLlama:7b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"


def ingest_pdf(doc_path: str) -> list[Document] | None:
    """Load PDF documents."""
    if os.path.exists(path=doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data: list[Document] = loader.load()
        logging.info(msg="PDF loaded successfully.")
        return data
    else:
        logging.error(msg=f"PDF file not found at path: {doc_path}")
        return None


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks."""
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
    )
    chunks: list[Document] = text_splitter.split_documents(documents=documents)
    logging.info(msg="Documents split into chunks.")
    return chunks


def create_vector_db(chunks: list[Document]) -> Chroma:
    """Create a vector database from document chunks."""
    # Pull the embedding model if not already available
    ollama.pull(model=EMBEDDING_MODEL)

    vector_db: Chroma = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info(msg="Vector database created.")
    return vector_db


def create_retriever(
    vector_db: Chroma,
    llm: ChatOllama,
) -> MultiQueryRetriever:
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever: MultiQueryRetriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(), llm=llm, prompt=QUERY_PROMPT
    )
    logging.info(msg="Retriever created.")
    return retriever


def create_chain(
    retriever: MultiQueryRetriever,
    llm: ChatOllama,
) -> RunnableSequence[dict[str, Any], str]:
    """Create the chain"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
        template=template,
    )

    chain: RunnableSequence[dict[str, Any], str] = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info(msg="Chain created successfully.")
    return chain


def main() -> None:
    # Load and process the PDF document
    data: list[Document] | None = ingest_pdf(doc_path=DOC_PATH)
    if data is None:
        return

    # Split the documents into chunks
    chunks: list[Document] = split_documents(documents=data)

    # Create the vector database
    vector_db: Chroma = create_vector_db(chunks=chunks)

    # Initialize the language model
    llm: ChatOllama = ChatOllama(model=MODEL_NAME)

    # Create the retriever
    retriever: MultiQueryRetriever = create_retriever(
        vector_db=vector_db,
        llm=llm,
    )

    # Create the chain with preserved syntax
    chain: RunnableSequence[dict[str, Any], str] = create_chain(
        retriever=retriever,
        llm=llm,
    )

    # Example query
    question: str = "How to report BOI?"

    # Get the response
    res: str = chain.invoke(input=question)
    print("Response:")
    print(res)


if __name__ == "__main__":
    main()
