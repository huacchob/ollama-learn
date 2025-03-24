# app.py
# To run this use the command `streamlit run pdf-rag-streamlit.py`

import logging
import os
from typing import Any, List

import ollama
import streamlit as st
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
DOC_PATH: str = "./data/BOI.pdf"
MODEL_NAME: str = "CodeLlama:7b"
EMBEDDING_MODEL: str = "nomic-embed-text"
VECTOR_STORE_NAME: str = "simple-rag"
PERSIST_DIRECTORY: str = "./chroma_db"


def ingest_pdf(doc_path: str) -> list[Document] | None:
    """Load PDF documents."""
    if os.path.exists(path=doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data: list[Document] = loader.load()
        logging.info(msg="PDF loaded successfully.")
        return data
    else:
        logging.error(msg=f"PDF file not found at path: {doc_path}")
        st.error(body="PDF file not found.")
        return None


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks."""
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
    )
    chunks: List[Document] = text_splitter.split_documents(documents=documents)
    logging.info(msg="Documents split into chunks.")
    return chunks


@st.cache_resource
def load_vector_db() -> None | Chroma:
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    ollama.pull(model=EMBEDDING_MODEL)

    embedding: OllamaEmbeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(path=PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info(msg="Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data: list[Document] | None = ingest_pdf(doc_path=DOC_PATH)
        if data is None:
            return None

        # Split the documents into chunks
        chunks: list[Document] = split_documents(documents=data)

        vector_db: Chroma = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info(msg="Vector database created and persisted.")
    return vector_db


def create_retriever(
    vector_db: Chroma,
    llm: ChatOllama,
) -> MultiQueryRetriever:
    """Create a multi-query retriever."""
    QUERY_PROMPT: PromptTemplate = PromptTemplate(
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
    """Create the chain with preserved syntax."""
    # RAG prompt
    template: str = """Answer the question based ONLY on the following context:
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

    logging.info(msg="Chain created with preserved syntax.")
    return chain


def main() -> None:
    st.title(body="Document Assistant")

    # User input
    user_input: str = st.text_input(label="Enter your question:", value="")

    if user_input:
        with st.spinner(text="Generating response..."):
            try:
                # Initialize the language model
                llm: ChatOllama = ChatOllama(model=MODEL_NAME)

                # Load the vector database
                vector_db: None | Chroma = load_vector_db()
                if vector_db is None:
                    st.error(body="Failed to load or create the vector database.")
                    return

                # Create the retriever
                retriever: MultiQueryRetriever = create_retriever(
                    vector_db=vector_db,
                    llm=llm,
                )

                # Create the chain
                chain: RunnableSequence[dict[str, Any], str] = create_chain(
                    retriever=retriever,
                    llm=llm,
                )

                # Get the response
                response: str = chain.invoke(input=user_input)

                st.markdown(body="**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(body=f"An error occurred: {str(object=e)}")
    else:
        st.info(body="Please enter a question to get started.")


if __name__ == "__main__":
    main()
