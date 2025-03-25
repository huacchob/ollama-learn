import os
from glob import glob
from pathlib import Path
from typing import Union

import ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever

from .utility import find_root_directory

current_dir: Path = find_root_directory(file=__file__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# you may need to intall the mpv package
# brew install mpv
# Update this with the model you would like to use
model = "CodeLlama:7b"

pdf_directory: Path = current_dir.joinpath("data/*.pdf")

pdf_files: list[str] = glob(pathname=str(object=pdf_directory))

# This list will contain all pages from the PDF
all_pages: list[Document] = []

for pdf_file in pdf_files:
    print(f"Processing PDF file: {pdf_file[pdf_file.rfind('/') + 1 :]}")

    # Load the PDF file using pdfplumber
    loader: PDFPlumberLoader = PDFPlumberLoader(file_path=pdf_file)
    # Split the PDF file into pages
    pages: list[Document] = loader.load_and_split()
    print(f"pages length: {len(pages)}")

    all_pages.extend(pages)

    # Extract text from the first page in the PDF file
    text: str = pages[0].page_content
    print(f"Text extracted from the PDF file '{pdf_file}':\n{text}\n")

    # Prepare the prompt for the model
    prompt: str = f"""
    You are an AI assistant that helps with summarizing PDF documents.
    
    Here is the content of the PDF file '{pdf_file}':
    
    {text}
    
    Please summarize the content of this document in a few sentences.
    """

    # Send the prompt and get the response
    try:
        response: ollama.GenerateResponse = ollama.generate(
            model=model,
            prompt=prompt,
        )
        # Grab response
        summary: str = response.get(key="response", default="")
    except Exception as e:
        print(
            f"An error occurred while summarizing the PDF file '{pdf_file}': {str(object=e)}"
        )

# Split and chunk
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create an object to split text with
text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300,
)

# List will contain all text chunks from the passed pdf pages
text_chunks: list[str] = []
for page in all_pages:
    chunks: list[str] = text_splitter.split_text(text=page.page_content)
    text_chunks.extend(chunks)

print(f"Number of text chunks: {text_chunks}")


# === Create Metadata for Text Chunks ===
# Example metadata management (customize as needed)
import datetime
import pprint


def add_metadata(
    chunks: list[str],
    doc_title: str,
) -> list[dict[str, Union[str, dict[str, str]]]]:
    """Add metadata to text chunks.

    Args:
        chunks (list[str]): Chunks of text to add metadata to.
        doc_title (str): Title of the document.

    Returns:
        list[dict[str, Union[str, dict[str, str]]]]: List of dictionaries
            containing text chunks with metadata.
    """
    metadata_chunks: list[dict[str, Union[str, dict[str, str]]]] = []
    for chunk in chunks:
        metadata: dict[str, str] = {
            "title": doc_title,
            "author": "US Business Bureau",  # Update based on document data
            "date": str(object=datetime.date.today()),
        }
        metadata_chunks.append({"text": chunk, "metadata": metadata})
    return metadata_chunks


# add metadata to text chunks
metadata_text_chunks: list[dict[str, Union[str, dict[str, str]]]] = add_metadata(
    chunks=text_chunks,
    doc_title="BOI US FinCEN",
)
pprint.pprint(object=f"metadata text chunks: {metadata_text_chunks}")


# === Create Embedding from Text Chunks ===
# Function to generate embeddings for text chunks
# def generate_embeddings(
#     text_chunks: list[str],
#     model_name: str = "nomic-embed-text",
# ) -> list[ollama.EmbedResponse]:
#     """Generate embeddings for text chunks.

#     Args:
#         text_chunks (list[str]): List of text chunks to generate embeddings for.
#         model_name (str, optional): Name of the model to use for generating embeddings.
#             Defaults to "nomic-embed-text".

#     Returns:
#         list[ollama.EmbedResponse]: List of embedding responses.
#     """
#     ollama.pull(model=model_name)
#     embeddings: list[ollama.EmbedResponse] = []
#     for chunk in text_chunks:
#         # Generate the embedding for each chunk
#         embedding: ollama.EmbedResponse = ollama.embed(model=model_name, input=chunk)
#         embeddings.append(embedding)
#     return embeddings


# # example embeddings
# texts: list[str] = [str(object=chunk["text"]) for chunk in metadata_text_chunks]
# embeddings: list[ollama.EmbedResponse] = generate_embeddings(
#     text_chunks=texts,
# )
# print(f"Embeddings: {embeddings}")


# === Add Embeddings to Vector Database Chromadb ===

# Wrap texts with their respective metadata into Document objects
docs: list[Document] = [
    Document(page_content=chunk["text"], metadata=chunk["metadata"])
    for chunk in metadata_text_chunks
]

# == Use fastEmbeddings model from Ollama ==
# for lightweight and fast embedding generation
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

fastembedding: FastEmbedEmbeddings = FastEmbedEmbeddings()
# Also for performance improvement, persist the vector database
vector_db_path: str = "./db/vector_db"

# Create a chroma vector store from the list of Documents
# Data will persist
vector_db: Chroma = Chroma.from_documents(
    documents=docs,
    embedding=fastembedding,
    persist_directory=vector_db_path,
    collection_name="docs-local-rag",
)


# Implement a Query Processing Muliti-query Retriever
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# Langchain integration to chat with any Ollama model
llm: ChatOllama = ChatOllama(model=model)

# Create a prompt template to use by the model
QUERY_PROMPT: PromptTemplate = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# A retriever in this case is used to
# retrieve data from a vector database
vector_retriever: VectorStoreRetriever = vector_db.as_retriever()

# Create a multi-query retriever from the vector retriever
# and a prompt template
# Takes a single user query
# uses an LLM to generate multiple variations of that query
# performs multiple vector searches
# combines the results and returns the most relevant documents.
retriever: MultiQueryRetriever = MultiQueryRetriever.from_llm(
    retriever=vector_retriever, llm=llm, prompt=QUERY_PROMPT
)

# RAG prompt
template: str = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

# Creates a role-based prompt
# Internally, wraps the prompt with the "human" role
prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
    template=template,
)

from langchain.schema.runnable import Runnable

# Output parser
# Parses LLMResult into the top likely string
# This means that the model will only return the most probable answer
output_parser: StrOutputParser = StrOutputParser()

# A chain is meant to run callables in sequence with additional inputs
# The first line is a dictionary of inputs mapped to a retriever and a question
# The second line is a prompt template
# The third line is an LLM
# The fourth line is an output parser
# The callables are chained together using a pipe
chain: Runnable[dict[str, str], str] = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

questions: str = """
by when should I file if my business was established in 2013?"""

# Actually calling the chain here
# Passing it the question
response: str = chain.invoke(input=questions)
print(response)

# # === TALK TO THE MODEL ===
# from dotenv import load_dotenv
# from elevenlabs import stream
# from elevenlabs.client import ElevenLabs

# load_dotenv()

# text_response: str = response

# # Add ELEVENLABS_API_KEY env var with your elevelabs api key
# # export ELEVEN_LABS_API_KEY="api-key"
# api_key: str | None = os.getenv(key="ELEVENLABS_API_KEY")

# # Generate the audio stream
# client: ElevenLabs = ElevenLabs(api_key=api_key)
# audio_stream: Iterator[bytes] = client.generate(
#     text=text_response,
#     model="eleven_turbo_v2",
#     stream=True,
# )
# # play(audio_stream)
# stream(audio_stream=audio_stream)
