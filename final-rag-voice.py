import os
from typing import Iterator, Union

import ollama
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = "CodeLlama:7b"

pdf_files: list[str] = [f for f in os.listdir(path="./data") if f.endswith(".pdf")]

all_pages: list[Document] = []

for pdf_file in pdf_files:
    file_path = os.path.join("./data", pdf_file)
    print(f"Processing PDF file: {pdf_file}")

    # Load the PDF file
    loader = PDFPlumberLoader(file_path=file_path)
    pages: list[Document] = loader.load_and_split()
    print(f"pages length: {len(pages)}")

    all_pages.extend(pages)

    # Extract text from the PDF file
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
        summary: str = response.get(key="response", default="")

        # print(f"Summary of the PDF file '{pdf_file}':\n{summary}\n")
    except Exception as e:
        print(
            f"An error occurred while summarizing the PDF file '{pdf_file}': {str(object=e)}"
        )

# Split and chunk
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300,
)

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
ollama.pull(model="nomic-embed-text")


# Function to generate embeddings for text chunks
def generate_embeddings(
    text_chunks: list[str],
    model_name: str = "nomic-embed-text",
) -> list[ollama.EmbeddingsResponse]:
    embeddings: list[ollama.EmbeddingsResponse] = []
    for chunk in text_chunks:
        # Generate the embedding for each chunk
        embedding: ollama.EmbeddingsResponse = ollama.embeddings(
            model=model_name, prompt=chunk
        )
        embeddings.append(embedding)
    return embeddings


# example embeddings
texts: list[str] = [str(chunk["text"]) for chunk in metadata_text_chunks]
embeddings: list[ollama.EmbeddingsResponse] = generate_embeddings(
    text_chunks=texts,
)
print(f"Embeddings: {embeddings}")


# === Add Embeddings to Vector Database Chromadb ===

# Wrap texts with their respective metadata into Document objects
docs: list[Document] = [
    Document(page_content=chunk["text"], metadata=chunk["metadata"])
    for chunk in metadata_text_chunks
]

# == Use fastEmbeddings model from Ollama ==
# to add embeddings into the vector database
# and have a better quality of the embeddings
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

fastembedding: FastEmbedEmbeddings = FastEmbedEmbeddings()
# Also for performance improvement, persist the vector database
vector_db_path: str = "./db/vector_db"

vector_db: Chroma = Chroma.from_documents(
    documents=docs,
    embedding=fastembedding,
    persist_directory=vector_db_path,
    # embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="docs-local-rag",
)


# Implement a Query Processing Muliti-query Retriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# LLM from Ollama
llm: ChatOllama = ChatOllama(model=model)

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

# RAG prompt
template: str = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
    template=template,
)

from langchain.schema.runnable import Runnable

chain: Runnable[dict[str, str], str] = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

questions: str = """
by when should I file if my business was established in 2013?"""

print((chain.invoke(input=questions)))
response: str = chain.invoke(input=questions)

# === TALK TO THE MODEL ===
from dotenv import load_dotenv
from elevenlabs import stream
from elevenlabs.client import ElevenLabs

load_dotenv()

text_response: str = response

api_key: str | None = os.getenv(key="ELEVENLABS_API_KEY")

# Generate the audio stream
client: ElevenLabs = ElevenLabs(api_key=api_key)
audio_stream: Iterator[bytes] = client.generate(
    text=text_response,
    model="eleven_turbo_v2",
    stream=True,
)
# play(audio_stream)
stream(audio_stream=audio_stream)
