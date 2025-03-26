import datetime
import os
from glob import glob
from pathlib import Path
from typing import Iterator, Union

from dotenv import load_dotenv
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import Runnable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import ChatOllama

from .utility import find_root_directory

# Update this with the model you would like to use
model: str = "CodeLlama:7b"
embeding_model: str = "nomic-embed-text"
text_to_speech_model: str = "eleven_turbo_v2"

# Directories and files
root_dir: Path = find_root_directory(file=__file__)
pdf_directory: Path = root_dir.joinpath("data/*.pdf")
vector_db_path: Path = root_dir.joinpath("db/vector_db")
dot_env_path: Path = root_dir.joinpath("creds.env")


class PDFToSpeech:
    def __init__(self) -> None:
        self.all_pages: list[Document]
        self.text_chunks: list[str]
        self.docs: list[Document]
        self.vector_db: Chroma
        self.llm: ChatOllama
        self.response: str

    def load_pdf_files(self) -> None:
        """Load all PDF files from a directory."""
        # Load the PDF files
        pdf_files: list[str] = glob(pathname=str(object=pdf_directory))
        # This list will contain all pages from the PDF
        self.all_pages: list[Document] = []

        for pdf_file in pdf_files:
            # Load the PDF file using pdfplumber
            loader: PDFPlumberLoader = PDFPlumberLoader(file_path=pdf_file)
            # Split the PDF file into pages
            pages: list[Document] = loader.load_and_split()

            self.all_pages.extend(pages)

    def split_and_chunk(self) -> None:
        # Split and chunk
        # Create an object to split text with
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
        )

        # List will contain all text chunks from the passed pdf pages
        self.text_chunks: list[str] = []
        for page in self.all_pages:
            chunks: list[str] = text_splitter.split_text(text=page.page_content)
            self.text_chunks.extend(chunks)

    # === Create Metadata for Text Chunks ===
    def add_metadata(
        self,
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

    def apply_metadata_to_documents(self) -> None:
        metadata_text_chunks: list[dict[str, Union[str, dict[str, str]]]] = (
            self.add_metadata(
                chunks=self.text_chunks,
                doc_title="BOI US FinCEN",
            )
        )
        self.docs: list[Document] = [
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in metadata_text_chunks
        ]

    def create_fast_embedding(self) -> None:
        fastembedding: FastEmbedEmbeddings = FastEmbedEmbeddings()

        # Create a chroma vector store from the list of Documents
        # Data will persist for better performance
        self.vector_db: Chroma = Chroma.from_documents(
            documents=self.docs,
            embedding=fastembedding,
            persist_directory=str(object=vector_db_path),
            collection_name="docs-local-rag",
        )

    def use_vector_db_as_retriever(self) -> VectorStoreRetriever:
        vector_retriever: VectorStoreRetriever = self.vector_db.as_retriever()
        return vector_retriever

    def create_ollama_chat(self) -> ChatOllama:
        self.llm: ChatOllama = ChatOllama(model=model)

    def create_rag(self) -> MultiQueryRetriever:
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
        # Create a multi-query retriever from the vector retriever
        # and a prompt template
        # Takes a single user query
        # uses an LLM to generate multiple variations of that query
        # performs multiple vector searches
        # combines the results and returns the most relevant documents.
        vector_retriever: VectorStoreRetriever = self.use_vector_db_as_retriever()
        retriever: MultiQueryRetriever = MultiQueryRetriever.from_llm(
            retriever=vector_retriever, llm=self.llm, prompt=QUERY_PROMPT
        )
        return retriever

    def generate_text_response(self) -> None:
        template: str = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """

        # Creates a role-based prompt
        # Internally, wraps the prompt with the "human" role
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
            template=template,
        )
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
            {"context": self.create_rag(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | output_parser
        )
        questions: str = """
        by when should I file if my business was established in 2013?"""

        # Actually calling the chain here
        # Passing it the question
        self.response: str = chain.invoke(input=questions)

    def generate_voice_response(self) -> None:
        load_dotenv(dotenv_path=str(object=dot_env_path))

        # you may need to intall the mpv package
        # brew install mpv
        # Add ELEVENLABS_API_KEY env var with your elevelabs api key
        # copy creds.env.example to creds.env and add your api key
        api_key: str | None = os.getenv(key="ELEVENLABS_API_KEY")

        # Initiate the ElevenLabs client
        if api_key is None:
            return
        client: ElevenLabs = ElevenLabs(api_key=api_key)

        # Generate the audio stream
        audio_stream: Iterator[bytes] = client.generate(
            text=self.response,
            model=text_to_speech_model,
            stream=True,
        )

        # play the audio stream
        stream(audio_stream=audio_stream)

    def main(self) -> None:
        self.create_ollama_chat()
        self.load_pdf_files()
        self.split_and_chunk()
        self.apply_metadata_to_documents()
        self.create_fast_embedding()
        self.create_rag()
        self.generate_text_response()
        self.generate_voice_response()


pdf_to_speech = PDFToSpeech()
pdf_to_speech.main()
