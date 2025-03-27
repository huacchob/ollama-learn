"""Script generating response from PDF files and generating voice."""

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
from langchain.text_splitter import RecursiveCharacterTextSplitter as RCSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import ChatOllama

from .utility import find_root_directory


class PDFProcessor:
    """PDF Processor class."""

    def __init__(self) -> None:
        """Initialize."""
        self.all_pages: list[Document]
        self.text_chunks: list[str]

    def load_pdf_files(self, directory: Path) -> None:
        """Load PDF files from a directory.

        Args:
            directory (Path): The directory to load PDF files from.
        """
        # Load the PDF files
        pdf_files: list[str] = glob(pathname=str(object=directory))
        # This list will contain all pages from the PDF
        self.all_pages: list[Document] = []

        for pdf_file in pdf_files:
            # Load the PDF file using pdfplumber
            loader: PDFPlumberLoader = PDFPlumberLoader(file_path=pdf_file)
            # Split the PDF file into pages
            pages: list[Document] = loader.load_and_split()

            self.all_pages.extend(pages)

    def split_and_chunk(self) -> None:
        """Split and chunk PDF pages into text chunks."""
        # Split and chunk
        # Create an object to split text with
        text_splitter: RCSplitter = RCSplitter(
            chunk_size=1200,
            chunk_overlap=300,
        )

        # List will contain all text chunks from the passed pdf pages
        self.text_chunks: list[str] = []
        for page in self.all_pages:
            chunks: list[str] = text_splitter.split_text(
                text=page.page_content,
            )
            self.text_chunks.extend(chunks)

    def add_metadata(
        self,
        chunks: list[str],
        doc_title: str,
        author: str,
    ) -> list[dict[str, Union[str, dict[str, str]]]]:
        """Add metadata to text chunks.

        Args:
            chunks (list[str]): Chunks of text to add metadata to.
            doc_title (str): Title of the document.
            author (str): Author of the document.

        Returns:
            list[dict[str, Union[str, dict[str, str]]]]: List of dictionaries
                containing text chunks with metadata.
        """
        metadata_chunks: list[dict[str, Union[str, dict[str, str]]]] = []
        for chunk in chunks:
            metadata: dict[str, str] = {
                "title": doc_title,
                "author": author,  # Update based on document data
                "date": str(object=datetime.date.today()),
            }
            metadata_chunks.append({"text": chunk, "metadata": metadata})
        return metadata_chunks

    def apply_metadata_to_documents(
        self,
        doc_title: str,
        author: str,
    ) -> list[Document]:
        """Apply metadata to text chunks.

        Args:
            doc_title (str): Title for document.
            author (str): Author of document.

        Returns:
            list[Document]: List of documents with metadata.
        """
        metadata_text_chunks: list[dict[str, Union[str, dict[str, str]]]] = (
            self.add_metadata(
                chunks=self.text_chunks,
                doc_title=doc_title,
                author=author,
            )
        )
        docs: list[Document] = [
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in metadata_text_chunks
        ]
        return docs


class RAGHandler:
    """RAG Handler class."""

    def __init__(self) -> None:
        """Initialize."""
        self.llm: ChatOllama

    def create_fast_embedding(
        self,
        docs: list[Document],
        directory: Path,
    ) -> Chroma:
        """Create a Chroma vector store from a list of Documents.

        Args:
            docs (list[Document]): List of Documents to create a
                Chroma vector store from.
            directory (Path): The directory to store the vector store in.

        Returns:
            Chroma: Chroma vector store.
        """
        fastembedding: FastEmbedEmbeddings = FastEmbedEmbeddings()

        # Create a chroma vector store from the list of Documents
        # Data will persist for better performance
        vector_db: Chroma = Chroma.from_documents(
            documents=docs,
            embedding=fastembedding,
            persist_directory=str(object=directory),
            collection_name="docs-local-rag",
        )
        return vector_db

    def use_vector_db_as_retriever(
        self,
        vector_db: Chroma,
    ) -> VectorStoreRetriever:
        """Use a Chroma vector store as a retriever.

        Args:
            vector_db (Chroma): Chroma vector store.

        Returns:
            VectorStoreRetriever: Vector store retriever.
        """
        return vector_db.as_retriever()

    def create_ollama_chat(self, model: str) -> None:
        """Create an Ollama chat model.

        Args:
            model (str): Name of the model to use.
        """
        self.llm: ChatOllama = ChatOllama(model=model)

    def create_rag(
        self,
        vector_db: Chroma,
    ) -> MultiQueryRetriever:
        """Create a MultiQueryRetriever from a Chroma vector store.

        Args:
            vector_db (Chroma): Chroma vector store.

        Returns:
            MultiQueryRetriever: Multi-query retriever.
        """
        query_prompt: PromptTemplate = PromptTemplate(
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
        vector_retriever: VectorStoreRetriever = self.use_vector_db_as_retriever(
            vector_db=vector_db,
        )
        retriever: MultiQueryRetriever = MultiQueryRetriever.from_llm(
            retriever=vector_retriever, llm=self.llm, prompt=query_prompt
        )
        return retriever

    def generate_text_response(
        self,
        vector_db: Chroma,
    ) -> str:
        """Generate a text response from a Chroma vector store.

        Args:
            vector_db (Chroma): Chroma vector store.

        Returns:
            str: Generated text response.
        """
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
            {
                "context": self.create_rag(vector_db=vector_db),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | output_parser
        )
        questions: str = """
        by when should I file if my business was established in 2013?"""

        # Actually calling the chain here
        # Passing it the question
        return chain.invoke(input=questions)


class GenerateVoice:
    """Generate Voice using ElevenLabs."""

    def __init__(self) -> None:
        """Initialize."""
        self.client: ElevenLabs | None

    def create_client(self, env_path: Path) -> None:
        """Create an ElevenLabs client.

        Returns:
            ElevenLabs | None: ElevenLabs client.
        """
        load_dotenv(dotenv_path=str(object=env_path))
        api_key: str | None = os.getenv(key="ELEVENLABS_API_KEY")
        if api_key is None:
            self.client: ElevenLabs | None = None
            return
        self.client: ElevenLabs | None = ElevenLabs(api_key=api_key)

    def generate_voice(self, text: str, model: str) -> None:
        """Generate voice from text.

        Args:
            text (str): Text to generate voice from.
        """
        # Generate the audio stream
        if not self.client:
            return
        audio_stream: Iterator[bytes] = self.client.generate(
            text=text,
            model=model,
            stream=True,
        )

        # play the audio stream
        stream(audio_stream=audio_stream)


# Update this with the model you would like to use
ollama_model: str = "CodeLlama:7b"
text_to_speech_model: str = "eleven_turbo_v2"

# Directories and files
root_dir: Path = find_root_directory(file=__file__)
pdf_directory: Path = root_dir.joinpath("data/*.pdf")
vector_db_path: Path = root_dir.joinpath("db/vector_db")
dot_env_path: Path = root_dir.joinpath("creds.env")


def main() -> None:
    """Run main function."""
    pdf_processor: PDFProcessor = PDFProcessor()
    pdf_processor.load_pdf_files(directory=pdf_directory)
    pdf_processor.split_and_chunk()
    docs: list[Document] = pdf_processor.apply_metadata_to_documents(
        doc_title="BOI US FinCEN",
        author="US Business Bureau",
    )
    print("Finished processing PDF files.")
    rag_handler: RAGHandler = RAGHandler()
    rag_handler.create_ollama_chat(model=ollama_model)
    vector_db: Chroma = rag_handler.create_fast_embedding(
        docs=docs,
        directory=vector_db_path,
    )
    text: str = rag_handler.generate_text_response(vector_db=vector_db)
    print("Generated text response:", text)
    elevenlabs = GenerateVoice()
    elevenlabs.create_client(env_path=dot_env_path)
    elevenlabs.generate_voice(text=text, model=text_to_speech_model)


if __name__ == "__main__":
    main()
