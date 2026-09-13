import os
import chromadb
from typing import List, Dict, Any
# from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        # self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        # self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_model = HuggingFaceEmbeddings(self.embedding_model_name)

        # Get or create collection
        # self.collection = self.client.get_or_create_collection(
        #     name=self.collection_name,
        #     metadata={"description": "RAG document collection"},
        # )

        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db"
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, docs: List[Dict[str, Any]], chunk_size: int = 500) -> List[Dict[str, Any]]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            docs: list of doc dictionaries with content and metadata keys
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # TODO: Implement text chunking logic
        # You have several options for chunking text - choose one or experiment with multiple:
        #
        # OPTION 1: Simple word-based splitting
        #   - Split text by spaces and group words into chunks of ~chunk_size characters
        #   - Keep track of current chunk length and start new chunks when needed
        #
        # OPTION 2: Use LangChain's RecursiveCharacterTextSplitter
        #   - from langchain_text_splitters import RecursiveCharacterTextSplitter
        #   - Automatically handles sentence boundaries and preserves context better
        #
        # OPTION 3: Semantic splitting (advanced)
        #   - Split by sentences using nltk or spacy
        #   - Group semantically related sentences together
        #   - Consider paragraph boundaries and document structure
        #
        # Feel free to try different approaches and see what works best!

        
        # Your implementation here

        chunks = []

        doc_objects = [
            Document(
                    page_content=d["content"],
                    metadata=d["metadata"]
                ) 
            for d in docs   
        ]

        splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=20
                )
        
        # Split Document objects
        chunked_docs = splitter.split_documents(doc_objects)

        for id, chunk in enumerate(chunked_docs, start=1):
            chunk.metadata["chunk_id"] = id
            # chunks.append({
            #     'content': chunk.page_content,                
            #     'metadata': chunk.metadata
            # })

        return chunked_docs    


    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        # TODO: Implement document ingestion logic
        # HINT: Loop through each document in the documents list
        # HINT: Extract 'content' and 'metadata' from each document dict
        # HINT: Use self.chunk_text() to split each document into chunks
        # HINT: Create unique IDs for each chunk (e.g., "doc_0_chunk_0")
        # HINT: Use self.embedding_model.encode() to create embeddings for all chunks
        # HINT: Store the embeddings, documents, metadata, and IDs in your vector database
        # HINT: Print progress messages to inform the user

        print(f"Processing {len(documents)} documents...")
        # Your implementation here

        chunked_docs = self.chunk_text(documents)

        #set the unique ids
        existing_count = len(self.vector_store.get()["ids"]) 
        ids = list(range(existing_count, existing_count + len(chunked_docs)))
        ids = [f"document_{id}" for id in ids]
        
        self.vector_store.add_documents(
            documents=chunked_docs,
            ids=ids
        )

        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # TODO: Implement similarity search logic
        # HINT: Use self.embedding_model.encode([query]) to create query embedding
        # HINT: Convert the embedding to appropriate format for your vector database
        # HINT: Use your vector database's search/query method with the query embedding and n_results
        # HINT: Return a dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        # HINT: Handle the case where results might be empty

        # Your implementation here 

        results = self.vector_store.similarity_search_with_score(query, n_results)

        documents = []
        metadatas = []
        distances = []
        ids = []

        for doc, score in results:
            documents.append(doc.page_content)
            metadatas.append(doc.metadata)
            distances.append(score)
            ids.append(doc.id)

        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "ids": ids,
        }
