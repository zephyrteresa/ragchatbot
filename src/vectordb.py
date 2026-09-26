import os
import chromadb
from typing import List, Dict, Any, Optional
# from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from utils import load_yaml
from pathlib import Path

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
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        # Get or create collection
        # self.collection = self.client.get_or_create_collection(
        #     name=self.collection_name,
        #     metadata={"description": "RAG document collection"},
        # )

        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db",
            client_settings=chromadb.config.Settings(allow_reset=True)
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, docs: List[Dict[str, Any]], chunk_size: int = 500) -> List[Document]:
        """
        Split text documents into chunks using RecursiveCharacterTextSplitter.

        Args:
            docs: list of dicts
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """

        if not docs:
            return []

        #convert dictionary objects to Document objects
        doc_objects = [
            Document(
                    page_content=d["content"],
                    metadata=d["metadata"]
                ) 
            for d in docs   
        ]

        splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=50
                )
        
        # Split Document objects, handle all documents in one call
        chunked_docs = splitter.split_documents(doc_objects)

        for id, chunk in enumerate(chunked_docs, start=1):
            chunk.metadata["chunk_id"] = id
           
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
        #existing_count = len(self.vector_store.get()["ids"]) 
        existing_count = self.vector_store._collection.count()
        ids = list(range(existing_count, existing_count + len(chunked_docs)))
        ids = [f"document_{id}" for id in ids]
        
        self.vector_store.add_documents(
            documents=chunked_docs,
            ids=ids
        )

        print("Documents added to vector database")

    def search(self, query: str, n_results: int = None, n_threshold: float = None) -> Dict[str, Any]:
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
        default_config = load_yaml(Path("config") / "config.yaml")
        if n_results is None:
            n_results = default_config["n_results"]

        if n_threshold is None:
            n_threshold = default_config["n_threshold"]    

        print(f"n_results: {n_results} , n_threshold: {n_threshold}")    

        results = self.vector_store.similarity_search_with_score(query, n_results)

        documents = []
        metadatas = []
        distances = []
        ids = []

        for doc, score in results:

            print(f"doc id {doc.id}, threshold: {score}")
            if score <= n_threshold:
                print("score is less than threshold, included")
                documents.append(doc.page_content)
                metadatas.append(doc.metadata)
                distances.append(score)
                ids.append(doc.id)
            else:
                print("score is greater than or equal threshold, exluded")

        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "ids": ids,
        }

    def reset_vector_store(self, collection_name: Optional[str] = None):
        print("inside reset vector store")
       
        try:
            count = 0
            if collection_name is not None:
                print("inside if")
                self.vector_store._client.delete_collection(collection_name)
                count = self.vector_store._client.get_collection(collection_name).count()
            else:
                print("inside else")
                self.vector_store._client.reset()
                count = self.vector_store._collection.count()
            
            print(f"{count} docs remaining")    

        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    vector = VectorDB()
    #vector.reset_vector_store()

     # 1. Get total number of stored documents
    doc_count = vector.vector_store._collection.count()
    print(f"Total documents in collection: {doc_count}")

    # 2. Inspect the raw stored records (first 3 chunks)
    if doc_count > 0:
        raw_data = vector.vector_store._collection.get(limit=3)
        print("Sample stored documents:", raw_data["documents"])
        print("Sample metadatas:", raw_data["metadatas"])