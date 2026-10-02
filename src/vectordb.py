import os
import chromadb
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from utils import get_config_value
import hashlib
import json

class VectorDB:
    """
    A simple vector database wrapper using Chroma with HuggingFace embeddings.
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

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db",
            client_settings=chromadb.config.Settings(allow_reset=True)
        )

        print(f"Vector database initialized with collection name: {self.collection_name}")

    def chunk_text(self, docs: List[Dict[str, Any]], chunk_size: int = 500) -> List[Document]:
        """
        Split text documents into chunks using RecursiveCharacterTextSplitter.

        Args:
            docs: list of dicts
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of Documents
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

        for chunk in chunked_docs:
            content_hash = hashlib.md5(
                chunk.page_content.encode("utf-8")
            ).hexdigest()
             
            #chunk.metadata["chunk_id"] = id
            chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}_page_{chunk.metadata['page']}_chunk_{content_hash}"
           
        return chunked_docs    

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """

        print(f"Chunking documents...")
    
        chunked_docs = self.chunk_text(documents)
        
        # get the chunked ids here and use them as unique ids
        ids = [chunk.metadata["chunk_id"] for chunk in chunked_docs]
        print(f"ids: \n{json.dumps(ids, indent=2)}" )

        # Get the actual text content
        texts = [chunk.page_content for chunk in chunked_docs]

        # Get the metadata
        metadatas = [chunk.metadata for chunk in chunked_docs]

        print(f"Embedding documents...")

        #embed docs
        embeddings = self.embedding_model.embed_documents(texts)

        #we use upsert to avoid duplicates when we process again the same document
        self.vector_store._collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Documents added to vector database, total documents in vector DB:  {self.vector_store._collection.count()}")

    def search(self, query: str, n_results: int = None, n_threshold: float = None) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return
            n_threshold: The distance threshold, we use cosine distance here

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
    
        if n_results is None:
            n_results = get_config_value("n_results")

        if n_threshold is None:
            n_threshold = get_config_value("n_threshold")      

        results = self.vector_store.similarity_search_with_score(query, n_results) #no need to embed query here

        documents = []
        metadatas = []
        distances = []
        ids = []
        
        for doc, score in results:
            print(f"doc id {doc.id}, threshold: {score}")
            if score <= n_threshold:
                print("score is less than or equal threshold, included")
                documents.append(doc.page_content)
                metadatas.append(doc.metadata)
                distances.append(score)
                ids.append(doc.id)
            else:
                print("score is greater than threshold, excluded")
      
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
    vector.reset_vector_store()

    try:

        #  1. Get total number of stored documents
        doc_count = vector.vector_store._collection.count()
        print(f"Total documents in collection: {doc_count}")

        # 2. Inspect the raw stored records (first 3 chunks)
        if doc_count > 0:
            raw_data = vector.vector_store._collection.get(limit=20)
            print("Sample stored documents:", raw_data["documents"])
            print("Sample metadatas:", raw_data["metadatas"])

    except Exception as e:
            print(f"Error: {str(e)}")
