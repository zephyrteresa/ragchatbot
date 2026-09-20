import os
from typing import List, Any, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from langchain_core.documents import Document
import pymupdf
import base64
from langchain_core.messages import HumanMessage
from groq import Groq
from utils import load_yaml, build_prompt, get_logger
import time


# Load environment variables
load_dotenv()


def load_documents() -> List[str]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """

    results = []
    # TODO: Implement document loading
    # HINT: Read the documents from the data directory
    # HINT: Return a list of documents
    # HINT: Your implementation depends on the type of documents you are using (.txt, .pdf, etc.)

    # Your implementation here
    data_dir = Path("data").resolve()
    print(f"Loading documents from: {data_dir}")

    if data_dir.exists(): 
        print(f"Data directory exists: {data_dir}")

        assistant = RAGAssistant()
        llm = assistant._initialize_llm()

        for file_path in data_dir.glob('*.pdf'):
            print(f"file path {file_path}")
            pdf = pymupdf.open(file_path)

            #extract first the texts
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text")
            
                doc = {
                        "content": text,
                        "metadata": {"type": "text", "page": page_num, "source":Path(file_path).name, "topic": "domestic animals"}
                    }
                
                results.append(doc)

                #extract the images in a page and ask the model to give the description for each image
                for img_index, img in enumerate(page.get_images(), start=1):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_ext = base_image["ext"]
                    image_bytes = base_image["image"]
                    b64_image = base64.b64encode(image_bytes).decode("utf-8")

                    image_summary_prompt = [
                        HumanMessage(
                        content = [
                            {'type':'text', "text": "Describe the image"},
                            {'type':'image_url', "image_url": {"url": f"data:image/{image_ext};base64,{b64_image}"}},
                        ])
                    ]
                                        
                    image_summary_response = llm.invoke(image_summary_prompt, max_tokens=500)
                   
                    doc = {
                        "content": f"Image Summary: {image_summary_response.content}",
                        "metadata": {"type": "image", "page": page_num, "b64_image": b64_image, "source":Path(file_path).name, "topic": "domestic animals"}
                    }
                    
                    results.append(doc)    
                             
                    # Pause for 2 seconds between image requests to respect rate limits
                    time.sleep(2)           

    else:
        print(f"Data directory does not exist: {data_dir}")            

    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template
        # TODO: Implement your RAG prompt template
        # HINT: Use ChatPromptTemplate.from_template() with a template string
        # HINT: Your template should include placeholders for {context} and {question}
        # HINT: Design your prompt to effectively use retrieved context to answer questions

        #use prompt config here
        config_dir = Path("config").resolve()
        prompt_config_file = os.path.join(config_dir, "prompt_config.yaml")
        prompt_config = load_yaml(prompt_config_file) 
        prompt = build_prompt(prompt_config, 'rag_prompt_config')

        print (f"prompt from config: {prompt}") 
        self.prompt_template = ChatPromptTemplate.from_template(f"{prompt} \n\n Answer the following question based on the provided context:\n\nContext: {{context}}\n\nQuestion: {{question}}")  

        # Create the chain
        if self.prompt_template:
            self.chain = self.prompt_template | self.llm | StrOutputParser()

        #set the logger
        self.logger = get_logger("RAG Assistant")   

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):

            # client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            # # Fetch all available models from Groq
            # models = client.models.list()

            # # Print active model IDs
            # for model in models.data:
            #     print(model.id)

            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), max_tokens=800, model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            LLM answer
        """
        
        # TODO: Implement the RAG query pipeline
        # HINT: Use self.vector_db.search() to retrieve relevant context chunks
        # HINT: Combine the retrieved document chunks into a single context string
        # HINT: Use self.chain.invoke() with context and question to generate the response
        # HINT: Return a string answer from the LLM

        # get context by querying the vector store
        rag_context = self.vector_db.search(input, n_results)
        context_str = "\n\n".join(rag_context['documents'])

        # for context in rag_context['documents']:
        #     context_str = context_str + " \n" + context

        prompt_inputs = {
            "context":context_str,
            "question":input
        }

        formatted_prompt = self.prompt_template.format(**prompt_inputs)
        self.logger.info(f"Prompt sent to LLM : {formatted_prompt}")
             
        response = self.chain.invoke(prompt_inputs)

        self.logger.info(f"response : {response}")
        self.logger.info("=====================================================")

        return response

def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.invoke(question)
                print(result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
    #load_documents()  # Call the load_documents function to test it
