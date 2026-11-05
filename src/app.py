import os
from typing import List, Any, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from pathlib import Path
import pymupdf
import base64
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from utils import get_logger, get_config_value, get_prompt
import time
from llms import get_llm
from langchain_core.load import dumps
import traceback


# Load environment variables
load_dotenv()


def load_documents() -> List[Dict[str, Any]]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """

    results = []
  
    data_dir = Path("data").resolve()
    print(f"Loading documents from: {data_dir}")

    if data_dir.exists(): 
        llm = get_llm(os.getenv("VISION_MODEL"))

        #get all pdf docs
        all_pdfs = list(data_dir.glob("*.pdf"))
       
        if not all_pdfs:
            raise FileNotFoundError(
                f"No PDF files found in directory: {data_dir}"
            )

        for file_path in all_pdfs:
            pdf = pymupdf.open(file_path)
           
            seen_xrefs = set()

            #extract first the texts
            for page_num, page in enumerate(pdf, start=1):
                print(f"Processing page {page_num}...")
                text = page.get_text("text")

                doc = {
                        "content": text,
                        "metadata": {"type": "text", "page": page_num, "source":Path(file_path).name}
                    }
                
                results.append(doc)

                #extract the images in a page and ask the model to give the description for each image
                for img_index, img in enumerate(page.get_images(), start=1):
                    try:
                        xref = img[0]
                        width = img[2]
                        height = img[3]
    
                        # Skip processing this image because it was already summarized on an earlier page
                        if xref in seen_xrefs or width < 150 or height < 150:
                            # Ignore tiny graphics, icons, or decorative elements
                            continue
                        
                        print(f"Loading image of page {page_num}...")
                            
                        seen_xrefs.add(xref)
    
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
                            "metadata": {"type": "image", "page": page_num, "xref": xref, "source":Path(file_path).name}
                        }
                        
                        results.append(doc)    
                                    
                        # Pause for .5 second between image requests to respect rate limits
                        time.sleep(0.5)           
                    except Exception as e:
                        print(f"API Error or Timeout on page image: {e}")
                        # continue processing the next image
                        continue  
        
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
        self.llm = get_llm(os.getenv("GROQ_MODEL"))
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()
               
        self.prompt_template = ChatPromptTemplate.from_messages([
                            ("system", get_prompt("rag_prompt_config")),
                            ("human", """
        
                        Answer the user's original question based on the provided context:
        
                        <context>
                            {context}
                        </context>
                        
                        Original Question: 
                            {original_question}

                        Standalone Question: 
                            {rewritten_question}    
        
                        """)
                        ])
                
        # Create the chain
        if self.prompt_template:
            self.chain = self.prompt_template | self.llm | StrOutputParser()

        #set the logger
        self.logger = get_logger("RAG Assistant")   

        #set the conversation history
        self.conversation_history = []
        self.conversation_summary = ""

        print("RAG Assistant initialized successfully")

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)
  
    def invoke(self, input: str, rewritten_input: str = None, n_results: int = None, n_threshold: int = None) -> str:
        """
        Query the RAG assistant.

        Args:
            input: Original user's input
            rewritten_input: Rewritten input, this is used to query the vector store
            n_results: Number of relevant chunks to retrieve
            n_threshold: The distance threshold

        Returns:
            LLM answer
        """

        # Use rewritten input for vector database retrieval
        search_query = rewritten_input or input

        # get context by querying the vector store
        rag_context = self.vector_db.search(search_query, n_results, n_threshold)

        if not rag_context['documents']:
            context_str = "No relevant context found"
        else:            
            context_str = "\n\n".join(rag_context['documents'])
    
        prompt_inputs = {         
            "context":context_str,            
            "original_question":input,
            "rewritten_question":rewritten_input,
        }

        # log the prompt sent to LLM
        json_prompt = dumps(self.prompt_template.format_prompt(**prompt_inputs), pretty=True)
        self.logger.info(f"Prompt sent to LLM:\n{json_prompt}")
                
        response = self.chain.invoke(prompt_inputs)

        self.logger.info(f"Response: {response}")
        self.logger.info("\n\n ==================================================== \n\n")
    
        return response
    
    def summarize_old_messages(self) -> str:
        """
        Summarize the old messages
    
        Returns:
            The summary
        """

        max_recent_msg = get_config_value("max_recent_messages")

        if (len(self.conversation_history) <= max_recent_msg):
            self.logger.info(f"history length is <= {max_recent_msg}, no summary")
            return "Not available"
        
        old_messages = self.conversation_history[:-max_recent_msg]

        old_msg = ""
        for msg in old_messages:
            old_msg = old_msg + "\n " + msg.content

        summary_prompt = f"""
        
            {get_prompt("summarize_prompt_config")}

            Existing summary: 
            {self.conversation_summary or "Not available"}

            Messages: 
            {old_msg}
            """

        self.logger.info(f"summary prompt: {summary_prompt}")
        
        llm = get_llm(os.getenv("GROQ_MODEL"))
        response = llm.invoke(summary_prompt)
        summary = response.content 

        # update the summary
        self.conversation_summary = summary

        #the history will now contain the unsummarized messages
        self.conversation_history = self.conversation_history[-max_recent_msg:]

        return summary

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite the query into standalone, context-complete search queries to improve retrieval in vector databases
        This considers the previous conversation in the update
    
        Args:
            query: The raw input question or prompt from the user.

        Returns:
            The updated query
        """
            
        rewrite_query_prompt = f"""
        
            {get_prompt("rewrite_query_prompt_config")}

            Existing summary: 
            {self.conversation_summary or "Not available"} 

            Recent conversation history: 
            {get_buffer_string(self.conversation_history) or "Not available"}

            Query:
            {query}
            
            """

        self.logger.info(f"rewrite query prompt: {rewrite_query_prompt}")
        
        llm = get_llm(os.getenv("GROQ_MODEL"))
        response = llm.invoke(rewrite_query_prompt)
    
        return response.content 



def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        sample_docs = load_documents() 
      
        print(f"Loaded {len(sample_docs)} sample documents")
        
        assistant.add_documents(sample_docs)

        done = False
        assistant.conversation_history = []

        while not done:
            rewritten_question = ""
            question = input("Enter a question or type 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                if len(assistant.conversation_history) > 0:
                    # summarize old messages here 
                    assistant.summarize_old_messages();

                    rewritten_question = assistant.rewrite_query(question)
                    assistant.logger.info(f"rewrite query, new query is: {rewritten_question}")  
                else:
                    assistant.logger.info('do not rewrite query, no history yet')    

                assistant.conversation_history.append(HumanMessage(content=question))
                result = assistant.invoke(question, rewritten_question)
                assistant.conversation_history.append(AIMessage(content=result))
                
                print(result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        traceback.print_exc()
       

if __name__ == "__main__":
    main()
