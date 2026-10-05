import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

def get_llm(model_name: str = None):
    """
    Initialize the LLM by checking for available API keys.
    Tries OpenAI, Groq, and Google Gemini in that order.
    """
    # Check for OpenAI API key
    if os.getenv("OPENAI_API_KEY"):
        model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"Using OpenAI model: {model_name}")
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
        )

    elif os.getenv("GROQ_API_KEY"):

        model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        print(f"Using Groq model: {model_name}")
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"), max_tokens=800, model=model_name, temperature=0.0
        )

    elif os.getenv("GOOGLE_API_KEY"):
        model_name = model_name or os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
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
