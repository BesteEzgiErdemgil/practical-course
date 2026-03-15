import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (try .env if available)
load_dotenv()

def get_openai_client():
    """
    Returns an initialized OpenAI client.
    Tries to get the key from OS environment variables first, then Streamlit secrets.
    """
    # Try .env / OS environment first (most common for local dev)
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Fallback to Streamlit secrets (for deployed apps)
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass  # No secrets file, that's OK
    
    if not api_key:
        return None
    
    return OpenAI(api_key=api_key)

def get_chat_response(messages, model="gpt-4o-mini", temperature=0.7):
    """
    Sends a list of messages to OpenAI and returns the content of the response.
    
    Args:
        messages (list): List of message dicts [{"role": "user", "content": "..."}]
        model (str): Model to use.
        temperature (float): Creativity factor.
    
    Returns:
        str or None: The response text or None if error/no key.
    """
    client = get_openai_client()
    if not client:
        return "⚠️ OpenAI API Key not found. Please set OPENAI_API_KEY in .env or .streamlit/secrets.toml."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ API Connection Error: {str(e)}"

def get_system_prompt_for_student(student_data_summary, risk_score):
    """
    Generates a system prompt for the chatbot based on a specific student's context.
    """
    return f"""
    You are an expert AI Assistant for a Student Success Dashboard used by university counselors.
    
    **Current Student Context:**
    - Risk Probability: {risk_score:.1%}
    - Key Attributes: {student_data_summary}
    
    **Your Role:**
    - Help the counselor understand WHY this student is at risk.
    - Suggest evidence-based interventions suitable for higher education.
    - Be professional, empathetic, and concise.
    - Do NOT make up data not provided in the context. If you don't know, say so.
    """
