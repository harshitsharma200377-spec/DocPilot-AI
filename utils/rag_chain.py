# BEFORE:
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# AFTER:
import os
from dotenv import load_dotenv

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def _get_api_key() -> str:
    """Return GROQ_API_KEY from Streamlit secrets (Cloud) or env (local)."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY", "")
