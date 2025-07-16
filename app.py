import streamlit as st
import os
import re
import requests
import pdfplumber
import docx
from transformers import MarianMTModel, MarianTokenizer

# Load API key from secrets
API_KEY = st.secrets["gemini"]["api_key"]
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# Language codes
LANG_CODES = {
    "English": None,
    "Gujarati": "gu"
}

# Translation cache
translation_models = {}

# Load translation model
def load_translation_model(src, tgt):
    key = f"{src}-{tgt}"
    if key in translation_models:
        return translation_models[key]
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translation_models[key] = (tokenizer, model)
    return tokenizer, model

# Translate text
def translate(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract text from file
def extract_text_from_file(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages[:10]])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return ""

# Clean sensitive errors
def safe_error_message(e):
    return re.sub(r'key=[a-zA-Z0-9\-_]+', 'key=****', str(e))

# Call Gemini API
def query_gemini_flash(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }
    try:
        res = requests.post(API_URL, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"❌ Gemini Error: {safe_error_message(e)}"

# Streamlit Page Config
st.set_page_config("EduBhasha AI Tutor", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    /* Sticky header */
    .sticky-header-container {
        position: fixed;
        top: 2rem;
        left: 0;
        right: 0;
        z-index: 998;
        background-color: #0e1117;
        padding: 0.8rem 1rem 1rem;
        border-bottom: 1px solid #333;
    }

    .sticky-header-content {
        max-width: 960px;
        margin: 0 auto;
        text-align: center;
    }

    .sticky-header-content h2 {
        font-size: 1.6rem;
        margin-bottom: 0.2rem;
    }

    .sticky-header-content p {
        font-size: 0.95rem;
        color: #aaa;
        margin-top: 0;
    }

    /* Main padding to prevent overlap */
    .main > div {
        padding-top: 135px !important;
    }

    @media (max-width: 788px) {
        .sticky-header-content h2 {
            font-size: 1.3rem;
        }

        .sticky-header-content p {
            font-size: 0.85rem;
        }

        .main > div {
            padding-top: 155px !important;
        }
    }

    /* Streamlit app bar */
    header.css-18ni7ap.e8zbici2 {
        z-index: 1000 !important;
    }
    </style>
""", unsafe_allow_html=True)



# Header
st.markdown("""
    <div class="sticky-header-container">
        <div class="sticky-header-content">
            <h2>
                🧠 EduAI – Multilingual Tutor
            </h2>
            <p>
                Ask educational questions in English and Gujarati. Upload PDFs for enhanced answers.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Language")
    selected_lang = st.selectbox("Choose your language", options=list(LANG_CODES.keys()))

    st.header("Upload PDF/DOCX")
    uploaded_file = st.file_uploader("PDF or DOCX only", type=["pdf", "docx"])
    use_file_only = st.checkbox("📄 Answer only using uploaded PDF/DOCX")

# Chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask your question here...")

if prompt:
    lang_code = LANG_CODES[selected_lang]
    if lang_code and lang_code != "gu":
        in_tok, in_mod = load_translation_model(lang_code, "en")
        out_tok, out_mod = load_translation_model("en", lang_code)
        prompt_en = translate(prompt, in_tok, in_mod)
    else:
        prompt_en = prompt

    context = extract_text_from_file(uploaded_file) if uploaded_file else ""

    if use_file_only and context:
        system_prompt = f"""You are a helpful educational AI tutor. Strictly answer based only on the following PDF/DOCX content. Respond in simple {selected_lang}.

Context:
{context}

Question:
{prompt_en}

🧠 Provide explanation and examples only if relevant to context.
"""
    else:
        system_prompt = f"""You are an educational AI tutor. Answer the following question clearly and in simple {selected_lang}. Use general knowledge.

Question:
{prompt_en}

🧠 Provide explanation and examples if needed.
"""

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response_en = query_gemini_flash(system_prompt)

    if lang_code and lang_code != "gu":
        response_native = translate(response_en, out_tok, out_mod)
    elif lang_code == "gu":
        response_native = f"(English only):\n{response_en}"
    else:
        response_native = response_en

    st.chat_message("assistant").markdown(response_native)
    st.session_state.messages.append({"role": "assistant", "content": response_native})
