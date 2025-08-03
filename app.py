import streamlit as st
from url_predictor import TrulyLazyPredictor
import config

st.set_page_config(page_title="SkimLit Classifier", layout="centered")

# --- Custom Styling: Black and Pink Theme ---
dark_mode = st.toggle("🌙 Dark Mode", value=True)

# Inject Custom CSS Based on Toggle
def inject_custom_css(dark):
    if dark:
        st.markdown("""
            <style>
                body, .stApp {
                    background-color: #0d0d0d;
                    color: #ff99cc;
                }
                .stTextInput > div > div > input,
                .stTextArea > div > textarea {
                    background-color: #1a1a1a;
                    color: #ff99cc;
                }
                .stButton > button {
                    background-color: #ff3399;
                    color: white;
                }
                .stButton > button:hover {
                    background-color: #cc0066;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                body, .stApp {
                    background-color: #fff0f5;
                    color: #000;
                }
                .stTextInput > div > div > input,
                .stTextArea > div > textarea {
                    background-color: #ffe6f0;
                    color: #000;
                }
                .stButton > button {
                    background-color: #ff3399;
                    color: white;
                }
                .stButton > button:hover {
                    background-color: #cc0066;
                }
            </style>
        """, unsafe_allow_html=True)

inject_custom_css(dark_mode)

# Cache the predictor for efficiency
@st.cache_resource
def load_predictor():
    return TrulyLazyPredictor(config.MODEL_PATH)

predictor = load_predictor()

# --- UI Layout ---
st.title("📑 SkimLit Abstract Classifier")
st.markdown("""
Enter a scientific paper or article URL (PubMed or arXiv preferred), and we'll extract its content, classify sentences into sections like:

- **Background**
- **Objective**
- **Methods**
- **Results**
- **Conclusions**

Each section is auto-labeled and presented in a readable format.
""")

# --- URL Input ---
url = st.text_input("🔗 Enter URL to analyze", placeholder="https://pubmed.ncbi.nlm.nih.gov/...")
go = st.button("Predict")

# --- On Predict ---
if go and url:
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    with st.spinner("Fetching and classifying sentences..."):
        result = predictor.predict_from_url(url)

    # --- Display Results ---
    if result["success"]:
        st.success("✅ Prediction completed!")
        st.markdown(f"**🔗 URL:** {result['url']}")
        st.markdown(f"** Total Sentences:** {result['total_sentences']}")

        st.divider()
        st.header("📘 Section-wise Predictions")

        for section, sentences in result["structured_sections"].items():
            with st.expander(f"📂 {section} ({len(sentences)} sentences)"):
                for i, sent in enumerate(sentences, 1):
                    st.markdown(f"{i}. {sent}")

    else:
        st.error("❌ Prediction failed.")
        st.code(result["error"], language="text")
