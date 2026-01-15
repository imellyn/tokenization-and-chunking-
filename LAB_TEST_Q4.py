
from nltk.tokenize import sent_tokenize
import streamlit as st
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


st.set_page_config(page_title="PDF Text Chunking - Sentence Tokenizer", layout="wide")

st.title("📄 PDF Text Extraction & Semantic Sentence Chunking")
st.markdown("Upload a PDF → Extract text → Perform sentence-level chunking using NLTK")

#Step 1 & 2: Upload and Extract Text from PDF
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

extracted_text = ""

if uploaded_file is not None:
    try:
        pdf_reader = PdfReader(uploaded_file)
        text_parts = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  
                text_parts.append(page_text)
        
        extracted_text = "\n\n".join(text_parts).strip()
        
        if not extracted_text:
            st.warning("No text could be extracted from this PDF (possibly scanned/image-based).")
        else:
            st.success(f"Text successfully extracted! ({len(extracted_text):,} characters)")
            
    except Exception as e:
        st.error(f"Error reading PDF: {e}")

# Show extracted text preview 
if extracted_text:
    with st.expander("Full Extracted Text (click to expand)", expanded=False):
        st.text_area("Extracted raw text", extracted_text, height=300)

    # Step 3: Preprocess & Show sentences 58 to 68 
    st.subheader("Step 3: Sample Sentences (indices 58 to 68)")
    
    sentences = sent_tokenize(extracted_text)
    
    if len(sentences) < 58:
        st.warning(f"The document contains only {len(sentences)} sentences.")
    else:
        sample_sentences = sentences[58:69]  
        
        st.info(f"Showing sentences **{58} to {68}** (zero-based indexing: {58}–{68})")
        
        for i, sent in enumerate(sample_sentences, start=58):
            st.markdown(f"**Sentence {i}:**  \n{sent.strip()}")

    # Step 4: Full Sentence Tokenization (Semantic Chunking)
    st.subheader("Step 4: Semantic Chunking - All Sentences")
    
    if st.button("Show All Sentence Chunks", type="primary"):
        if not sentences:
            st.warning("No sentences found.")
        else:
            st.success(f"Total sentences found: **{len(sentences)}**")
            
            # Show in nice numbered list
            for i, sentence in enumerate(sentences, 1):
                st.markdown(f"**Chunk {i}** (sentence)")
                st.write(sentence.strip())
                st.markdown("---")

with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. Upload any PDF file  
    2. Text is extracted using **PyPDF2**  
    3. Text is split into sentences using **NLTK sent_tokenize**  
    4. You can see sample sentences 58–68  
    5. Finally, view all sentence chunks (semantic units)
    """)
    
    st.markdown("---")
    st.caption("Text Chunking Demo • NLTK Sentence Tokenizer • 2025/2026")

if not uploaded_file:
    st.info("Please upload a PDF file to begin text extraction and chunking.")