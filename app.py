# app.py
import os
import streamlit as st
import PyPDF2
from difflib import SequenceMatcher
import pandas as pd

# Try to load OpenAI (new SDK). If not present or no key, app will use mock summarizer.
USE_OPENAI = False
try:
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        USE_OPENAI = True
except Exception:
    # openai not available or not configured
    USE_OPENAI = False

st.set_page_config(page_title="Regulation Comparator", layout="wide")
st.title("📑 Automotive Regulation Comparator")
st.caption("Upload two regulatory PDFs. Summaries and differences will appear in a table.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    similarity_threshold = st.slider("Show differences with similarity below:", 0.0, 1.0, 0.75)
    max_chunk_chars = st.number_input("Max characters per chunk (for summarization)", min_value=500, max_value=5000, value=3000, step=100)
    use_mock = st.checkbox("Force mock summarizer (no OpenAI calls)", value=not USE_OPENAI)
    st.markdown("---")
    st.caption("If you have an OpenAI API key set in environment as OPENAI_API_KEY, the app will use it unless you force mock.")

# Utility: PDF text extraction
def extract_text_from_pdf_fileobj(file_obj):
    """Extract text from an uploaded file-like object (PyPDF2)."""
    reader = PyPDF2.PdfReader(file_obj)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages).strip()

# Utility: naive chunker by characters
def chunk_text(text, max_chars=3000):
    """Split text into chunks of roughly max_chars at nearest newline."""
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + max_chars
        if end >= n:
            chunks.append(text[start:].strip())
            break
        # try to break at last double newline for cleaner chunk boundaries
        slice_ = text[start:end]
        cut = slice_.rfind("\n\n")
        if cut == -1:
            cut = slice_.rfind("\n")
        if cut == -1:
            cut = max_chars
        chunks.append(text[start:start+cut].strip())
        start = start + cut
    return [c for c in chunks if c]

# Fallback/mock summarizer (very simple extractive)
def mock_summarize(chunk):
    """Return short extractive-style summary of chunk (first 2-3 sentences)."""
    # naive split into sentences
    sentences = [s.strip() for s in chunk.replace("\r", " ").split(".") if s.strip()]
    if not sentences:
        return ""
    summary_sents = sentences[:3]
    return ". ".join(summary_sents) + (". " if summary_sents else "")

# OpenAI summarizer (new client API usage). Will try to call chat completion for each chunk.
def openai_summarize(chunk):
    """Call OpenAI to summarize a chunk. Returns text or raises exception."""
    # If client not configured, raise
    if not USE_OPENAI:
        raise RuntimeError("OpenAI client not configured")
    try:
        # Use chat completions via new SDK
        # Keep prompt short and instructive
        prompt = (
            "You are a concise legal/compliance assistant. "
            "Summarize the following regulatory text into short bullet points (3-6 bullets), "
            "focusing on requirements and obligations. Be concise and use plain English.\n\n"
            f"{chunk}"
        )
        # Adjust model name if needed (gpt-5-nano, gpt-4o-mini, etc.). Use a generic placeholder here.
        resp = client.chat.completions.create(
            model="gpt-5-nano",  # change to available model if necessary
            messages=[{"role":"user","content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        # New SDK returns object; extract text robustly:
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            # resp.choices[0].message.content (typical)
            choice = resp.choices[0]
            # depending on SDK shape:
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content.strip()
            # fallback to dict-style access:
            try:
                return choice["message"]["content"].strip()
            except Exception:
                return str(resp)
        else:
            return str(resp)
    except Exception as e:
        # Return a visible error summary so the UI does not break
        return f"[OpenAI summarization failed: {e}]"

# Summarize a full document by chunking
def summarize_document(text, max_chars, use_mock_summarizer):
    chunks = chunk_text(text, max_chars)
    summaries = []
    for c in chunks:
        if use_mock_summarizer:
            summaries.append(mock_summarize(c))
        else:
            # try OpenAI, fall back to mock on any error
            try:
                s = openai_summarize(c)
                # if response looks like an error string, fallback
                if s.strip().startswith("[OpenAI summarization failed"):
                    s = mock_summarize(c)
                summaries.append(s)
            except Exception:
                summaries.append(mock_summarize(c))
    # Combine chunk summaries into single text
    return "\n\n".join([f"Chunk {i+1} Summary:\n{summaries[i]}" for i in range(len(summaries))])

# Simple similarity function using SequenceMatcher (0..1)
def similarity_score(a, b):
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

# Function to create table rows comparing sections
def build_comparison_table(text_a, text_b, max_chars, use_mock_summarizer):
    # Create simple "sections" by splitting into paragraphs (double newlines)
    paras_a = [p.strip() for p in text_a.split("\n\n") if p.strip()]
    paras_b = [p.strip() for p in text_b.split("\n\n") if p.strip()]

    # If documents are short, create single section
    if not paras_a:
        paras_a = [text_a]
    if not paras_b:
        paras_b = [text_b]

    # For simplicity, take up to N top paragraphs from each doc (to keep table manageable)
    N = max( max( len(paras_a), len(paras_b) ), 1 )
    N = min(N, 20)  # limit to first 20 sections for display

    rows = []
    # We'll summarize each paragraph (or chunk) independently
    for i in range(N):
        pa = paras_a[i] if i < len(paras_a) else ""
        pb = paras_b[i] if i < len(paras_b) else ""

        sum_a = summarize_document(pa, max_chars, use_mock_summarizer)
        sum_b = summarize_document(pb, max_chars, use_mock_summarizer)
        sim = similarity_score(sum_a, sum_b)

        rows.append({
            "Section Index": i+1,
            "Doc A Excerpt": (pa[:250] + "...") if len(pa) > 250 else pa,
            "Doc B Excerpt": (pb[:250] + "...") if len(pb) > 250 else pb,
            "Summary A": sum_a,
            "Summary B": sum_b,
            "Similarity": round(sim, 3)
        })
    df = pd.DataFrame(rows)
    return df

# UI: file upload area
col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload Document - A (Left)")
    uploaded_a = st.file_uploader("PDF A (e.g., EU)", type=["pdf"], key="pdf_a")
with col2:
    st.subheader("Upload Document - B (Right)")
    uploaded_b = st.file_uploader("PDF B (e.g., India)", type=["pdf"], key="pdf_b")

# Run comparison when both provided
if uploaded_a and uploaded_b:
    st.info("Extracting text from PDFs...")
    try:
        text_a = extract_text_from_pdf_fileobj(uploaded_a)
    except Exception as e:
        st.error(f"Failed to extract text from Document A: {e}")
        text_a = ""
    try:
        text_b = extract_text_from_pdf_fileobj(uploaded_b)
    except Exception as e:
        st.error(f"Failed to extract text from Document B: {e}")
        text_b = ""

    if not text_a and not text_b:
        st.warning("Could not extract text from the uploaded PDFs. Are they scanned images? Try OCR first.")
    else:
        st.info("Generating summaries and comparison. This may take a moment...")
        use_mock_final = use_mock or (not USE_OPENAI)
        df = build_comparison_table(text_a, text_b, max_chunk_chars, use_mock_final)

        # Filter rows by similarity threshold
        filtered = df[df["Similarity"] < similarity_threshold].reset_index(drop=True)

        st.success("Analysis complete!")

        # Tabs for Results
        tab1, tab2, tab3 = st.tabs(["Summary Table", "Side-by-Side", "Raw Text"])

        with tab1:
            st.header("Summary / Differences Table")
            st.write("Rows shown where similarity < threshold (lower = more different).")
            st.dataframe(filtered, use_container_width=True)

            # Download CSV
            csv = filtered.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV of results", csv, "comparison_results.csv", "text/csv")

        with tab2:
            st.header("Side-by-Side Comparison (first few rows)")
            # Show first 5 rows side-by-side
            for idx, row in filtered.head(5).iterrows():
                st.subheader(f"Section {row['Section Index']}")
                left_col, right_col = st.columns(2)
                with left_col:
                    st.markdown("**Doc A Excerpt**")
                    st.write(row["Doc A Excerpt"])
                    st.markdown("**Summary A**")
                    st.write(row["Summary A"])
                with right_col:
                    st.markdown("**Doc B Excerpt**")
                    st.write(row["Doc B Excerpt"])
                    st.markdown("**Summary B**")
                    st.write(row["Summary B"])
                st.markdown("---")

        with tab3:
            st.header("Extracted Raw Text")
            with st.expander("Document A (full extracted text)"):
                st.text_area("Doc A Text", text_a, height=300)
            with st.expander("Document B (full extracted text)"):
                st.text_area("Doc B Text", text_b, height=300)

else:
    st.info("Upload two PDFs (left and right) to start the comparison.")
    st.markdown("If you don't have an OpenAI key, check the 'Force mock summarizer' option in the sidebar.")

