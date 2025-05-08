import streamlit as st
import os
import re
import difflib
from sentence_transformers import SentenceTransformer
import numpy as np
from Semantic_Retrieval import retrieve as Sr
from Embedding_Retrieval import retrieve as Er
import faiss
import hnswlib
from LLM import generate_answer
import time

# --- Streamlit setup
st.set_page_config(layout="wide")
st.title("Video Search and Retrieval")
st.markdown("This app allows you to locate the answer to your questions in a video and play the corresponding video segment.")
st.sidebar.header("Select Video Source")
st.subheader("Retrieval-Augmented Generation (RAG)")

# --- Utility functions
def convert_timestamp(ts_str):
    if isinstance(ts_str, str) and ":" in ts_str:
        try:
            parts = list(map(int, ts_str.strip().split(":")))
            if len(parts) == 2:
                minutes, seconds = parts
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return hours * 3600 + minutes * 60 + seconds
        except:
            st.error("Invalid timestamp format. Use mm:ss or hh:mm:ss.")
            return 0
    try:
        return int(float(ts_str))
    except:
        st.error("Invalid input. Please enter a number or mm:ss.")
        return 0

def time_to_seconds(time_str):
    try:
        hms, ms = time_str.split(',')
        h, m, s = map(int, hms.split(':'))
        total_seconds = h * 3600 + m * 60 + s + int(ms) / 1000
        return total_seconds
    except Exception as e:
        st.error(f"Error converting time '{time_str}': {e}")
        return 0

def parse_srt(srt_file):
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)\s*(?=\n\d|\Z)', re.DOTALL)
    try:
        with open(srt_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()
    except FileNotFoundError:
        st.error(f"Transcript file '{srt_file}' not found.")
        return []
    matches = pattern.findall(srt_content)
    parsed = []
    for start, end, text in matches:
        parsed.append({
            'start': start,
            'end': end,
            'start_sec': time_to_seconds(start),
            'end_sec': time_to_seconds(end),
            'text': text.replace('\n', ' ').strip()
        })
    return parsed

# --- Initialize session state
if "start_time" not in st.session_state:
    st.session_state["start_time"] = 0

# --- Sidebar controls
query = st.sidebar.text_input("Search Phrase in Transcript")
Retrieval_Type = st.sidebar.selectbox("Select Retrieval Type", ("Semantic Retrieval", "Embedding Retrieval"), index=1)
Transcription_Model = st.sidebar.selectbox("Select Transcription Model", ("Faster_Whisper", "Whisper"), index=1)
Embedding_Model = st.sidebar.selectbox("Select Embedding Model", ("multi-qa-MiniLM-L6-cos-v1", "all-MiniLM-L6-v2"), index=0)
top_k = st.sidebar.slider("Number of top results", min_value=1, max_value=10, value=5)

transcript_path = Transcription_Model + ".srt"
transcript = parse_srt(transcript_path)

# --- Perform searches
search_results = {}

# if query.strip():
#     if Retrieval_Type == "Semantic Retrieval":
#         search_results = Sr(query, Model=Transcription_Model, top_k=top_k)
#     else:
#         search_results = Er(query, model_name=Embedding_Model, Transcription_model=Transcription_Model, top_k=top_k)


# COMMENT THIS AREA To get the results without LLM and uncomment the above area

llm_answer = ""

if query.strip():
    if Retrieval_Type == "Semantic Retrieval":
        search_results = Sr(query, Model=Transcription_Model, top_k=top_k)
        combined_results = search_results.get('tfidf', []) + search_results.get('bm25', [])
        context_text = " ".join([res['paragraph'] for res in combined_results])
    else:
        search_results = Er(query, model_name=Embedding_Model, Transcription_model=Transcription_Model, top_k=top_k)
        combined_results = search_results.get('faiss', []) + search_results.get('hnsw', [])
        context_text = " ".join([res['phrase'] for res in combined_results])

    if context_text.strip():
        prompt = f"Given that the data say: {context_text}, what is the answer to: {query}"
        llm_answer = generate_answer(prompt)
if llm_answer:
    st.markdown("### 💬 LLM Answer")
    st.info(llm_answer)





# --- Display video
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.video("Parameterized Complexity of token sliding, token jumping - Amer Mouawad [dARr3lGKwk8].mkv", start_time=st.session_state["start_time"])
    st.write(f"Playing from {st.session_state['start_time']} seconds")

# --- Display search results
st.markdown("### Search Results")

if Retrieval_Type == "Semantic Retrieval":
    col_tfidf, col_bm25 = st.columns(2)
    with col_tfidf:
        st.subheader("TF-IDF Results")
        tfidf_results = search_results.get('tfidf', [])
        if tfidf_results:
            for i, res in enumerate(tfidf_results):
                with st.expander(f"{res['start']} → {res['end']} | Score: {res['score']:.4f}"):
                    st.write(res['paragraph'])
                    if st.button(f"Play TF-IDF Clip {i + 1}"):
                        st.session_state["start_time"] = time_to_seconds(res['start'])
                        st.rerun()
        else:
            st.write("❌ No TF-IDF result found")

    with col_bm25:
        st.subheader("BM25 Results")
        bm25_results = search_results.get('bm25', [])
        if bm25_results:
            for i, res in enumerate(bm25_results):
                with st.expander(f"{res['start']} → {res['end']} | Score: {res['score']:.4f}"):
                    st.write(res['paragraph'])
                    if st.button(f"Play BM25 Clip {i + 1}"):
                        st.session_state["start_time"] = time_to_seconds(res['start'])
                        st.rerun()
        else:
            st.write("❌ No BM25 result found")

else:  # Embedding Retrieval
    col_faiss, col_hnsw = st.columns(2)
    with col_faiss:
        st.subheader("Faiss Results")
        faiss_results = search_results.get('faiss', [])
        if faiss_results:
            for i, res in enumerate(faiss_results):
                with st.expander(f"{res['start']} → {res['end']} | Score: {res['score']:.4f}"):
                    st.write(res['phrase'])
                    if st.button(f"Play Faiss Clip {i + 1}"):
                        st.session_state["start_time"] = time_to_seconds(res['start'])
                        st.rerun()
        else:
            st.write("❌ No Faiss result found")

    with col_hnsw:
        st.subheader("HNSW Results")
        hnsw_results = search_results.get('hnsw', [])
        if hnsw_results:
            for i, res in enumerate(hnsw_results):
                with st.expander(f"{res['start']} → {res['end']} | Score: {res['score']:.4f}"):
                    st.write(res['phrase'])
                    if st.button(f"Play HNSW Clip {i + 1}"):
                        st.session_state["start_time"] = time_to_seconds(res['start'])
                        st.rerun()
        else:
            st.write("❌ No HNSW result found")
