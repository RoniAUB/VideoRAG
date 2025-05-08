# VideoRAG
This repository contains all the file used to implement the Video RAG along with the Streamlit UI.

# Semantic_Retrieval.py

This script contains two classes and two seperate functions:
  ## BM25Searcher:
  This class is used to save load and search using BM25
  ## TFIDFSearcher:
  This class is used to save load and search using BM25
  ## retrieve:
  This function is a wrapper that is used to obtain the search results of a given query using both methods
  ## load_transcript_with_timestamps
  This function is used to load the corresponding transcript and the timestamps for the given top_k elements

# Embedding_Retrieval.py
This scrip contains two classes and two seperate functions:

  ## FaissSearcher:
  This class is used to save load and search using Faiss indexing
  ## HNSWSearcher:
  This class is used to save load and search using HNSW indexing
  ## retrieve:
  This function is a wrapper that is used to obtain the search results of a given query using both methods
  ## srt_to_json
  This function is used to load the transcript and parse it into a json file to retrieve the data using HNSWSearcher
# UI.py
This is the streamlit User Interface

# LLM.py
This function contains the main LLM model that is used to generate text (for the augmentation part)

# Video_RAG.ipynb
This is a notebook that contains some main functions that we use only once (for example: downloading the video, Seperating the audio, Transcribing the words, embedding the frames, generating the indexes, and finetuninig the LLM)
