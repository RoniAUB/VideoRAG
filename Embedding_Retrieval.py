import faiss
import hnswlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import re

def srt_to_json(srt_path, json_path):
    pattern = re.compile(r'\d+\s*\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\d+\n|\Z)', re.DOTALL)
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    matches = pattern.findall(srt_content)
    transcript = []
    for start, end, text in matches:
        clean_text = text.replace('\n', ' ').strip()
        transcript.append({
            'start': start,
            'end': end,
            'text': clean_text
        })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2)
    
    srt_to_json('Whisper.srt', 'Whisper_transcript.json')
    srt_to_json('Faster_Whisper.srt', 'Faster_Whisper_transcript.json')

class HNSWSearcher:
    def __init__(self, index_path, embeddings_path, transcript_path, model_name='all-MiniLM-L6-v2'):
        # Load transcript from JSON file
        with open(transcript_path, 'r', encoding='utf-8') as f:
            self.transcript = json.load(f)
        
        # Load embeddings to get dimension
        self.embeddings = np.load(embeddings_path)
        dim = self.embeddings.shape[1]
        
        # Load HNSW index from .bin file
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.load_index(index_path)
        
        # Load embedding model
        self.model = SentenceTransformer(model_name)
    
    def search(self, query, top_k=5):
        # Encode query to vector
        query_vec = self.model.encode([query]).astype(np.float32)
        
        # Perform search in HNSWlib
        labels, distances = self.index.knn_query(query_vec, k=top_k)
        
        # Collect results with phrase and timestamp
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            entry = self.transcript[idx]
            results.append({
                'phrase': entry['text'],
                'start': entry['start'],
                'end': entry['end'],
                'distance': float(dist)
            })
        return results

class FaissSearcher:
    def __init__(self, faiss_index_path, embeddings_path, transcript_path, model_name='all-MiniLM-L6-v2'):
        self.embeddings = np.load(embeddings_path)
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            self.transcript = json.load(f)
        
        self.model = SentenceTransformer(model_name)
        dim = self.embeddings.shape[1]
        
        # Load Faiss index
        self.index = faiss.read_index(faiss_index_path)

    def search(self, query, top_k=5):
        query_vec = self.model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            entry = self.transcript[idx]
            results.append({
                'phrase': entry['text'],
                'start': entry['start'],
                'end': entry['end'],
                'score': float(dist)
            })
        
        return results


def retrieve(query,model_name="all-MiniLM-L6-v2",Transcription_model="Faster_Whisper", top_k=5):
    transcript_path=f"{Transcription_model}_transcript.json"
    embeddings_path=f"{Transcription_model}_{model_name}_embeddings.npy"
    faiss_index_path=f"faiss_{Transcription_model}_{model_name}.index"
    hnsw_index_path=f"hnsw_{Transcription_model}_{model_name}.bin"

    # Initialize Faiss searcher
    faiss_searcher = FaissSearcher(
        faiss_index_path,
        embeddings_path,
        transcript_path,
        model_name
    )
    
    # Initialize HNSW searcher
    hnsw_searcher = HNSWSearcher(
        hnsw_index_path,
        embeddings_path,
        transcript_path,
        model_name
    )
    
    # Run searches
    faiss_raw_results = faiss_searcher.search(query, top_k=top_k)
    hnsw_raw_results = hnsw_searcher.search(query, top_k=top_k)

    # Process results consistently
    def process_results(results, score_field):
        processed = []
        for r in results:
            processed.append({
                'start': r.get('start'),
                'end': r.get('end'),
                'phrase': r.get('phrase'),
                'score': r.get(score_field)
            })
        return processed

    return {
        'faiss': process_results(faiss_raw_results, 'score'),
        'hnsw': process_results(hnsw_raw_results, 'distance')
    }

if __name__ == "__main__":

    results = retrieve("introduction to complexity", top_k=5)

    print("FAISS RESULTS:")
    for r in results['faiss']:
        print(f"{r['start']} → {r['end']} | {r['phrase']} | Score: {r['score']:.4f}")

    print("\nHNSW RESULTS:")
    for r in results['hnsw']:
        print(f"{r['start']} → {r['end']} | {r['phrase']} | Distance: {r['score']:.4f}")
