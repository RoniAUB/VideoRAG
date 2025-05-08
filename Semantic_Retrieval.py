import sys
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re

# ---------- BM25 Searcher ----------
class BM25Searcher:
    def __init__(self):
        self.bm25 = None
        self.paragraphs = None
        self.tokenized_corpus = None
        self.timestamps = None

    def fit(self, paragraphs, timestamps):
        self.paragraphs = paragraphs
        self.timestamps = timestamps
        self.tokenized_corpus = [p.lower().split() for p in paragraphs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def save(self, bm25_path, paragraphs_path, timestamps_path):
        joblib.dump(self.bm25, bm25_path)
        joblib.dump(self.paragraphs, paragraphs_path)
        joblib.dump(self.timestamps, timestamps_path)

    def load(self, bm25_path, paragraphs_path, timestamps_path):
        self.bm25 = joblib.load(bm25_path)
        self.paragraphs = joblib.load(paragraphs_path)
        self.timestamps = joblib.load(timestamps_path)

    def search(self, query, top_k=5, threshold=0.3):
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score >= threshold:
                results.append((idx, self.paragraphs[idx], self.timestamps[idx], score))
        if not results:
            return ["No match found"]
        return results


# ---------- TF-IDF Searcher ----------
class TFIDFSearcher:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.paragraphs = None
        self.timestamps = None

    def fit(self, paragraphs, timestamps):
        self.paragraphs = paragraphs
        self.timestamps = timestamps
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(paragraphs)

    def save(self, vectorizer_path, matrix_path, paragraphs_path, timestamps_path):
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.tfidf_matrix, matrix_path)
        joblib.dump(self.paragraphs, paragraphs_path)
        joblib.dump(self.timestamps, timestamps_path)

    def load(self, vectorizer_path, matrix_path, paragraphs_path, timestamps_path):
        self.vectorizer = joblib.load(vectorizer_path)
        self.tfidf_matrix = joblib.load(matrix_path)
        self.paragraphs = joblib.load(paragraphs_path)
        self.timestamps = joblib.load(timestamps_path)

    def search(self, query, top_k=5, threshold=0.2):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score >= threshold:
                results.append((idx, self.paragraphs[idx], self.timestamps[idx], score))
        if not results:
            return ["No match found"]
        return results


# ---------- Transcript Loader ----------
def load_transcript_with_timestamps(filepath):
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)(?=\n\d{2}:\d{2}|\Z)', re.DOTALL)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    matches = pattern.findall(content)
    paragraphs = [m[2].strip().replace('\n', ' ') for m in matches]
    timestamps = [(m[0], m[1]) for m in matches]
    return paragraphs, timestamps
def retrieve(query, Model="Faster_Whisper", top_k=5):
    prefix = Model
    tfidf_searcher = TFIDFSearcher()
    tfidf_searcher.load(f'tfidf_vec_{prefix}.pkl', f'tfidf_mat_{prefix}.pkl', f'paragraphs_{prefix}.pkl', f'timestamps_{prefix}.pkl')

    bm25_searcher = BM25Searcher()
    bm25_searcher.load(f'bm25_{prefix}.pkl', f'paragraphs_{prefix}.pkl', f'timestamps_{prefix}.pkl')

    tfidf_results = tfidf_searcher.search(query, top_k=top_k, threshold=0.2)
    bm25_results = bm25_searcher.search(query, top_k=top_k, threshold=0.3)

    def process_results(raw_results):
        results = []
        if raw_results != ["No match found"]:
            for idx, para, (start, end), score in raw_results:
                results.append({
                    'idx': idx,
                    'paragraph': para,
                    'start': start,
                    'end': end,
                    'score': score
                })
        return results

    return {
        'tfidf': process_results(tfidf_results),
        'bm25': process_results(bm25_results)
    }


if __name__ == "__main__":

    filename = 'Faster_Whisper.srt'
    prefix = 'Faster_Whisper'
    filename2 = 'Whisper.srt'
    prefix2 = 'Whisper'
    paragraphs, timestamps = load_transcript_with_timestamps(filename)
    paragraphs2, timestamps2 = load_transcript_with_timestamps(filename2)

    # Fit and run TF-IDF search
    tfidf_searcher = TFIDFSearcher()
    tfidf_searcher.fit(paragraphs, timestamps)
    tfidf_searcher.save(f'tfidf_vec_{prefix}.pkl', f'tfidf_mat_{prefix}.pkl', f'paragraphs_{prefix}.pkl', f'timestamps_{prefix}.pkl')

    tfidf_searcher2 = TFIDFSearcher()
    tfidf_searcher2.fit(paragraphs2, timestamps2)
    tfidf_searcher2.save(f'tfidf_vec_{prefix2}.pkl', f'tfidf_mat_{prefix2}.pkl', f'paragraphs_{prefix2}.pkl', f'timestamps_{prefix2}.pkl')

    # Fit and run BM25 search
    bm25_searcher = BM25Searcher()
    bm25_searcher.fit(paragraphs, timestamps)
    bm25_searcher.save(f'bm25_{prefix}.pkl', f'paragraphs_{prefix}.pkl', f'timestamps_{prefix}.pkl')
    bm25_searcher2 = BM25Searcher()
    bm25_searcher2.fit(paragraphs2, timestamps2)
    bm25_searcher2.save(f'bm25_{prefix2}.pkl', f'paragraphs_{prefix2}.pkl', f'timestamps_{prefix2}.pkl')
