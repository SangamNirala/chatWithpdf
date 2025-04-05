# reranking.py
from typing import List, Dict
from sentence_transformers import CrossEncoder
import os
import google.generativeai as genai
from langchain_core.documents import Document
import requests

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.rerank_model = CrossEncoder(model_name)
        self.gemini_client = genai  # Configured in app.py
        
    def semantic_rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """Rerank documents using cross-encoder model"""
        try:
            doc_texts = [doc.page_content for doc in documents]
            pairs = [(query, doc_text) for doc_text in doc_texts]
            scores = self.rerank_model.predict(pairs)
            
            scored_docs = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            return [doc for doc, _ in scored_docs[:top_n]]
            
        except Exception as e:
            print(f"Reranking error: {str(e)}")
            return documents[:top_n]  # Fallback to original order

    def gemini_rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """Alternative reranking using Gemini models"""
        try:
            doc_texts = [doc.page_content[:2000] for doc in documents]  # Truncate for context
            prompt = f"""Rerank these documents by relevance to query: {query}
                        Documents:\n{'\n\n'.join(doc_texts)}
                        Return only indices of top {top_n} documents in order of relevance."""
                        
            response = self.gemini_client.generate_content(prompt)
            return self._parse_gemini_response(response.text, documents)
            
        except Exception as e:
            print(f"Gemini reranking error: {str(e)}")
            return documents[:top_n]

    def _parse_gemini_response(self, text: str, docs: List[Document]) -> List[Document]:
        """Parse Gemini's numbered list response"""
        indices = []
        for line in text.split('\n'):
            if line.strip().isdigit():
                idx = int(line.strip()) - 1  # Convert to 0-based index
                if 0 <= idx < len(docs):
                    indices.append(idx)
        return [docs[i] for i in indices[:len(docs)]]
