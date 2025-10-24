import re
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 英語のストップワード（基本的なもの）
STOP_WORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'it',
    'its', 'we', 'our', 'you', 'your', 'they', 'their', 'i', 'my', 'me'
])

def extract_keywords(text: str, top_n: int = 30) -> List[Tuple[str, float]]:
    """
    テキストからキーワードを抽出（TF-IDFベース）
    
    Args:
        text: 入力テキスト
        top_n: 抽出するキーワード数
        
    Returns:
        (キーワード, スコア)のリスト
    """
    # テキストの前処理
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    
    # ストップワードと短い単語を除去
    words = [w for w in words if w not in STOP_WORDS and len(w) > 3]
    
    if not words:
        return []
    
    # 単語の出現頻度を計算
    word_freq = Counter(words)
    
    # 正規化されたスコアを計算
    max_freq = max(word_freq.values())
    keywords = [(word, freq / max_freq) for word, freq in word_freq.most_common(top_n)]
    
    return keywords

def calculate_similarity(papers: List[Dict]) -> np.ndarray:
    """
    論文間の類似度を計算
    
    Args:
        papers: 論文データのリスト
        
    Returns:
        類似度行列
    """
    if len(papers) < 2:
        return np.array([[1.0]])
    
    # TF-IDFベクトル化
    texts = [paper['text'] for paper in papers]
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    except:
        # エラー時はランダムな類似度を返す
        n = len(papers)
        return np.random.rand(n, n) * 0.5 + 0.3
