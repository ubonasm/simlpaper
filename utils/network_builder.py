import numpy as np
from typing import Dict, List
from collections import Counter
from utils.keyword_extractor import calculate_similarity

def build_network_data(papers: List[Dict]) -> Dict:
    """
    全体ネットワークのデータを構築（球体レイアウト）
    
    Args:
        papers: 論文データのリスト
        
    Returns:
        3D可視化用のデータ辞書
    """
    if not papers:
        return _empty_network_data()
    
    n_papers = len(papers)
    
    paper_positions = {}
    radius_outer = 3.0
    
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ゴールデンアングル
    
    for i in range(n_papers):
        y = 1 - (i / float(n_papers - 1 if n_papers > 1 else 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = golden_angle * i
        
        x = radius_outer * np.cos(theta) * radius
        z = radius_outer * np.sin(theta) * radius
        y = radius_outer * y
        
        paper_positions[i] = (x, y, z)
    
    # 全論文からキーワードを収集
    all_keywords = Counter()
    for paper in papers:
        if 'keywords' in paper and paper['keywords']:
            for keyword, score in paper['keywords'][:20]:
                if isinstance(keyword, str) and isinstance(score, (int, float)):
                    all_keywords[keyword] += score
    
    if not all_keywords:
        print("[v0] No keywords found in papers")
        return _empty_network_data()
    
    top_keywords = [kw for kw, _ in all_keywords.most_common(25)]
    
    if not top_keywords:
        print("[v0] No top keywords found")
        return _empty_network_data()
    
    keyword_positions = {}
    radius_inner = 1.5
    n_keywords = len(top_keywords)
    
    for i, kw in enumerate(top_keywords):
        y = 1 - (i / float(n_keywords - 1 if n_keywords > 1 else 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = golden_angle * i
        
        x = radius_inner * np.cos(theta) * radius
        z = radius_inner * np.sin(theta) * radius
        y = radius_inner * y
        
        keyword_positions[kw] = (x, y, z)
    
    # 類似度行列を計算
    try:
        similarity_matrix = calculate_similarity(papers)
        if np.any(np.isnan(similarity_matrix)) or np.any(np.isinf(similarity_matrix)):
            print("[v0] Warning: similarity matrix contains NaN or Inf values")
            similarity_matrix = np.eye(n_papers)
    except Exception as e:
        print(f"[v0] Error calculating similarity: {e}")
        similarity_matrix = np.eye(n_papers)
    
    edge_x, edge_y, edge_z = [], [], []
    
    # 論文間のエッジ（類似度が高い場合のみ）
    for i in range(n_papers):
        for j in range(i + 1, n_papers):
            try:
                similarity = float(similarity_matrix[i][j])
                if not np.isnan(similarity) and not np.isinf(similarity) and similarity > 0.3:
                    x0, y0, z0 = paper_positions[i]
                    x1, y1, z1 = paper_positions[j]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_z.extend([z0, z1, None])
            except (IndexError, ValueError, TypeError) as e:
                print(f"[v0] Error processing edge {i}-{j}: {e}")
                continue
    
    # 論文とキーワード間のエッジ用の別データ
    keyword_edge_x, keyword_edge_y, keyword_edge_z = [], [], []
    
    for i, paper in enumerate(papers):
        if 'keywords' not in paper or not paper['keywords']:
            continue
            
        paper_keywords = dict(paper['keywords'][:20])
        x0, y0, z0 = paper_positions[i]
        
        for keyword in top_keywords:
            if keyword in paper_keywords:
                try:
                    score = float(paper_keywords[keyword])
                    if not np.isnan(score) and not np.isinf(score) and score > 0.3:
                        x1, y1, z1 = keyword_positions[keyword]
                        keyword_edge_x.extend([x0, x1, None])
                        keyword_edge_y.extend([y0, y1, None])
                        keyword_edge_z.extend([z0, z1, None])
                except (ValueError, TypeError) as e:
                    print(f"[v0] Error processing keyword edge: {e}")
                    continue
    
    # ノードデータの構築
    paper_x = [pos[0] for pos in paper_positions.values()]
    paper_y = [pos[1] for pos in paper_positions.values()]
    paper_z = [pos[2] for pos in paper_positions.values()]
    paper_labels = [f"P{i+1}" for i in range(n_papers)]
    paper_hover = [
        f"<b>{p.get('name', 'Unknown')}</b><br>キーワード数: {len(p.get('keywords', []))}" 
        for p in papers
    ]
    paper_ids = list(range(n_papers))
    
    keyword_x = [pos[0] for pos in keyword_positions.values()]
    keyword_y = [pos[1] for pos in keyword_positions.values()]
    keyword_z = [pos[2] for pos in keyword_positions.values()]
    keyword_labels = list(top_keywords)
    keyword_hover = [f"<b>{kw}</b><br>出現回数: {all_keywords[kw]:.2f}" for kw in top_keywords]
    
    return {
        'edge_x': edge_x,
        'edge_y': edge_y,
        'edge_z': edge_z,
        'keyword_edge_x': keyword_edge_x,
        'keyword_edge_y': keyword_edge_y,
        'keyword_edge_z': keyword_edge_z,
        'paper_x': paper_x,
        'paper_y': paper_y,
        'paper_z': paper_z,
        'paper_labels': paper_labels,
        'paper_hover': paper_hover,
        'paper_ids': paper_ids,
        'keyword_x': keyword_x,
        'keyword_y': keyword_y,
        'keyword_z': keyword_z,
        'keyword_labels': keyword_labels,
        'keyword_hover': keyword_hover
    }

def _empty_network_data() -> Dict:
    """空のネットワークデータを返す"""
    return {
        'edge_x': [], 'edge_y': [], 'edge_z': [],
        'keyword_edge_x': [], 'keyword_edge_y': [], 'keyword_edge_z': [],
        'paper_x': [], 'paper_y': [], 'paper_z': [],
        'paper_labels': [], 'paper_hover': [], 'paper_ids': [],
        'keyword_x': [], 'keyword_y': [], 'keyword_z': [],
        'keyword_labels': [], 'keyword_hover': []
    }

def build_paper_detail_network(paper: Dict) -> Dict:
    """
    個別論文のキーワードネットワークを構築
    
    Args:
        paper: 論文データ
        
    Returns:
        3D可視化用のデータ辞書
    """
    keywords = paper.get('keywords', [])[:25]
    n_keywords = len(keywords)
    
    if n_keywords == 0:
        return {
            'edge_x': [], 'edge_y': [], 'edge_z': [],
            'node_x': [], 'node_y': [], 'node_z': [],
            'node_sizes': [], 'node_colors': [], 'node_labels': [], 'node_hover': []
        }
    
    positions = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    for i in range(n_keywords):
        y = 1 - (i / float(n_keywords - 1 if n_keywords > 1 else 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = golden_angle * i
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        positions.append((x, y, z))
    
    # エッジデータ（近いキーワード同士を接続）
    edge_x, edge_y, edge_z = [], [], []
    for i in range(n_keywords):
        for j in range(i + 1, min(i + 3, n_keywords)):
            x0, y0, z0 = positions[i]
            x1, y1, z1 = positions[j]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
    
    # ノードデータ
    node_x = [pos[0] for pos in positions]
    node_y = [pos[1] for pos in positions]
    node_z = [pos[2] for pos in positions]
    node_labels = [kw for kw, _ in keywords]
    node_sizes = [10 + score * 20 for _, score in keywords]
    node_colors = [score for _, score in keywords]
    node_hover = [f"<b>{kw}</b><br>スコア: {score:.3f}" for kw, score in keywords]
    
    return {
        'edge_x': edge_x,
        'edge_y': edge_y,
        'edge_z': edge_z,
        'node_x': node_x,
        'node_y': node_y,
        'node_z': node_z,
        'node_sizes': node_sizes,
        'node_colors': node_colors,
        'node_labels': node_labels,
        'node_hover': node_hover
    }
