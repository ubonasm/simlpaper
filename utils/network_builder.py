import numpy as np
from typing import Dict, List
from collections import Counter
from utils.keyword_extractor import calculate_similarity

def build_network_data(papers: List[Dict]) -> Dict:
    """
    全体ネットワークのデータを構築
    
    Args:
        papers: 論文データのリスト
        
    Returns:
        3D可視化用のデータ辞書
    """
    if not papers:
        return _empty_network_data()
    
    n_papers = len(papers)
    
    # 論文の位置を円形に配置
    angles = np.linspace(0, 2 * np.pi, n_papers, endpoint=False)
    radius = 2.0
    paper_positions = {
        i: (radius * np.cos(angle), radius * np.sin(angle), 0)
        for i, angle in enumerate(angles)
    }
    
    # 全論文からキーワードを収集
    all_keywords = Counter()
    for paper in papers:
        if 'keywords' in paper and paper['keywords']:
            for keyword, score in paper['keywords'][:10]:
                all_keywords[keyword] += score
    
    if not all_keywords:
        return _empty_network_data()
    
    # 上位キーワードを選択
    top_keywords = [kw for kw, _ in all_keywords.most_common(15)]
    
    # キーワードの位置を内側の円に配置
    keyword_angles = np.linspace(0, 2 * np.pi, len(top_keywords), endpoint=False)
    keyword_radius = 1.0
    keyword_positions = {
        kw: (
            keyword_radius * np.cos(angle),
            keyword_radius * np.sin(angle),
            0.5
        )
        for kw, angle in zip(top_keywords, keyword_angles)
    }
    
    # 類似度行列を計算
    try:
        similarity_matrix = calculate_similarity(papers)
    except Exception as e:
        print(f"[v0] Error calculating similarity: {e}")
        similarity_matrix = np.eye(n_papers)
    
    edge_x, edge_y, edge_z = [], [], []
    
    # 論文間のエッジ（類似度が高い場合のみ）
    for i in range(n_papers):
        for j in range(i + 1, n_papers):
            similarity = similarity_matrix[i][j]
            if similarity > 0.3:  # 閾値
                x0, y0, z0 = paper_positions[i]
                x1, y1, z1 = paper_positions[j]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
    
    # 論文とキーワード間のエッジ用の別データ
    keyword_edge_x, keyword_edge_y, keyword_edge_z = [], [], []
    
    for i, paper in enumerate(papers):
        if 'keywords' not in paper or not paper['keywords']:
            continue
            
        paper_keywords = dict(paper['keywords'][:10])
        x0, y0, z0 = paper_positions[i]
        
        for keyword in top_keywords:
            if keyword in paper_keywords:
                score = paper_keywords[keyword]
                if score > 0.3:  # 閾値を追加
                    x1, y1, z1 = keyword_positions[keyword]
                    keyword_edge_x.extend([x0, x1, None])
                    keyword_edge_y.extend([y0, y1, None])
                    keyword_edge_z.extend([z0, z1, None])
    
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
    keywords = paper.get('keywords', [])[:15]
    n_keywords = len(keywords)
    
    if n_keywords == 0:
        return {
            'edge_x': [], 'edge_y': [], 'edge_z': [],
            'node_x': [], 'node_y': [], 'node_z': [],
            'node_sizes': [], 'node_colors': [], 'node_labels': [], 'node_hover': []
        }
    
    # キーワードを球面上に配置
    phi = np.linspace(0, np.pi, int(np.sqrt(n_keywords)) + 1)
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(n_keywords)) + 1)
    
    positions = []
    for i in range(n_keywords):
        p = phi[i % len(phi)]
        t = theta[i % len(theta)]
        x = np.sin(p) * np.cos(t)
        y = np.sin(p) * np.sin(t)
        z = np.cos(p)
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
