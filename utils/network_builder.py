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
        for keyword, score in paper['keywords'][:10]:
            all_keywords[keyword] += score
    
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
    similarity_matrix = calculate_similarity(papers)
    
    # エッジデータの構築
    edge_x, edge_y, edge_z = [], [], []
    edge_colors, edge_widths = [], []
    
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
                edge_widths.append(similarity * 5)
                edge_colors.append(f'rgba(59, 130, 246, {similarity})')
    
    # 論文とキーワード間のエッジ
    for i, paper in enumerate(papers):
        paper_keywords = dict(paper['keywords'][:10])
        x0, y0, z0 = paper_positions[i]
        
        for keyword in top_keywords:
            if keyword in paper_keywords:
                score = paper_keywords[keyword]
                x1, y1, z1 = keyword_positions[keyword]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
                edge_widths.append(score * 3)
                edge_colors.append(f'rgba(16, 185, 129, {score * 0.7})')
    
    # ノードデータの構築
    paper_x = [pos[0] for pos in paper_positions.values()]
    paper_y = [pos[1] for pos in paper_positions.values()]
    paper_z = [pos[2] for pos in paper_positions.values()]
    paper_labels = [f"P{i+1}" for i in range(n_papers)]
    paper_hover = [f"<b>{p['name']}</b><br>キーワード数: {len(p['keywords'])}" for p in papers]
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
        'edge_colors': edge_colors,
        'edge_widths': edge_widths,
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

def build_paper_detail_network(paper: Dict) -> Dict:
    """
    個別論文のキーワードネットワークを構築
    
    Args:
        paper: 論文データ
        
    Returns:
        3D可視化用のデータ辞書
    """
    keywords = paper['keywords'][:15]
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
