import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.pdf_processor import extract_text_from_pdf
from utils.keyword_extractor import extract_keywords, calculate_similarity
from utils.network_builder import build_network_data, build_paper_detail_network

st.set_page_config(page_title="論文ネットワーク解析", layout="wide", page_icon="📚")

# セッション状態の初期化
if 'papers' not in st.session_state:
    st.session_state.papers = []
if 'selected_paper' not in st.session_state:
    st.session_state.selected_paper = None

st.title("📚 論文ネットワーク解析システム")
st.markdown("最大10個の論文をアップロードして、3Dネットワーク図で類似度を可視化します")

# サイドバー：論文アップロード
with st.sidebar:
    st.header("論文アップロード")
    uploaded_files = st.file_uploader(
        "PDFファイルを選択（最大10個）",
        type=['pdf'],
        accept_multiple_files=True,
        help="論文のPDFファイルをアップロードしてください"
    )
    
    if uploaded_files and len(uploaded_files) > 10:
        st.error("⚠️ 最大10個までのファイルをアップロードできます")
        uploaded_files = uploaded_files[:10]
    
    if st.button("解析開始", type="primary", disabled=not uploaded_files):
        with st.spinner("論文を解析中..."):
            st.session_state.papers = []
            for idx, file in enumerate(uploaded_files):
                text = extract_text_from_pdf(file)
                keywords = extract_keywords(text)
                st.session_state.papers.append({
                    'id': idx,
                    'name': file.name,
                    'text': text,
                    'keywords': keywords
                })
            st.success(f"✅ {len(uploaded_files)}個の論文を解析しました")
    
    if st.session_state.papers:
        st.divider()
        st.subheader("アップロード済み論文")
        for paper in st.session_state.papers:
            st.text(f"📄 {paper['name']}")

# メインエリア
if not st.session_state.papers:
    st.info("👈 左側のサイドバーから論文をアップロードして解析を開始してください")
else:
    # タブで表示を切り替え
    tab1, tab2 = st.tabs(["🌐 全体ネットワーク", "🔍 論文詳細"])
    
    with tab1:
        st.subheader("論文とキーワードの3Dネットワーク")
        
        # ネットワークデータの構築
        network_data = build_network_data(st.session_state.papers)
        
        # 3D可視化
        fig = go.Figure(data=[
            # 論文間のエッジ（青色）
            go.Scatter3d(
                x=network_data['edge_x'],
                y=network_data['edge_y'],
                z=network_data['edge_z'],
                mode='lines',
                line=dict(
                    color='rgba(59, 130, 246, 0.4)',
                    width=2
                ),
                hoverinfo='none',
                showlegend=False,
                name='論文間の類似'
            ),
            # キーワードエッジ（緑色）
            go.Scatter3d(
                x=network_data['keyword_edge_x'],
                y=network_data['keyword_edge_y'],
                z=network_data['keyword_edge_z'],
                mode='lines',
                line=dict(
                    color='rgba(16, 185, 129, 0.3)',
                    width=1
                ),
                hoverinfo='none',
                showlegend=False,
                name='論文-キーワード'
            ),
            # 論文ノード
            go.Scatter3d(
                x=network_data['paper_x'],
                y=network_data['paper_y'],
                z=network_data['paper_z'],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='#3b82f6',
                    symbol='diamond',
                    line=dict(color='white', width=2)
                ),
                text=network_data['paper_labels'],
                textposition='top center',
                hovertext=network_data['paper_hover'],
                hoverinfo='text',
                name='論文',
                customdata=network_data['paper_ids']
            ),
            # キーワードノード
            go.Scatter3d(
                x=network_data['keyword_x'],
                y=network_data['keyword_y'],
                z=network_data['keyword_z'],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='#10b981',
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                text=network_data['keyword_labels'],
                textposition='top center',
                hovertext=network_data['keyword_hover'],
                hoverinfo='text',
                name='キーワード'
            )
        ])
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title=''),
            ),
            showlegend=True,
            height=700,
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode='closest'
        )
        
        # グラフをクリック可能にする
        selected_points = st.plotly_chart(fig, use_container_width=True, key="main_network")
        
        # 論文選択UI
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_paper_name = st.selectbox(
                "論文を選択して詳細を表示",
                options=[p['name'] for p in st.session_state.papers],
                key="paper_selector"
            )
        with col2:
            if st.button("詳細表示", type="primary"):
                st.session_state.selected_paper = next(
                    p for p in st.session_state.papers if p['name'] == selected_paper_name
                )
                st.rerun()
    
    with tab2:
        if st.session_state.selected_paper:
            paper = st.session_state.selected_paper
            st.subheader(f"📄 {paper['name']}")
            
            # 論文内キーワードネットワーク
            st.markdown("### キーワード関係図")
            detail_network = build_paper_detail_network(paper)
            
            fig_detail = go.Figure(data=[
                # エッジ
                go.Scatter3d(
                    x=detail_network['edge_x'],
                    y=detail_network['edge_y'],
                    z=detail_network['edge_z'],
                    mode='lines',
                    line=dict(color='rgba(125,125,125,0.3)', width=2),
                    hoverinfo='none',
                    showlegend=False
                ),
                # キーワードノード
                go.Scatter3d(
                    x=detail_network['node_x'],
                    y=detail_network['node_y'],
                    z=detail_network['node_z'],
                    mode='markers+text',
                    marker=dict(
                        size=detail_network['node_sizes'],
                        color=detail_network['node_colors'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="重要度"),
                        line=dict(color='white', width=2)
                    ),
                    text=detail_network['node_labels'],
                    textposition='top center',
                    hovertext=detail_network['node_hover'],
                    hoverinfo='text'
                )
            ])
            
            fig_detail.update_layout(
                scene=dict(
                    xaxis=dict(showbackground=False, showticklabels=False, title=''),
                    yaxis=dict(showbackground=False, showticklabels=False, title=''),
                    zaxis=dict(showbackground=False, showticklabels=False, title=''),
                ),
                height=600,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig_detail, use_container_width=True)
            
            # キーワードリスト
            st.markdown("### 抽出されたキーワード")
            cols = st.columns(3)
            for idx, (keyword, score) in enumerate(paper['keywords'][:15]):
                with cols[idx % 3]:
                    st.metric(keyword, f"{score:.3f}")
            
            if st.button("← 全体ネットワークに戻る"):
                st.session_state.selected_paper = None
                st.rerun()
        else:
            st.info("左の「全体ネットワーク」タブから論文を選択してください")
