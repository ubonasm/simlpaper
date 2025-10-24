import streamlit as st
import plotly.graph_objects as go
import numpy as np
import json
from utils.pdf_processor import extract_text_from_pdf
from utils.keyword_extractor import extract_keywords, calculate_similarity
from utils.network_builder import build_network_data, build_paper_detail_network, filter_network_data

st.set_page_config(page_title="論文ネットワーク解析", layout="wide", page_icon="📚")

# セッション状態の初期化
if 'papers' not in st.session_state:
    st.session_state.papers = []
if 'selected_paper' not in st.session_state:
    st.session_state.selected_paper = None
if 'selected_paper_ids' not in st.session_state:
    st.session_state.selected_paper_ids = []
if 'similarity_matrix' not in st.session_state:
    st.session_state.similarity_matrix = None
if 'network_data' not in st.session_state:
    st.session_state.network_data = None

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
            st.session_state.selected_paper_ids = [p['id'] for p in st.session_state.papers]
            st.session_state.similarity_matrix = calculate_similarity(st.session_state.papers)
            st.session_state.network_data = build_network_data(st.session_state.papers)
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
    tab1, tab2, tab3 = st.tabs(["🌐 全体ネットワーク", "🔍 論文詳細", "📊 類似度分析"])
    
    with tab1:
        st.subheader("論文とキーワードの3Dネットワーク")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### 表示する論文を選択")
            cols = st.columns(min(5, len(st.session_state.papers)))
            for idx, paper in enumerate(st.session_state.papers):
                with cols[idx % len(cols)]:
                    is_selected = st.checkbox(
                        f"P{idx+1}",
                        value=paper['id'] in st.session_state.selected_paper_ids,
                        key=f"paper_select_{idx}",
                        help=paper['name']
                    )
                    if is_selected and paper['id'] not in st.session_state.selected_paper_ids:
                        st.session_state.selected_paper_ids.append(paper['id'])
                    elif not is_selected and paper['id'] in st.session_state.selected_paper_ids:
                        st.session_state.selected_paper_ids.remove(paper['id'])
        
        with col2:
            st.markdown("#### フィルタリング")
            min_similarity = st.slider(
                "最小類似度",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="この値より低い類似度のエッジは表示されません"
            )
            
            max_keywords = st.slider(
                "表示キーワード数",
                min_value=5,
                max_value=30,
                value=25,
                step=5,
                help="表示する最大キーワード数"
            )
        
        st.divider()
        
        network_data = filter_network_data(
            st.session_state.network_data,
            st.session_state.papers,
            min_similarity=min_similarity,
            max_keywords=max_keywords
        )
        
        selected_ids = set(st.session_state.selected_paper_ids)
        
        # 論文ノードの色と透明度を調整
        paper_colors = []
        for paper_id in network_data['paper_ids']:
            if paper_id in selected_ids:
                paper_colors.append('rgba(59, 130, 246, 1.0)')
            else:
                paper_colors.append('rgba(148, 163, 184, 0.2)')
        
        if network_data['keyword_labels']:
            with st.expander("🔍 キーワードでフィルタリング（オプション）"):
                selected_keywords = st.multiselect(
                    "表示するキーワードを選択（空の場合は全て表示）",
                    options=network_data['keyword_labels'],
                    default=[],
                    help="特定のキーワードのみを表示したい場合に選択してください"
                )
                
                if selected_keywords:
                    keyword_indices = [i for i, kw in enumerate(network_data['keyword_labels']) if kw in selected_keywords]
                    network_data['keyword_x'] = [network_data['keyword_x'][i] for i in keyword_indices]
                    network_data['keyword_y'] = [network_data['keyword_y'][i] for i in keyword_indices]
                    network_data['keyword_z'] = [network_data['keyword_z'][i] for i in keyword_indices]
                    network_data['keyword_labels'] = [network_data['keyword_labels'][i] for i in keyword_indices]
                    network_data['keyword_hover'] = [network_data['keyword_hover'][i] for i in keyword_indices]
        
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
                    color=paper_colors,
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
                xaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
                yaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
                zaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
                bgcolor='rgba(240, 240, 250, 0.9)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            height=700,
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="main_network")
        
        st.divider()
        st.markdown("### 📥 エクスポート")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            html_str = fig.to_html(include_plotlyjs='cdn')
            st.download_button(
                label="📄 HTMLとしてダウンロード",
                data=html_str,
                file_name="network_visualization.html",
                mime="text/html",
                help="インタラクティブな3D可視化をHTMLファイルとして保存"
            )
        
        with col2:
            try:
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                st.download_button(
                    label="🖼️ PNGとしてダウンロード",
                    data=img_bytes,
                    file_name="network_visualization.png",
                    mime="image/png",
                    help="静的画像として保存"
                )
            except:
                st.info("PNG出力には追加のライブラリが必要です")
        
        with col3:
            export_data = {
                'papers': [{'id': p['id'], 'name': p['name'], 'keywords': p['keywords'][:10]} for p in st.session_state.papers],
                'network_data': {
                    'nodes': {
                        'papers': [{'id': i, 'label': label, 'position': [x, y, z]} 
                                  for i, label, x, y, z in zip(network_data['paper_ids'], 
                                                                network_data['paper_labels'],
                                                                network_data['paper_x'],
                                                                network_data['paper_y'],
                                                                network_data['paper_z'])],
                        'keywords': [{'label': label, 'position': [x, y, z]} 
                                    for label, x, y, z in zip(network_data['keyword_labels'],
                                                              network_data['keyword_x'],
                                                              network_data['keyword_y'],
                                                              network_data['keyword_z'])]
                    }
                }
            }
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="📊 JSONとしてダウンロード",
                data=json_str,
                file_name="network_data.json",
                mime="application/json",
                help="ネットワークデータをJSON形式で保存"
            )
        
        st.divider()
        
        # 論文選択UI
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
                    xaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
                    yaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
                    zaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
                    bgcolor='rgba(240, 240, 250, 0.9)',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                height=600,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
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
    
    with tab3:
        st.subheader("📊 論文間の類似度分析")
        
        if st.session_state.similarity_matrix is not None:
            similarity_matrix = st.session_state.similarity_matrix
            n_papers = len(st.session_state.papers)
            
            st.markdown("### 📈 統計情報")
            col1, col2, col3, col4 = st.columns(4)
            
            # 上三角行列の値のみを取得（対角線を除く）
            upper_triangle = []
            for i in range(n_papers):
                for j in range(i + 1, n_papers):
                    upper_triangle.append(similarity_matrix[i][j])
            
            if upper_triangle:
                with col1:
                    st.metric("平均類似度", f"{np.mean(upper_triangle):.3f}")
                with col2:
                    st.metric("最大類似度", f"{np.max(upper_triangle):.3f}")
                with col3:
                    st.metric("最小類似度", f"{np.min(upper_triangle):.3f}")
                with col4:
                    st.metric("標準偏差", f"{np.std(upper_triangle):.3f}")
            
            st.divider()
            
            st.markdown("### 🔥 類似度ヒートマップ")
            
            paper_names = [f"P{i+1}" for i in range(n_papers)]
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=paper_names,
                y=paper_names,
                colorscale='RdYlGn',
                text=np.round(similarity_matrix, 3),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="類似度"),
                hoverongaps=False,
                hovertemplate='%{y} と %{x}<br>類似度: %{z:.3f}<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title="論文間の類似度マトリックス",
                xaxis_title="論文",
                yaxis_title="論文",
                height=600,
                width=700
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.divider()
            
            st.markdown("### 📋 論文ペアの詳細類似度")
            
            # 類似度の高い順にソート
            paper_pairs = []
            for i in range(n_papers):
                for j in range(i + 1, n_papers):
                    paper_pairs.append({
                        '論文1': st.session_state.papers[i]['name'],
                        '論文2': st.session_state.papers[j]['name'],
                        '類似度': similarity_matrix[i][j]
                    })
            
            paper_pairs.sort(key=lambda x: x['類似度'], reverse=True)
            
            # データフレームとして表示
            import pandas as pd
            df = pd.DataFrame(paper_pairs)
            
            # 類似度でフィルタリング
            min_sim_filter = st.slider(
                "表示する最小類似度",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="similarity_filter"
            )
            
            filtered_df = df[df['類似度'] >= min_sim_filter]
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=400,
                column_config={
                    "類似度": st.column_config.ProgressColumn(
                        "類似度",
                        help="論文間の類似度スコア",
                        format="%.3f",
                        min_value=0,
                        max_value=1,
                    ),
                }
            )
            
            csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 類似度データをCSVでダウンロード",
                data=csv,
                file_name="similarity_analysis.csv",
                mime="text/csv"
            )
        else:
            st.info("論文を解析すると、類似度分析が表示されます")
