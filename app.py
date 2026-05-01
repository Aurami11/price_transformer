import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

# Configuration de la page
st.set_page_config(page_title="Quant Dashboard", layout="wide")
st.title("⚡ Dashboard Quantitatif : Stratégie GDELT")

# ==========================================
# 1. CHARGEMENT DES DONNÉES (Mises en cache pour la vitesse)
# ==========================================
@st.cache_data
def load_data():
    try:
        mat_w = pd.read_csv("dashboard_data/matrice_w.csv", index_col=0)
        ret = pd.read_csv("dashboard_data/market_returns.csv", index_col=0, parse_dates=True)
        z20 = pd.read_csv("dashboard_data/macro_z20.csv", index_col=0, parse_dates=True)
        ema5 = pd.read_csv("dashboard_data/macro_ema5.csv", index_col=0, parse_dates=True)
        return mat_w, ret, z20, ema5
    except FileNotFoundError:
        st.error("Fichiers introuvables. Faites d'abord tourner main.py pour générer le dossier dashboard_data.")
        st.stop()

matrice_w, market_returns, macro_z20, macro_ema5 = load_data()

# ==========================================
# 2. LOGIQUE DU MOTEUR DE BACKTEST DYNAMIQUE
# ==========================================
def run_dynamic_backtest(macro_signals, mat_w, returns, strategy, top_n):
    # Multiplication Macro x Micro
    asset_signals = macro_signals.dot(mat_w.T)
    
    # Alignement temporel
    if asset_signals.index.tz is not None: asset_signals.index = asset_signals.index.tz_localize(None)
    if returns.index.tz is not None: returns.index = returns.index.tz_localize(None)
    
    sigs, rets = asset_signals.align(returns, join='inner', axis=0)
    weights = pd.DataFrame(0.0, index=sigs.index, columns=sigs.columns)

    for date, row in sigs.iterrows():
        if row.sum() == 0: continue
        ranked = row[row != 0].sort_values(ascending=False)
        current_top_n = min(top_n, len(ranked) // 2 if strategy == 'long_short' else len(ranked))
        if current_top_n == 0: continue
        
        if strategy == 'long_only':
            weights.loc[date, ranked.head(current_top_n).index] = 1.0 / current_top_n
        elif strategy == 'long_short':
            weights.loc[date, ranked.head(current_top_n).index] = 1.0 / current_top_n
            weights.loc[date, ranked.tail(current_top_n).index] = -1.0 / current_top_n

    port_ret = (weights * rets).sum(axis=1)
    eq_curve = (1 + port_ret).cumprod()
    
    ann_ret = (eq_curve.iloc[-1]**(252/len(eq_curve)) - 1) if len(eq_curve)>0 else 0
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    drawdown = (eq_curve - eq_curve.cummax()) / eq_curve.cummax()
    
    return eq_curve, drawdown, ann_ret, sharpe, drawdown.min()

# ==========================================
# 3. INTERFACE UTILISATEUR (SIDEBAR & ONGLETS)
# ==========================================
st.sidebar.header("⚙️ Paramètres du Backtest")
metric_choice = st.sidebar.radio("Métrique Temporelle", ["Z20 (Anomalie)", "EMA5 (Tendance)"])
strat_choice = st.sidebar.selectbox("Stratégie d'Allocation", ["long_only", "long_short"])
top_n_choice = st.sidebar.slider("Concentration (Top N)", min_value=1, max_value=20, value=5)

# Sélection des données selon le choix
selected_macro = macro_z20 if "Z20" in metric_choice else macro_ema5

# Recalcul instantané
eq_curve, drawdown, ann_ret, sharpe, max_dd = run_dynamic_backtest(
    selected_macro, matrice_w, market_returns, strat_choice, top_n_choice
)

# Création des onglets
tab1, tab2, tab3 = st.tabs(["📈 Performances (Sweep)", "🧠 Espace Latent (PCA)", "📊 Matrice Nette"])

# --- ONGLET 1 : PERFORMANCES ---
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Ratio de Sharpe", f"{sharpe:.2f}")
    col2.metric("Rendement Annualisé", f"{ann_ret*100:.2f} %")
    col3.metric("Max Drawdown", f"{max_dd*100:.2f} %")
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=eq_curve.index, y=eq_curve.values, name="Portefeuille", line=dict(color='#2980b9', width=2)))
    fig_perf.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown", fill='tozeroy', yaxis='y2', line=dict(color='#e74c3c', width=0)))
    
    fig_perf.update_layout(
        height=600, template="plotly_white", margin=dict(t=30),
        yaxis=dict(title="Valeur du Portefeuille (Base 1)"),
        yaxis2=dict(title="Drawdown", overlaying='y', side='right', range=[-0.5, 0], showgrid=False)
    )
    st.plotly_chart(fig_perf, use_container_width=True)

# --- ONGLET 2 : ESPACE LATENT (PCA) ---
with tab2:
    st.markdown("### Projection des Actifs dans l'Univers Sémantique")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrice_w)
    
    df_pca = pd.DataFrame({
        'X': coords[:, 0], 'Y': coords[:, 1],
        'Ticker': matrice_w.index,
        'Secteur Dominant': matrice_w.idxmax(axis=1).values
    })
    
    fig_pca = px.scatter(
        df_pca, x='X', y='Y', text='Ticker', color='Secteur Dominant', 
        height=700, template="plotly_white"
    )
    fig_pca.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='white')))
    st.plotly_chart(fig_pca, use_container_width=True)

# --- ONGLET 3 : MATRICE ---
with tab3:
    st.markdown("### Poids Sémantiques (Micro x Macro)")
    fig_mat = px.imshow(matrice_w, aspect="auto", color_continuous_scale="Blues", height=800)
    st.plotly_chart(fig_mat, use_container_width=True)