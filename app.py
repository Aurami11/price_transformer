import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
import os

# Configuration de la page
st.set_page_config(page_title="Quant Dashboard V3", layout="wide")
st.title("⚡ Dashboard Quantitatif : Stratégie GDELT (Édition Finale)")

# ==========================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================
@st.cache_data
def load_data():
    try:
        mat_w = pd.read_csv("dashboard_data/matrice_w.csv", index_col=0)
        ret = pd.read_csv("dashboard_data/market_returns.csv", index_col=0, parse_dates=True)
        z20 = pd.read_csv("dashboard_data/macro_z20.csv", index_col=0, parse_dates=True)
        ema5 = pd.read_csv("dashboard_data/macro_ema5.csv", index_col=0, parse_dates=True)
        
        country_data = None
        if os.path.exists("dashboard_data/country_macro.csv"):
            country_data = pd.read_csv("dashboard_data/country_macro.csv", parse_dates=['Trading_Date'])
            
        return mat_w, ret, z20, ema5, country_data
    except FileNotFoundError:
        st.error("Fichiers introuvables. Faites d'abord tourner main.py pour générer les CSV.")
        st.stop()

matrice_w, market_returns, macro_z20, macro_ema5, country_data = load_data()

# ==========================================
# 2. MOTEUR DE BACKTEST 
# ==========================================
def run_dynamic_backtest(macro_signals, mat_w, returns, strategy, top_n):
    asset_signals = macro_signals.dot(mat_w.T)
    
    asset_signals.index = pd.to_datetime(asset_signals.index, utc=True)
    returns.index = pd.to_datetime(returns.index, utc=True)
    
    asset_signals.index = asset_signals.index.tz_localize(None).normalize()
    returns.index = returns.index.tz_localize(None).normalize()
    
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
    
    return eq_curve, drawdown, ann_ret, sharpe, drawdown.min(), weights

# ==========================================
# 3. INTERFACE UTILISATEUR
# ==========================================
st.sidebar.header("⚙️ Paramètres du Backtest")
metric_choice = st.sidebar.radio("Métrique Temporelle", ["Z20 (Anomalie)", "EMA5 (Tendance)"])
strat_choice = st.sidebar.selectbox("Stratégie d'Allocation", ["long_only", "long_short"])
top_n_choice = st.sidebar.slider("Concentration (Top N)", min_value=1, max_value=20, value=5)

selected_macro = macro_z20 if "Z20" in metric_choice else macro_ema5

eq_curve, drawdown, ann_ret, sharpe, max_dd, weights = run_dynamic_backtest(
    selected_macro, matrice_w, market_returns, strat_choice, top_n_choice
)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Perf & Allocation", "🌍 Macro & Pays", "🧠 Espace Latent (Actifs)", "📊 Matrice Nette"])

# --- ONGLET 1 : PERFORMANCES ET ALLOCATION ANIMÉE ---
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Ratio de Sharpe", f"{sharpe:.2f}")
    col2.metric("Rendement Annualisé", f"{ann_ret*100:.2f} %")
    col3.metric("Max Drawdown", f"{max_dd*100:.2f} %")
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=eq_curve.index, y=eq_curve.values, name="Portefeuille", line=dict(color='#2980b9', width=2)))
    fig_perf.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown", fill='tozeroy', yaxis='y2', line=dict(color='#e74c3c', width=0), opacity=0.3))
    fig_perf.update_layout(height=400, template="plotly_white", margin=dict(t=10, b=10), yaxis2=dict(overlaying='y', side='right', range=[-0.5, 0], showgrid=False))
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown("---")
    st.subheader("▶️ Film des Allocations (Achat/Vente en Auto-Play)")
    
    # -----------------------------------------------------
    # NOUVEAUTÉ : Animation des Allocations
    # -----------------------------------------------------
    # On isole les actions qui ont été tradées au moins une fois
    active_tickers = weights.columns[(weights != 0).any(axis=0)]
    df_anim = weights[active_tickers].reset_index().melt(id_vars='index', var_name='Ticker', value_name='Poids')
    df_anim.rename(columns={'index': 'Date'}, inplace=True)
    df_anim['Date_str'] = df_anim['Date'].dt.strftime('%Y-%m-%d')
    
    # Nettoyage pour que l'animation soit fluide
    df_anim = df_anim[df_anim['Poids'] != 0]
    df_anim = df_anim.sort_values(['Date', 'Poids']) 
    df_anim['Sens'] = np.where(df_anim['Poids'] > 0, 'Long (Achat)', 'Short (Vente)')
    
    if not df_anim.empty:
        # On fixe les limites de l'axe X pour éviter que le graphe ne saute
        max_weight = 1.0 / top_n_choice
        
        fig_alloc = px.bar(
            df_anim,
            x="Poids", y="Ticker",
            color="Sens",
            color_discrete_map={'Long (Achat)': '#2ecc71', 'Short (Vente)': '#e74c3c'},
            animation_frame="Date_str",
            animation_group="Ticker",
            orientation='h',
            range_x=[-max_weight * 1.2, max_weight * 1.2]
        )
        
        fig_alloc.update_layout(
            height=500, template="plotly_white",
            xaxis_tickformat=".0%",
            yaxis_title="",
            margin=dict(t=30)
        )
        # Accélérer la vitesse d'animation
        if len(fig_alloc.layout.updatemenus) > 0:
            fig_alloc.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 400
            
        st.plotly_chart(fig_alloc, use_container_width=True)
    else:
        st.info("Aucun signal généré pour cette configuration.")


# --- ONGLET 2 : MACRO ET PAYS ÉPURÉ ---
with tab2:
    if country_data is not None:
        st.subheader("Série Temporelle : Évolution d'un Concept par Pays")
        
        concept_list = country_data['Concept'].unique()
        selected_concept = st.selectbox("Choisissez le concept à analyser :", concept_list)
        
        df_filtered = country_data[country_data['Concept'] == selected_concept]
        df_time_country = df_filtered.pivot(index='Trading_Date', columns='Country', values='Z20').fillna(0)
        
        # -----------------------------------------------------
        # NOUVEAUTÉ : Multiselect avec Pays par défaut
        # -----------------------------------------------------
        all_countries = sorted(list(df_time_country.columns))
        # On met US et Chine par défaut si dispo, sinon on prend les 2 premiers
        default_countries = [c for c in ['US', 'CH'] if c in all_countries]
        if not default_countries and len(all_countries) > 0:
            default_countries = all_countries[:2]
            
        selected_countries = st.multiselect(
            "Pays affichés (Ajoutez-en pour comparer) :", 
            options=all_countries, 
            default=default_countries
        )
        
        if selected_countries:
            fig_time_country = px.line(
                df_time_country[selected_countries], 
                title=f"Intensité médiatique (Z20) du concept : {selected_concept}", 
                template="plotly_white"
            )
            st.plotly_chart(fig_time_country, use_container_width=True)
        else:
            st.warning("Veuillez sélectionner au moins un pays.")
        
        st.markdown("---")
        st.subheader("PCA : Proximité Géopolitique et Thématique")
        
        country_concept_matrix = country_data.groupby(['Country', 'Concept'])['Z20'].mean().unstack(fill_value=0)
        pca_geo = PCA(n_components=2)
        coords_geo = pca_geo.fit_transform(country_concept_matrix)
        
        df_pca_geo = pd.DataFrame({'X': coords_geo[:, 0], 'Y': coords_geo[:, 1], 'Pays': country_concept_matrix.index})
        
        fig_pca_geo = px.scatter(df_pca_geo, x='X', y='Y', text='Pays', height=600, template="plotly_white", size_max=15)
        fig_pca_geo.update_traces(textposition='top center', marker=dict(size=12, color='#9b59b6', line=dict(width=1, color='white')))
        st.plotly_chart(fig_pca_geo, use_container_width=True)
        
    else:
        st.warning("⚠️ Les données par pays n'ont pas été trouvées.")

# --- ONGLET 3 : ESPACE LATENT (ACTIFS) ---
with tab3:
    st.subheader("PCA : Comment l'IA voit les entreprises")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrice_w)
    
    df_pca = pd.DataFrame({
        'X': coords[:, 0], 'Y': coords[:, 1],
        'Ticker': matrice_w.index,
        'Secteur Dominant': matrice_w.idxmax(axis=1).values
    })
    
    fig_pca = px.scatter(df_pca, x='X', y='Y', text='Ticker', color='Secteur Dominant', height=700, template="plotly_white")
    fig_pca.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='white')))
    st.plotly_chart(fig_pca, use_container_width=True)

# --- ONGLET 4 : MATRICE NETTE ---
with tab4:
    st.subheader("Poids Sémantiques (Entreprise x Concept)")
    fig_mat = px.imshow(matrice_w, aspect="auto", color_continuous_scale="Blues", height=800)
    st.plotly_chart(fig_mat, use_container_width=True)