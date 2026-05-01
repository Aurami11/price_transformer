import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

class QuantReportGenerator:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def calculate_metrics(self, daily_returns, equity_curve):
        trading_days = 252
        cum_return = equity_curve.iloc[-1] - 1
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        ann_return = (1 + cum_return) ** (365.25 / days) - 1 if days > 0 else 0
        ann_vol = daily_returns.std() * np.sqrt(trading_days)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        return {
            "Rendement Cumulé": f"{cum_return*100:.2f} %",
            "Rendement Annualisé": f"{ann_return*100:.2f} %",
            "Volatilité Annualisée": f"{ann_vol*100:.2f} %",
            "Ratio de Sharpe": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd*100:.2f} %"
        }, drawdown

    def generate_mega_report(self, config, gdelt_long_df, portfolio, bt_engine):
        print("\n=== GÉNÉRATION DU MÉGA RAPPORT D'ANALYSE ===")
        
        matrice_w = portfolio.get_matrix(normalize=False)
        matrice_w.to_csv(os.path.join(self.output_dir, "matrice_exposition.csv"))

        # ---------------------------------------------------------
        # VISUAL 1 : PROJECTION PCA (L'ESPACE LATENT)
        # ---------------------------------------------------------
        print(" -> Calcul de la projection spatiale des actifs (PCA)...")
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(matrice_w)
        
        # On trouve le concept dominant pour chaque action pour la couleur
        dominant_concepts = matrice_w.idxmax(axis=1)
        
        fig_pca = go.Figure()
        for concept in matrice_w.columns:
            mask = dominant_concepts == concept
            if mask.sum() > 0:
                fig_pca.add_trace(go.Scatter(
                    x=coords_2d[mask, 0], 
                    y=coords_2d[mask, 1],
                    mode='markers+text',
                    text=matrice_w.index[mask],
                    textposition="top center",
                    name=concept,
                    marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey'))
                ))

        fig_pca.update_layout(
            title="Carte d'Espace Latent : Clusters Sémantiques des Actifs",
            xaxis_title="Composante Principale 1", 
            yaxis_title="Composante Principale 2",
            height=700, 
            template="plotly_white",
            legend_title="Concept Dominant"
        )

        # ---------------------------------------------------------
        # VISUAL 2 : HEATMAP MATRICE W
        # ---------------------------------------------------------
        fig_matrix = go.Figure(data=go.Heatmap(
            z=matrice_w.values, x=matrice_w.columns, y=matrice_w.index, colorscale='Blues'
        ))
        fig_matrix.update_layout(height=800, title="Exposition Nette Actifs / Concepts (Matrice W)")

        # ---------------------------------------------------------
        # VISUAL 3 : PARAMETER SWEEP CORRIGÉ
        # ---------------------------------------------------------
        print(" -> Exécution du Parameter Sweep (Stratégies x Top_N x Métriques)...")
        
        metrics_to_test = ['Z20', 'EMA5'] if 'Z20' in gdelt_long_df.columns and 'EMA5' in gdelt_long_df.columns else [config['backtest']['signal_metric']]
        strategies = ['long_only', 'long_short']
        top_ns = [3, 5, 10]
        
        fig_sweep = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Étape 1 : On ajoute TOUTES les traces d'abord
        summary_table_rows = []
        metrics_data = [] # Stocke les titres pour les boutons
        
        trace_idx = 0
        for metric in metrics_to_test:
            daily_macro = gdelt_long_df.groupby(['Trading_Date', 'Concept'])[metric].sum().unstack(fill_value=0)
            asset_signals = portfolio.apply_signals(daily_macro)
            
            for strat in strategies:
                for n in top_ns:
                    combo_name = f"{metric} | {strat.upper()} | Top {n}"
                    
                    daily_ret, eq_curve = bt_engine.run_strategy(asset_signals, strategy=strat, top_n=n)
                    perf_metrics, drawdown = self.calculate_metrics(daily_ret, eq_curve)
                    
                    is_visible = (trace_idx == 0) # Seule la toute première est visible
                    
                    fig_sweep.add_trace(go.Scatter(x=eq_curve.index, y=eq_curve.values, name=f"Equity ({combo_name})", line=dict(color='#2980b9'), visible=is_visible), row=1, col=1)
                    fig_sweep.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name=f"DD ({combo_name})", fill='tozeroy', line=dict(color='#e74c3c'), visible=is_visible), row=2, col=1)
                    
                    metrics_data.append((combo_name, perf_metrics))
                    
                    summary_table_rows.append(f"""
                        <tr>
                            <td>{metric}</td><td>{strat.upper()}</td><td>{n}</td>
                            <td><b>{perf_metrics['Ratio de Sharpe']}</b></td>
                            <td>{perf_metrics['Rendement Cumulé']}</td>
                            <td>{perf_metrics['Max Drawdown']}</td>
                        </tr>
                    """)
                    trace_idx += 1

        # Étape 2 : On crée les boutons avec la bonne longueur de tableau de visibilité
        total_traces = len(fig_sweep.data)
        buttons = []
        
        for i, (combo_name, perf_metrics) in enumerate(metrics_data):
            # Tableau de booléens : tout est faux, sauf les 2 traces (Equity + DD) de l'itération 'i'
            visibility = [False] * total_traces
            visibility[i * 2] = True
            visibility[(i * 2) + 1] = True
            
            buttons.append(dict(
                label=combo_name,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Performance: {combo_name} | Sharpe: {perf_metrics['Ratio de Sharpe']} | Max DD: {perf_metrics['Max Drawdown']}"}
                ]
            ))

        # Étape 3 : Correction de la marge (margin t=120) pour que le menu ne cache pas le titre
        fig_sweep.update_layout(
            updatemenus=[dict(
                active=0, 
                buttons=buttons, 
                direction="down",
                x=0.01, 
                xanchor="left", 
                y=1.15, 
                yanchor="bottom"
            )],
            height=700, 
            template="plotly_white", 
            showlegend=False,
            margin=dict(t=120),
            title=f"Performance: {metrics_data[0][0]} | Sharpe: {metrics_data[0][1]['Ratio de Sharpe']} | Max DD: {metrics_data[0][1]['Max Drawdown']}"
        )

        # ---------------------------------------------------------
        # ASSEMBLAGE HTML FINAL
        # ---------------------------------------------------------
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>Mega Rapport Quantitatif</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f0f2f5; color: #333; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px 40px; }}
                .container {{ padding: 20px 40px; }}
                .card {{ background: white; padding: 20px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; text-align: left; }}
                th, td {{ padding: 12px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .memo {{ background: #e8f8f5; border-left: 5px solid #1abc9c; padding: 15px; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="header">
                <h1>Dashboard de Recherche Quantitative</h1>
                <p>Analyse de Sensibilité et Espace Latent</p>
            </div>
            <div class="container">
                
                <div class="card">
                    <h2>1. Simulateur de Stratégies (Parameter Sweep)</h2>
                    {fig_sweep.to_html(full_html=False, include_plotlyjs=False)}
                </div>

                <div class="card">
                    <h2>2. Tableau de Synthèse des Combinaisons</h2>
                    <table>
                        <tr><th>Métrique</th><th>Stratégie</th><th>Top N</th><th>Ratio de Sharpe</th><th>Rend. Cumulé</th><th>Max Drawdown</th></tr>
                        {"".join(summary_table_rows)}
                    </table>
                </div>

                <div class="card">
                    <h2>3. L'Espace Latent (Vision Interne de l'IA)</h2>
                    <p>La carte PCA ci-dessous réduit les 17 dimensions conceptuelles en 2 axes majeurs. Les entreprises de même couleur sont celles qui partagent le même concept dominant selon le Transformer. <b>Si NVDA et AMD sont côte à côte, le modèle a une excellente compréhension du secteur.</b></p>
                    {fig_pca.to_html(full_html=False, include_plotlyjs=False)}
                </div>
                
                <div class="card">
                    <h2>4. Matrice d'Exposition Brute</h2>
                    {fig_matrix.to_html(full_html=False, include_plotlyjs=False)}
                </div>

            </div>
        </body>
        </html>
        """

        report_path = os.path.join(self.output_dir, "Mega_Dashboard_Backtest.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ MÉGA RAPPORT GÉNÉRÉ AVEC SUCCÈS : {report_path}")