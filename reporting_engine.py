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
        """Calcule les métriques quantitatives avec protection contre les divisions par zéro."""
        trading_days = 252
        cum_return = equity_curve.iloc[-1] - 1
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        ann_return = (1 + cum_return) ** (365.25 / days) - 1 if days > 0 else 0
        
        std_dev = daily_returns.std()
        ann_vol = std_dev * np.sqrt(trading_days) if std_dev != 0 else 0
        
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        return {
            "Rendement Cumulé": f"{cum_return*100:.2f}%",
            "Rendement Annuel": f"{ann_return*100:.2f}%",
            "Volatilité": f"{ann_vol*100:.2f}%",
            "Ratio de Sharpe": round(sharpe, 2),
            "Max Drawdown": f"{max_dd*100:.2f}%"
        }, drawdown

    def generate_mega_report(self, config, gdelt_long_df, portfolio, bt_engine):
        print("\n=== GÉNÉRATION DU MÉGA RAPPORT D'ANALYSE ===")
        
        matrice_w = portfolio.get_matrix(normalize=False)
        matrice_w.to_csv(os.path.join(self.output_dir, "matrice_exposition.csv"))

        # =========================================================
        # COMPOSANT 1 : PROJECTION PCA (ESPACE LATENT 2D)
        # =========================================================
        print(" -> Calcul de la projection spatiale des actifs (PCA)...")
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(matrice_w)
        
        # Sécurisation des données via un DataFrame temporaire pour éviter les crashs d'index
        df_pca = pd.DataFrame({
            'X': coords_2d[:, 0],
            'Y': coords_2d[:, 1],
            'Ticker': matrice_w.index,
            'Dominant_Concept': matrice_w.idxmax(axis=1).values
        })
        
        fig_pca = go.Figure()
        for concept in df_pca['Dominant_Concept'].unique():
            subset = df_pca[df_pca['Dominant_Concept'] == concept]
            fig_pca.add_trace(go.Scatter(
                x=subset['X'], y=subset['Y'],
                mode='markers+text', text=subset['Ticker'], textposition="top center",
                name=concept, marker=dict(size=12, line=dict(width=1, color='white'))
            ))

        fig_pca.update_layout(
            title="Projection de l'Espace Latent (Clustering Sémantique)",
            xaxis_title="Axe Principal 1", yaxis_title="Axe Principal 2",
            height=500, template="plotly_white", margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )

        # =========================================================
        # COMPOSANT 2 : HEATMAP MATRICE W
        # =========================================================
        fig_matrix = go.Figure(data=go.Heatmap(
            z=matrice_w.values, x=matrice_w.columns, y=matrice_w.index, colorscale='Blues'
        ))
        fig_matrix.update_layout(
            title="Matrice d'Exposition (W) Brute",
            height=500, margin=dict(t=50, b=20, l=20, r=20)
        )

        # =========================================================
        # COMPOSANT 3 : PARAMETER SWEEP (COURBES ET DD)
        # =========================================================
        print(" -> Exécution du Parameter Sweep (Stratégies x Top_N x Métriques)...")
        
        metrics_to_test = ['Z20', 'EMA5'] if 'Z20' in gdelt_long_df.columns and 'EMA5' in gdelt_long_df.columns else [config['backtest']['signal_metric']]
        strategies = ['long_only', 'long_short']
        top_ns = [3, 5, 10]
        
        fig_sweep = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        
        summary_data = []
        metrics_meta = []
        trace_idx = 0
        
        for metric in metrics_to_test:
            daily_macro = gdelt_long_df.groupby(['Trading_Date', 'Concept'])[metric].sum().unstack(fill_value=0)
            asset_signals = portfolio.apply_signals(daily_macro)
            
            for strat in strategies:
                for n in top_ns:
                    combo_name = f"{metric} | {strat.upper()} | Top {n}"
                    
                    daily_ret, eq_curve = bt_engine.run_strategy(asset_signals, strategy=strat, top_n=n)
                    perf_metrics, drawdown = self.calculate_metrics(daily_ret, eq_curve)
                    
                    # Seule la première trace est visible au chargement
                    is_visible = (trace_idx == 0) 
                    
                    fig_sweep.add_trace(go.Scatter(
                        x=eq_curve.index, y=eq_curve.values, name=f"Equity", 
                        line=dict(color='#2980b9', width=2), visible=is_visible
                    ), row=1, col=1)
                    
                    fig_sweep.add_trace(go.Scatter(
                        x=drawdown.index, y=drawdown.values, name=f"Drawdown", 
                        fill='tozeroy', line=dict(color='#e74c3c', width=1), visible=is_visible
                    ), row=2, col=1)
                    
                    metrics_meta.append((combo_name, perf_metrics))
                    
                    # Construction des données pour le tableau HTML
                    summary_data.append(f"""
                        <tr>
                            <td>{metric}</td>
                            <td>{strat.upper()}</td>
                            <td>{n}</td>
                            <td style="font-weight:bold; color:{'#27ae60' if float(perf_metrics['Ratio de Sharpe']) > 1 else '#c0392b'}">{perf_metrics['Ratio de Sharpe']}</td>
                            <td>{perf_metrics['Rendement Cumulé']}</td>
                            <td>{perf_metrics['Max Drawdown']}</td>
                        </tr>
                    """)
                    trace_idx += 1

        # Création des boutons du Dropdown
        total_traces = len(fig_sweep.data)
        buttons = []
        
        for i, (combo_name, perf_metrics) in enumerate(metrics_meta):
            visibility = [False] * total_traces
            visibility[i * 2] = True         # Active l'Equity
            visibility[(i * 2) + 1] = True   # Active le Drawdown
            
            buttons.append(dict(
                label=combo_name,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"<b>{combo_name}</b> | Sharpe: {perf_metrics['Ratio de Sharpe']} | Rendement: {perf_metrics['Rendement Cumulé']} | Max DD: {perf_metrics['Max Drawdown']}"}
                ]
            ))

        # Layout blindé avec marge supérieure (t=80) pour isoler le Dropdown du Titre
        fig_sweep.update_layout(
            updatemenus=[dict(
                active=0, buttons=buttons, direction="down",
                x=0.0, xanchor="left", y=1.15, yanchor="bottom",
                bgcolor="white", bordercolor="#bdc3c7", borderwidth=1
            )],
            height=600, template="plotly_white", showlegend=False,
            margin=dict(t=80, b=20, l=40, r=20),
            title=dict(
                text=f"<b>{metrics_meta[0][0]}</b> | Sharpe: {metrics_meta[0][1]['Ratio de Sharpe']} | Rendement: {metrics_meta[0][1]['Rendement Cumulé']} | Max DD: {metrics_meta[0][1]['Max Drawdown']}",
                y=0.95, x=0.0, xanchor='left', yanchor='top'
            )
        )

        # =========================================================
        # ASSEMBLAGE HTML / CSS FLEXBOX
        # =========================================================
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Quantitative Dashboard</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #ecf0f1; color: #2c3e50; }}
                h1 {{ margin-top: 0; color: #2980b9; }}
                
                /* Layout Flexbox Principal */
                .dashboard-container {{ display: flex; flex-direction: column; gap: 20px; }}
                .flex-row {{ display: flex; flex-direction: row; gap: 20px; align-items: stretch; }}
                
                /* Boîtes des composants */
                .box {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
                
                /* Largeurs spécifiques */
                .box-chart {{ flex: 7; }}
                .box-table {{ flex: 3; overflow-y: auto; max-height: 600px; }}
                .box-half {{ flex: 1; overflow: hidden; }}
                
                /* Style du tableau */
                table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
                th, td {{ padding: 10px; border-bottom: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #f8f9fa; position: sticky; top: 0; z-index: 10; }}
                tr:hover {{ background-color: #f1f2f6; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Dashboard de Recherche Quantitative GDELT</h1>
            
            <div class="dashboard-container">
                
                <div class="flex-row">
                    <div class="box box-chart">
                        <h2 style="margin-top:0; font-size:16px;">Analyse de Sensibilité (Parameter Sweep)</h2>
                        {fig_sweep.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                    
                    <div class="box box-table">
                        <h2 style="margin-top:0; font-size:16px;">Synthèse des Combinaisons</h2>
                        <table>
                            <tr><th>Signal</th><th>Stratégie</th><th>Top N</th><th>Sharpe</th><th>Rendement</th><th>Drawdown</th></tr>
                            {"".join(summary_data)}
                        </table>
                    </div>
                </div>

                <div class="flex-row">
                    <div class="box box-half">
                        <h2 style="margin-top:0; font-size:16px;">Compréhension Sémantique de l'IA (PCA)</h2>
                        {fig_pca.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                    
                    <div class="box box-half">
                        <h2 style="margin-top:0; font-size:16px;">Pondérations Nettes (Matrice W)</h2>
                        {fig_matrix.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                </div>
                
            </div>
        </body>
        </html>
        """

        report_path = os.path.join(self.output_dir, "Mega_Dashboard_Backtest.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ MÉGA RAPPORT GÉNÉRÉ AVEC SUCCÈS : {report_path}")