import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class QuantReportGenerator:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def calculate_metrics(self, daily_returns, equity_curve):
        """Calcule les métriques quantitatives standards."""
        trading_days = 252
        
        # Rendement cumulé
        cum_return = equity_curve.iloc[-1] - 1
        
        # Rendement annualisé
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        ann_return = (1 + cum_return) ** (365.25 / days) - 1 if days > 0 else 0
        
        # Volatilité annualisée
        ann_vol = daily_returns.std() * np.sqrt(trading_days)
        
        # Ratio de Sharpe (Assumant un taux sans risque de 0% pour simplifier)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        
        # Maximum Drawdown (Perte maximale)
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

    def generate_html_report(self, config, equity_curve, daily_returns, asset_signals, matrice_w):
        """Crée un tableau de bord HTML interactif."""
        print("\nGénération du rapport HTML interactif...")
        
        # 1. Sauvegarde des données brutes en CSV
        perf_df = pd.DataFrame({'Daily_Return': daily_returns, 'Equity_Curve': equity_curve})
        perf_df.to_csv(os.path.join(self.output_dir, "performance.csv"))
        asset_signals.to_csv(os.path.join(self.output_dir, "asset_signals.csv"))
        matrice_w.to_csv(os.path.join(self.output_dir, "matrice_exposition.csv"))

        # 2. Calcul des métriques
        metrics, drawdown = self.calculate_metrics(daily_returns, equity_curve)

        # 3. Création des figures Plotly
        # Fig 1: Equity Curve & Drawdown
        fig_perf = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                                 vertical_spacing=0.05, subplot_titles=("Évolution du Portefeuille (Base 1)", "Drawdown (Perte latente)"))
        
        fig_perf.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Portefeuille", line=dict(color='blue')), row=1, col=1)
        fig_perf.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown", fill='tozeroy', line=dict(color='red')), row=2, col=1)
        fig_perf.update_layout(height=600, showlegend=False, template="plotly_white")

        # Fig 2: Heatmap de la Matrice W (Vision interne du modèle sur les actifs)
        fig_matrix = go.Figure(data=go.Heatmap(
            z=matrice_w.values,
            x=matrice_w.columns,
            y=matrice_w.index,
            colorscale='Viridis'
        ))
        fig_matrix.update_layout(height=800, title="Projection des Actifs dans l'Espace Latent (Matrice W)",
                                 xaxis_title="Concepts Macro", yaxis_title="Tickers")

        # 4. Assemblage du fichier HTML avec le Mémo
        html_content = f"""
        <html>
        <head>
            <title>Rapport Quantitatif - {config['backtest']['strategy'].upper()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }}
                .container {{ background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                h1, h2 {{ color: #2c3e50; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
                .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; flex: 1; min-width: 150px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
                .memo {{ background: #e8f8f5; border-left: 5px solid #1abc9c; padding: 15px; margin-top: 30px; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="container">
                <h1>Rapport de Backtest : {config['backtest']['strategy'].upper()}</h1>
                <p>Période : {config['backtest']['start_date']} à {config['backtest']['end_date']} | Top N : {config['backtest']['top_n']} | Signal : {config['backtest']['signal_metric']}</p>
                
                <div class="metrics">
                    {"".join([f'<div class="metric-card"><div>{k}</div><div class="metric-value">{v}</div></div>' for k, v in metrics.items()])}
                </div>

                <h2>1. Performance Financière</h2>
                {fig_perf.to_html(full_html=False, include_plotlyjs=False)}
                
                <h2>2. Compréhension Sémantique (La Matrice W)</h2>
                <p>Cette carte thermique montre comment l'IA perçoit chaque entreprise. Plus la couleur est claire (jaune), plus l'entreprise est fortement corrélée au concept selon son profil Yahoo Finance.</p>
                {fig_matrix.to_html(full_html=False, include_plotlyjs=False)}

                <div class="memo">
                    <h3>💡 Mémo d'Analyse Quantitatives</h3>
                    <ul>
                        <li><b>Ratio de Sharpe :</b> Juge le rendement par rapport au risque pris. Supérieur à 1 = Bon. Supérieur à 2 = Excellent (potentiellement sur-optimisé).</li>
                        <li><b>Max Drawdown :</b> Mesure la plus forte chute du portefeuille depuis son plus haut historique. Un drawdown de -20% nécessite un gain de +25% juste pour revenir à zéro. Pour une stratégie Long/Short neutre, on vise un drawdown inférieur à -10%.</li>
                        <li><b>Matrice d'Exposition :</b> Vérifie les "faux positifs". Si l'IA assigne un fort score "Agriculture" à Microsoft, tes phrases d'ancrage dans le JSON manquent de précision financière.</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

        report_path = os.path.join(self.output_dir, f"report_{config['backtest']['strategy']}.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Fichiers bruts exportés dans le dossier '{self.output_dir}'")
        print(f"✅ Dashboard HTML généré avec succès : {report_path}")