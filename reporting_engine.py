import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class QuantReportGenerator:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    def calculate_metrics(self, daily_returns, equity_curve):
        trading_days = 252

        cum_return = equity_curve.iloc[-1] - 1
        days = (equity_curve.index[-1] - equity_curve.index[0]).days

        ann_return = (1 + cum_return) ** (365.25 / days) - 1 if days > 0 else 0
        ann_vol = daily_returns.std() * np.sqrt(trading_days)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0

        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max

        return {
            "Rendement Cumulé": f"{cum_return*100:.2f} %",
            "Rendement Annualisé": f"{ann_return*100:.2f} %",
            "Volatilité Annualisée": f"{ann_vol*100:.2f} %",
            "Ratio de Sharpe": f"{sharpe:.2f}",
            "Max Drawdown": f"{drawdown.min()*100:.2f} %"
        }, drawdown

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------
    def generate_mega_report(self, config, gdelt_long_df, portfolio, bt_engine):

        print("\n=== GÉNÉRATION DU MÉGA RAPPORT ===")

        matrice_w = portfolio.get_matrix(normalize=False)
        matrice_w.to_csv(os.path.join(self.output_dir, "matrice_exposition.csv"))

        # =========================================================
        # 1. PCA (ESPACE LATENT CORRIGÉ)
        # =========================================================
        print(" -> PCA espace latent...")

        scaler = StandardScaler()
        scaled_w = scaler.fit_transform(matrice_w)

        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(scaled_w)

        dominant_concepts = matrice_w.idxmax(axis=1).values

        fig_pca = go.Figure()

        for concept in matrice_w.columns:
            mask = (dominant_concepts == concept)

            if mask.sum() == 0:
                continue

            fig_pca.add_trace(go.Scatter(
                x=coords_2d[mask, 0],
                y=coords_2d[mask, 1],
                mode='markers+text',
                text=matrice_w.index[mask],
                textposition="top center",
                name=concept,
                marker=dict(size=12, opacity=0.85, line=dict(width=1, color='white'))
            ))

        fig_pca.update_layout(
            title="Espace latent (PCA) des actifs",
            xaxis_title="PC1",
            yaxis_title="PC2",
            height=700,
            template="plotly_white",
            legend_title="Concept dominant"
        )

        # =========================================================
        # 2. HEATMAP
        # =========================================================
        fig_matrix = go.Figure(data=go.Heatmap(
            z=matrice_w.values,
            x=matrice_w.columns,
            y=matrice_w.index,
            colorscale='Blues'
        ))

        fig_matrix.update_layout(
            height=800,
            title="Matrice d'exposition actifs / concepts"
        )

        # =========================================================
        # 3. PARAMETER SWEEP
        # =========================================================
        print(" -> Parameter sweep...")

        metrics_to_test = ['Z20', 'EMA5'] if 'Z20' in gdelt_long_df.columns else [config['backtest']['signal_metric']]
        strategies = ['long_only', 'long_short']
        top_ns = [3, 5, 10]

        fig_sweep = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05
        )

        metrics_data = []
        summary_rows = []
        trace_idx = 0

        for metric in metrics_to_test:

            daily_macro = gdelt_long_df.groupby(['Trading_Date', 'Concept'])[metric].sum().unstack(fill_value=0)
            asset_signals = portfolio.apply_signals(daily_macro)

            for strat in strategies:
                for n in top_ns:

                    name = f"{metric} | {strat.upper()} | Top {n}"

                    daily_ret, eq_curve = bt_engine.run_strategy(
                        asset_signals,
                        strategy=strat,
                        top_n=n
                    )

                    perf, dd = self.calculate_metrics(daily_ret, eq_curve)

                    visible = (trace_idx == 0)

                    fig_sweep.add_trace(
                        go.Scatter(
                            x=eq_curve.index,
                            y=eq_curve.values,
                            name=name,
                            visible=visible
                        ),
                        row=1, col=1
                    )

                    fig_sweep.add_trace(
                        go.Scatter(
                            x=dd.index,
                            y=dd.values,
                            fill='tozeroy',
                            name=name,
                            visible=visible
                        ),
                        row=2, col=1
                    )

                    metrics_data.append((name, perf))

                    summary_rows.append(f"""
                        <tr>
                            <td>{metric}</td>
                            <td>{strat}</td>
                            <td>{n}</td>
                            <td>{perf['Ratio de Sharpe']}</td>
                            <td>{perf['Rendement Cumulé']}</td>
                            <td>{perf['Max Drawdown']}</td>
                        </tr>
                    """)

                    trace_idx += 1

        # =========================================================
        # DROPDOWN FIXÉ
        # =========================================================
        total_traces = len(fig_sweep.data)

        buttons = []

        for i, (name, perf) in enumerate(metrics_data):

            visibility = [False] * total_traces
            visibility[i * 2] = True
            visibility[i * 2 + 1] = True

            buttons.append(dict(
                label=name,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title.text": f"{name} | Sharpe: {perf['Ratio de Sharpe']} | DD: {perf['Max Drawdown']}"}
                ]
            ))

        fig_sweep.update_layout(
            updatemenus=[dict(
                buttons=buttons,
                direction="down",
                x=0.01,
                y=1.15,
                showactive=True
            )],
            height=700,
            template="plotly_white",
            showlegend=False,
            margin=dict(t=120),
            title=metrics_data[0][0]
        )

        # =========================================================
        # HTML EXPORT SAFE
        # =========================================================
        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>Quant Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>

        <h1>Quant Dashboard</h1>

        <div>{fig_sweep.to_html(full_html=False, include_plotlyjs=False)}</div>

        <h2>Table</h2>
        <table border="1">
            <tr>
                <th>Metric</th><th>Strategy</th><th>Top N</th>
                <th>Sharpe</th><th>Return</th><th>DD</th>
            </tr>
            {"".join(summary_rows)}
        </table>

        <h2>PCA</h2>
        <div>{fig_pca.to_html(full_html=False, include_plotlyjs=False)}</div>

        <h2>Heatmap</h2>
        <div>{fig_matrix.to_html(full_html=False, include_plotlyjs=False)}</div>

        </body>
        </html>
        """

        path = os.path.join(self.output_dir, "report.html")

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"\n✅ Rapport généré : {path}")