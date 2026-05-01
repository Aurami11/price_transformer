import json
import pandas as pd
import matplotlib.pyplot as plt

# Importation de tes 4 modules sur-mesure
from latent_signal_generator import LatentSpaceIndicator
from gdelt_historical_pipeline import GdeltBacktestPipeline
from auto_portfolio_allocator import AutoThematicPortfolio
from backtest_engine import BacktestEngine

def main():
    print("=== DÉMARRAGE DU SYSTÈME QUANTITATIF THÉMATIQUE ===\n")
    
    # ==========================================
    # ÉTAPE 0 : Chargement de la Configuration
    # ==========================================
    print("[0/5] Chargement du fichier config.json...")
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ==========================================
    # ÉTAPE 1 : Moteur NLP (Espace Latent)
    # ==========================================
    print("\n[1/5] Initialisation de l'IA Sémantique...")
    engine = LatentSpaceIndicator()
    
    # Injection dynamique des concepts depuis le JSON
    for concept, anchors in config['latent_space']['concepts'].items():
        engine.define_concept(concept, anchors)
        
    # Injection dynamique des sentiments
    for sentiment, anchors in config['latent_space']['sentiments'].items():
        engine.define_sentiment(sentiment, anchors)

    # ==========================================
    # ÉTAPE 2 : Ingestion GDELT & Rémanence
    # ==========================================
    print("\n[2/5] Traitement du dossier d'archives GDELT...")
    pipeline = GdeltBacktestPipeline(engine, target_countries=config['gdelt_pipeline']['target_countries'])

    # Appel de process_directory au lieu de process_parquet_file
    gdelt_long_df = pipeline.process_directory(
        directory_path=config['gdelt_pipeline']['data_directory'],
        file_pattern=config['gdelt_pipeline']['file_pattern'],
        batch_size=config['gdelt_pipeline']['batch_size'],
        _skip_processing=True # Utilise le cache existant sans traiter de nouveaux fichiers (mode "dry-run")
    )

    # --- FILET DE SÉCURITÉ ---
    if gdelt_long_df.empty:
        print("❌ ERREUR : Le DataFrame GDELT est vide. Vérifiez votre fichier cache.")
        return
    # -------------------------

    print(gdelt_long_df.head())
    
    # LA "COLLE" DATA : On passe du format Long au format Wide pour le portefeuille
    # On choisit la métrique configurée (ex: 'Z20' pour trader l'anomalie, ou 'EMA5' pour la tendance)
    metric = config['backtest']['signal_metric']
    print(f"Extraction de la métrique {metric} pour la génération du signal d'exécution...")
    
    # On somme les Z-Scores de tous les pays/sessions pour avoir la "Force Globale du Concept" du jour
    daily_concept_signals = gdelt_long_df.groupby(['Trading_Date', 'Concept'])[metric].sum().unstack(fill_value=0)

    # ==========================================
    # ÉTAPE 3 : Allocation de Portefeuille (La Matrice W)
    # ==========================================
    print("\n[3/5] Construction de la matrice d'allocation (Actifs x Concepts)...")
    tickers = config['universe']['tickers']
    concepts = list(config['latent_space']['concepts'].keys())
    
    portfolio = AutoThematicPortfolio(engine, tickers, concepts)
    portfolio.fetch_company_profiles()
    portfolio.build_correlation_matrix()
    
    # On multiplie les signaux Macro (GDELT) par les sensibilités Micro (Yahoo)
    asset_signals = portfolio.apply_signals(daily_concept_signals)

    # ==========================================
    # ÉTAPE 4 : Moteur de Backtest & Exécution
    # ==========================================
    print("\n[4/5] Téléchargement des prix et exécution financière...")
    bt_engine = BacktestEngine(
        tickers=tickers, 
        start_date=config['backtest']['start_date'], 
        end_date=config['backtest']['end_date']
    )
    
    daily_returns, equity_curve = bt_engine.run_strategy(
        asset_signals, 
        strategy=config['backtest']['strategy'], 
        top_n=config['backtest']['top_n']
    )

    # ==========================================
    # ÉTAPE 5 : Visualisation et Export des Résultats
    # ==========================================
    print("\n[5/5] Génération du rapport de performance et sauvegarde...")
    
    # --- EXPORTATION DES DONNÉES (CSV) ---
    # 1. Sauvegarde des performances financières
    results_df = pd.DataFrame({
        'Daily_Return': daily_returns,
        'Equity_Curve': equity_curve
    })
    results_df.to_csv("output_performance.csv")
    print(" -> Performances sauvegardées dans 'output_performance.csv'")
    
    # 2. Sauvegarde des signaux par actif (Idéal pour déboguer le comportement de l'IA)
    asset_signals.to_csv("output_asset_signals.csv")
    print(" -> Signaux des actifs sauvegardés dans 'output_asset_signals.csv'")
    
    # --- CRÉATION ET EXPORTATION DU GRAPHIQUE ---
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve.values, label=f"Stratégie {config['backtest']['strategy'].upper()}", color='blue')
    
    plt.title(f"Performance du Portefeuille Thématique ({config['backtest']['start_date']} - {config['backtest']['end_date']})")
    plt.ylabel("Valeur du Portefeuille (Base 1)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename_plot = f"output_equity_curve_{config['backtest']['strategy']}.png"
    plt.savefig(filename_plot)
    print(f" -> Graphique sauvegardé sous '{filename_plot}'")
    
    # Affichage final à l'écran
    plt.show()

if __name__ == "__main__":
    main()