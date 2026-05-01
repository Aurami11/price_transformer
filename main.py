import json
import pandas as pd

# Importation de tes modules sur-mesure
from latent_signal_generator import LatentSpaceIndicator
from gdelt_historical_pipeline import GdeltBacktestPipeline
from auto_portfolio_allocator import AutoThematicPortfolio
from backtest_engine import BacktestEngine
from reporting_engine import QuantReportGenerator

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
    # On initialise le générateur de dashboard (il crée le dossier 'output' automatiquement)
    report_gen = QuantReportGenerator(output_dir="output")
    
    # On récupère la matrice W pour l'inclure dans le rapport (pour voir les "yeux" de l'IA)
    matrice_w_brute = portfolio.get_matrix(normalize=False)
    
    # Génération du HTML et des CSV
    report_gen.generate_html_report(
        config=config,
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        asset_signals=asset_signals,
        matrice_w=matrice_w_brute
    )

if __name__ == "__main__":
    main()