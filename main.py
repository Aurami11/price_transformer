import json
import pandas as pd
import os

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
    
    # ==========================================
    # ÉTAPE 4 : Initialisation Moteur Backtest
    # ==========================================
    print("\n[4/5] Préparation du Moteur de Simulation Boursière...")
    bt_engine = BacktestEngine(
        tickers=tickers, 
        start_date=config['backtest']['start_date'], 
        end_date=config['backtest']['end_date']
    )

    # ==========================================
    # ÉTAPE 5 : Exportation pour le Dashboard Streamlit
    # ==========================================
    print("\n[5/5] Exportation des données pour l'interface interactive...")
    os.makedirs("dashboard_data", exist_ok=True)
    
    # 1. Sauvegarde de la Matrice W
    portfolio.get_matrix(normalize=False).to_csv("dashboard_data/matrice_w.csv")
    
    # 2. Sauvegarde des Signaux Macro (Z20 et EMA5)
    if 'Z20' in gdelt_long_df.columns:
        gdelt_long_df.groupby(['Trading_Date', 'Concept'])['Z20'].sum().unstack(fill_value=0).to_csv("dashboard_data/macro_z20.csv")
    if 'EMA5' in gdelt_long_df.columns:
        gdelt_long_df.groupby(['Trading_Date', 'Concept'])['EMA5'].sum().unstack(fill_value=0).to_csv("dashboard_data/macro_ema5.csv")
        
    # 3. Sauvegarde des rendements du marché (Yahoo) pour ne pas retélécharger
    bt_engine.returns.to_csv("dashboard_data/market_returns.csv")
    
    print("✅ Base de données prête ! Lancez la commande : streamlit run app.py")

if __name__ == "__main__":
    main()