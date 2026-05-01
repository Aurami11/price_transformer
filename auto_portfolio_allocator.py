import pandas as pd
import yfinance as yf
import json
import os
from sentence_transformers import util
from latent_signal_generator import LatentSpaceIndicator

class AutoThematicPortfolio:
    def __init__(self, indicator_engine, tickers, concepts, cache_file="company_profiles_cache.json"):
        self.engine = indicator_engine
        self.tickers = tickers
        self.concepts = concepts
        self.cache_file = cache_file
        
        # Chargement du cache au démarrage
        self.company_profiles = self._load_cache()
        
        # Matrice d'exposition (W)
        self.matrice_W = pd.DataFrame(0.0, index=self.tickers, columns=self.concepts)

    def _load_cache(self):
        """Charge les données stockées localement si le fichier existe."""
        if os.path.exists(self.cache_file):
            print(f"--- Chargement du cache depuis {self.cache_file} ---")
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Sauvegarde les profils récupérés dans le fichier JSON."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.company_profiles, f, indent=4, ensure_ascii=False)
        print(f"--- Cache mis à jour et sauvegardé dans {self.cache_file} ---")

    def fetch_company_profiles(self, force_refresh=False):
        """
        Récupère les descriptions. Vérifie d'abord le cache avant d'appeler l'API.
        """
        new_data_downloaded = False
        print(f"Vérification des profils pour {len(self.tickers)} tickers...")

        for ticker in self.tickers:
            # Si le ticker est déjà dans le cache et qu'on ne force pas le rafraîchissement
            if ticker in self.company_profiles and not force_refresh:
                continue
            
            # Sinon, on télécharge
            try:
                print(f"[{ticker}] Téléchargement depuis Yahoo Finance...")
                stock = yf.Ticker(ticker)
                summary = stock.info.get('longBusinessSummary', '')
                
                if summary:
                    self.company_profiles[ticker] = summary
                    new_data_downloaded = True
                else:
                    print(f"   ! Attention : Aucun résumé trouvé pour {ticker}")
                    # On stocke une valeur vide pour éviter de retélécharger inutilement
                    self.company_profiles[ticker] = "N/A"
                    
            except Exception as e:
                print(f"   ! Erreur pour {ticker} : {e}")

        if new_data_downloaded:
            self._save_cache()
        else:
            print("Tous les profils sont déjà en cache. Aucune requête API nécessaire.")

    def build_correlation_matrix(self, similarity_threshold=0.15):
        """
        Génère la matrice d'exposition via l'espace latent.
        """
        print("\nCalcul de l'alignement sémantique (Entreprise <-> Concepts)...")
        
        for ticker in self.tickers:
            summary = self.company_profiles.get(ticker, "")
            if summary == "" or summary == "N/A":
                continue

            # Projection du profil dans l'espace latent
            company_vector = self.engine.embedder.encode(summary, convert_to_tensor=True)
            
            for concept in self.concepts:
                centroid = self.engine.concept_centroids[concept]
                # Calcul de la similarité cosinus
                score = util.cos_sim(company_vector, centroid).item()
                
                # Application du seuil anti-bruit
                self.matrice_W.loc[ticker, concept] = score if score > similarity_threshold else 0.0

    def get_matrix(self, normalize=False):
      """
      Retourne la matrice d'exposition.
      :param normalize: Si True, normalise les colonnes pour que la somme = 1.
                        Si False (défaut), retourne les similarités cosinus brutes.
      """
      if not normalize:
         return self.matrice_W
         
      # Création d'une copie pour ne pas altérer la matrice brute stockée en mémoire
      matrice_norm = self.matrice_W.copy()
      
      # Normalisation
      sommes = matrice_norm.sum(axis=0)
      # On évite la division par zéro en remplaçant les sommes nulles par 1
      sommes = sommes.replace(0, 1) 
      
      matrice_norm = matrice_norm.div(sommes, axis=1)
      return matrice_norm
    
    def apply_signals(self, daily_concept_signals):
        """
        Multiplie les signaux macroéconomiques (GDELT) par la matrice d'exposition
        pour obtenir le signal directionnel final par action.
        """
        # On récupère la matrice d'exposition (brute par défaut)
        matrice_w = self.get_matrix(normalize=False)
        
        # On s'assure que les colonnes du signal GDELT correspondent exactement aux colonnes de notre matrice
        # daily_concept_signals a pour colonnes : les Concepts
        # matrice_w a pour index : les Tickers, et pour colonnes : les Concepts
        
        # Produit matriciel : (Dates x Concepts) * (Concepts x Actifs) = (Dates x Actifs)
        # .T transpose la matrice W pour l'alignement
        asset_signals = daily_concept_signals.dot(matrice_w.T)
        
        return asset_signals

# ==========================================
# EXEMPLE D'UTILISATION
# ==========================================
if __name__ == "__main__":
    # On réutilise le moteur de l'étape 1
    indicator = LatentSpaceIndicator()
    indicator.define_concept("Artificial_Intelligence", ["artificial intelligence", "deep learning", "GPU training"])
    indicator.define_concept("Renewable_Energy", ["solar panels", "wind turbines", "green energy transition"])

    # Liste d'actifs à tester
    tickers = ["NVDA", "MSFT", "FSLR", "ENPH", "AAPL"] 
    
    # Initialisation avec fichier de cache
    portfolio = AutoThematicPortfolio(indicator, tickers, ["Artificial_Intelligence", "Renewable_Energy"])
    
    # 1. Récupération (intelligente) des données
    portfolio.fetch_company_profiles()
    
    # 2. Construction de la logique d'investissement
    portfolio.build_correlation_matrix()
    
    print("\n--- MATRICE D'EXPOSITION (W) ---")
    print(portfolio.get_matrix().round(3))