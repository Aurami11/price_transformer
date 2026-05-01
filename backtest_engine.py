import pandas as pd
import numpy as np
import yfinance as yf
import os

class BacktestEngine:
    def __init__(self, tickers, start_date, end_date, cache_file="market_prices_cache.csv"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.cache_file = cache_file
        
        print(f"Initialisation du moteur de backtest...")
        self.prices = self._get_prices()
        
        # CALCUL DES RENDEMENTS (Décalés de 1 jour pour éviter le biais d'anticipation)
        # Si le signal est calculé le jour J à 16h, on trade le rendement du jour J+1
        self.returns = self.prices.pct_change().shift(-1)

    def _get_prices(self):
        """
        Récupère les prix. Utilise le cache local si valide, sinon télécharge via Yahoo.
        """
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        needs_download = True
        
        # 1. VÉRIFICATION DU CACHE
        if os.path.exists(self.cache_file):
            print(f"Lecture du cache local ({self.cache_file})...")
            # On charge le CSV
            cached_data = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
            
            # Vérification A : Tous les tickers sont-ils présents ?
            missing_tickers = [t for t in self.tickers if t not in cached_data.columns]
            
            # Vérification B : Les dates sont-elles couvertes ?
            dates_covered = (cached_data.index.min() <= start_dt) and (cached_data.index.max() >= end_dt)
            
            if missing_tickers:
                print(f"-> Tickers manquants dans le cache : {missing_tickers}")
            elif not dates_covered:
                print(f"-> La plage de dates demandée déborde du cache actuel.")
            else:
                print("-> Cache parfait ! Utilisation des données locales.")
                needs_download = False
                
                # On filtre le cache pour ne renvoyer que ce qui est demandé
                mask = (cached_data.index >= start_dt) & (cached_data.index <= end_dt)
                return cached_data.loc[mask, self.tickers]

        # 2. TÉLÉCHARGEMENT SI NÉCESSAIRE
        if needs_download:
            print("Téléchargement des données depuis Yahoo Finance...")
            raw_data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)
            
            # --- GESTION ROBUSTE DE L'API YFINANCE ---
            # On vérifie si 'Adj Close' existe, sinon on se rabat sur 'Close'
            if 'Adj Close' in raw_data.columns.levels[0] if isinstance(raw_data.columns, pd.MultiIndex) else 'Adj Close' in raw_data:
                data = raw_data['Adj Close']
            elif 'Close' in raw_data.columns.levels[0] if isinstance(raw_data.columns, pd.MultiIndex) else 'Close' in raw_data:
                data = raw_data['Close']
            else:
                print(f"❌ ERREUR STRUCTURE YAHOO. Colonnes disponibles : {raw_data.columns}")
                # On retourne un DataFrame vide pour ne pas crasher la suite
                return pd.DataFrame()
            # -----------------------------------------
            
            # Si un seul ticker est demandé, yfinance renvoie une Series. On la convertit en DataFrame.
            if isinstance(data, pd.Series):
                data = data.to_frame(name=self.tickers[0])
                
            # Remplissage des jours sans cotation (ex: action suspendue) par le prix précédent
            data = data.ffill()
            
            # 3. SAUVEGARDE ET FUSION
            if os.path.exists(self.cache_file):
                print("Fusion avec l'ancien cache et sauvegarde...")
                old_data = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
                # combine_first permet de fusionner deux DataFrames en gardant les index temporels
                combined_data = data.combine_first(old_data)
                combined_data.to_csv(self.cache_file)
            else:
                print("Création du nouveau fichier de cache...")
                data.to_csv(self.cache_file)
                
            return data

    def run_strategy(self, asset_signals_df, strategy='long_only', top_n=2):
        print(f"\n--- Exécution de la stratégie : {strategy.upper()} (Top {top_n}) ---")
        
        # ==========================================
        # 1. NETTOYAGE ET INTERSECTION DES COLONNES
        # ==========================================
        # On ne garde que les tickers présents à la fois dans nos signaux ET dans nos prix téléchargés
        valid_tickers = self.returns.columns.intersection(asset_signals_df.columns)
        
        if len(valid_tickers) == 0:
            print("❌ ERREUR : Aucun ticker commun entre les signaux et les prix Yahoo.")
            return pd.Series(dtype=float), pd.Series(dtype=float)
            
        print(f"Tickers valides pour l'exécution : {len(valid_tickers)} / {len(asset_signals_df.columns)}")
        
        # On filtre nos DataFrames pour s'aligner parfaitement
        signals_df = asset_signals_df[valid_tickers].copy()
        market_returns = self.returns[valid_tickers].copy()

        # ==========================================
        # 2. CORRECTION DES FUSEAUX HORAIRES
        # ==========================================
        if signals_df.index.tz is not None:
            signals_df.index = signals_df.index.tz_localize(None)
        if market_returns.index.tz is not None:
            market_returns.index = market_returns.index.tz_localize(None)
            
        signals_df.index = pd.to_datetime(signals_df.index).normalize()
        market_returns.index = pd.to_datetime(market_returns.index).normalize()
        
        # Alignement temporel final (Inner Join)
        signals, market_returns = signals_df.align(market_returns, join='inner', axis=0)
        
        if signals.empty:
            print("❌ ERREUR : L'alignement temporel a échoué (Dates incohérentes).")
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        # ==========================================
        # 3. EXÉCUTION DU PORTFEUILLE
        # ==========================================
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for date, row_signals in signals.iterrows():
            if row_signals.sum() == 0: continue
            
            # On trie uniquement les signaux non nuls
            ranked = row_signals[row_signals != 0].sort_values(ascending=False)
            
            # Sécurité : Si un jour on n'a pas assez d'actions avec des signaux
            current_top_n = min(top_n, len(ranked) // 2 if strategy == 'long_short' else len(ranked))
            if current_top_n == 0: continue

            if strategy == 'long_only':
                longs = ranked.head(current_top_n).index
                weights.loc[date, longs] = 1.0 / current_top_n
            elif strategy == 'long_short':
                longs = ranked.head(current_top_n).index
                weights.loc[date, longs] = 1.0 / current_top_n
                shorts = ranked.tail(current_top_n).index
                weights.loc[date, shorts] = -1.0 / current_top_n

        # Calcul du rendement
        portfolio_returns = (weights * market_returns).sum(axis=1)
        cumulative_return = (1 + portfolio_returns).cumprod()
        
        if portfolio_returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        else:
            sharpe_ratio = 0.0

        print(f"Rendement Total Cumulé : {(cumulative_return.iloc[-1] - 1) * 100:.2f}%")
        print(f"Ratio de Sharpe        : {sharpe_ratio:.2f}")

        return portfolio_returns, cumulative_return
    
# ==========================================
# EXEMPLE D'UTILISATION
# ==========================================
if __name__ == "__main__":
    tickers = ["NVDA", "MSFT", "AAPL", "DE", "CNHI"]
    
    # 1. Initialisation du moteur (Téléchargement de Yahoo Finance)
    engine = BacktestEngine(tickers, start_date="2022-01-01", end_date="2022-12-31")
    
    # 2. Création de signaux fictifs (Dans la réalité, c'est la sortie de ta Matrice W)
    # Imaginons que le signal IA soit très fort pour NVDA et MSFT
    dates = engine.returns.index
    fake_signals = pd.DataFrame(np.random.randn(len(dates), len(tickers)), index=dates, columns=tickers)
    
    # 3. Test de la stratégie Directionnelle (Achat des 2 meilleures)
    ret_long, eq_long = engine.run_strategy(fake_signals, strategy='long_only', top_n=2)
    
    # 4. Test de la stratégie Neutre (Achat des 2 meilleures, Short des 2 pires)
    ret_ls, eq_ls = engine.run_strategy(fake_signals, strategy='long_short', top_n=2)