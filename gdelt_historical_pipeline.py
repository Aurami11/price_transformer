import os
import glob
import pandas as pd
import numpy as np
import re
import urllib.parse
from pandas.tseries.offsets import BDay
import datetime
from latent_signal_generator import LatentSpaceIndicator

class GdeltBacktestPipeline:
    def __init__(self, indicator_engine, target_countries=None, cache_path="processed_gdelt_cache.parquet"):
        self.engine = indicator_engine
        self.target_countries = target_countries if target_countries else ['GLOBAL']
        self.cache_path = cache_path

    def clean_v2_string(self, v2_str):
        """
        Nettoie le format GDELT "THEME,position;THEME2,position2"
        Retourne une chaîne propre : "THEME THEME2"
        """
        if pd.isna(v2_str) or not str(v2_str).strip():
            return ""
        
        # On coupe par ';' puis on garde uniquement ce qui est avant la ','
        elements = [chunk.split(',')[0] for chunk in str(v2_str).split(';') if chunk]
        
        # On remplace les underscores par des espaces pour le Transformer
        clean_string = " ".join(elements).replace('_', ' ').lower()
        return clean_string

    def extract_tone_description(self, v2tone_str):
        """
        Convertit le score V2Tone de GDELT en une description textuelle
        pour guider le Transformer vers le bon hyperplan de sentiment.
        """
        if pd.isna(v2tone_str) or not str(v2tone_str).strip():
            return "neutral"
            
        try:
            # GDELT V2Tone: "Tone,Positive,Negative,Polarity,..."
            global_tone = float(str(v2tone_str).split(',')[0])
            
            # Conversion du score en adverbes/adjectifs
            if global_tone > 3.0: return "highly positive and optimistic"
            elif global_tone > 1.0: return "positive"
            elif global_tone < -3.0: return "highly negative, catastrophic and pessimistic"
            elif global_tone < -1.0: return "negative and pessimistic"
            else: return "neutral and factual"
        except:
            return "neutral"

   #  def extract_pseudo_text(self, row):
   #      """
   #      Tente d'extraire un texte lisible (Plan A) ou génère une synthèse (Plan B).
   #      """
   #      url = str(row.get('documentidentifier', ''))
        
   #      # ==========================================
   #      # PLAN A : Extraction depuis l'URL
   #      # ==========================================
   #      try:
   #          parsed_url = urllib.parse.urlparse(url)
   #          path = parsed_url.path.strip('/')
   #          if path:
   #              slug = path.split('/')[-1]
   #              clean_title = re.sub(r'\.(html|htm|php|aspx|cms)$', '', slug)
   #              clean_title = clean_title.replace('-', ' ').replace('_', ' ')
                
   #              if len(clean_title.split()) > 3:
   #                  return clean_title.capitalize()
   #      except:
   #          pass
            
   #      # ==========================================
   #      # PLAN B : Génération avec V2 et injection du Tone
   #      # ==========================================
   #      themes = self.clean_v2_string(row.get('v2themes', ''))
   #      orgs = self.clean_v2_string(row.get('v2organizations', ''))
   #      tone_desc = self.extract_tone_description(row.get('v2tones', ''))
        
   #      # On limite la taille pour ne pas noyer le modèle
   #      themes_trunc = " ".join(themes.split()[:30]) # On garde les 30 premiers thèmes max
   #      orgs_trunc = " ".join(orgs.split()[:10])     # On garde les 10 premières orgs max
        
   #      # La magie opère ici : on fabrique une phrase que le Transformer va comprendre !
   #      synthetic_text = f"News regarding {orgs_trunc}. Key themes: {themes_trunc}. The overall financial and economic outlook of this event is {tone_desc}."
        
   #      return synthetic_text

    def extract_countries(self, v2locations_str):
        """
        Extrait les codes pays de la colonne V2Locations de GDELT.
        Format GDELT: Type#FullName#CountryCode#RegionCode#...;
        """
        if pd.isna(v2locations_str) or not str(v2locations_str).strip():
            return ['GLOBAL']
            
        found_countries = set()
        blocks = str(v2locations_str).split(';')
        for block in blocks:
            parts = block.split('#')
            # Le code pays FIPS est généralement en 3ème position (index 2)
            if len(parts) >= 3 and len(parts[2]) == 2:
                found_countries.add(parts[2].upper())
                
        # On filtre pour ne garder que les pays cibles de notre stratégie
        if 'GLOBAL' not in self.target_countries:
            relevant_countries = list(found_countries.intersection(set(self.target_countries)))
            return relevant_countries if relevant_countries else [] # Retourne vide si aucun pays cible
            
        return list(found_countries) if found_countries else ['GLOBAL']

    def extract_pseudo_text(self, row):
        """
        Tente d'extraire un texte (Plan A) ou génère une synthèse (Plan B).
        Filtre intelligemment les UUIDs et les hashs dans les URLs.
        """
        url = str(row.get('documentidentifier', ''))
        
        # ==========================================
        # PLAN A : Extraction intelligente depuis l'URL (Slug)
        # ==========================================
        try:
            parsed_url = urllib.parse.urlparse(url)
            path = parsed_url.path.strip('/')
            
            if path:
                best_slug = ""
                max_words = 0
                
                # On teste chaque segment de l'URL séparé par un '/'
                for segment in path.split('/'):
                    # 1. On retire l'extension
                    seg = re.sub(r'\.(html|htm|php|aspx|cms|jsp|story)$', '', segment)
                    
                    # 2. On supprime les patterns de type UUID (ex: 947ae72a-ecae-11e7-8a6a-80acf0774e64)
                    seg = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', '', seg)
                    
                    # 3. On coupe par tirets et underscores
                    words = seg.replace('-', ' ').replace('_', ' ').split()
                    
                    # 4. On filtre les "mots" qui sont en fait des restes de hash (mélange de chiffres/lettres > 4 caractères)
                    # ex: '11e7', 'idukkbn1ep0mt' seront détruits. '2018', 'G7', '5G' seront gardés.
                    valid_words = [w for w in words if not (re.search(r'\d', w) and not w.isdigit() and len(w) > 3)]
                    
                    # 5. On garde le segment qui a la phrase la plus longue
                    if len(valid_words) > max_words:
                        max_words = len(valid_words)
                        best_slug = " ".join(valid_words)

                # Si on a trouvé un vrai titre de plus de 3 mots, on le valide
                if max_words > 3:
                    return best_slug.capitalize(), "Plan_A_Slug"
        except:
            pass
            
        # ==========================================
        # PLAN B : Génération synthétique (Fallback)
        # ==========================================
        themes = self.clean_v2_string(row.get('v2themes', ''))
        orgs = self.clean_v2_string(row.get('v2organizations', ''))
        tone_desc = self.extract_tone_description(row.get('v2tones', ''))
        
        themes_trunc = " ".join(themes.split()[:30])
        orgs_trunc = " ".join(orgs.split()[:10])
        
        synthetic_text = f"News regarding {orgs_trunc}. Key themes: {themes_trunc}. The overall financial and economic outlook of this event is {tone_desc}."
        
        return synthetic_text, "Plan_B_Synthesis"

    def _process_single_file(self, file_path):
        """Extraction avec tracking du texte et de la méthode."""
        df = pd.read_parquet(file_path)
        
        df['datetime'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M%S', errors='coerce')
        df = df.dropna(subset=['datetime'])
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        
        all_rows = []
        for index, row in df.iterrows():
            countries = self.extract_countries(row.get('v2locations', ''))
            if not countries: continue
            
            # Récupération du texte ET de la méthode
            texte, methode = self.extract_pseudo_text(row)
            analyse = self.engine.analyze_news(texte)
            
            for country in countries:
                for concept, val in analyse['directed_signals'].items():
                    all_rows.append({
                        'datetime': row['datetime'],
                        'Country': country,
                        'Concept': concept,
                        'Raw_Signal': val,
                        'Processed_Text': texte,      # Pour vérification visuelle
                        'Extraction_Method': methode, # Pour statistiques
                        'Source_File': os.path.basename(file_path)
                    })
                    
        return pd.DataFrame(all_rows)

    def process_directory(self, directory_path, file_pattern="*.slim.parquet", batch_size=200, _skip_processing=False):
        """
        Gère le traitement de masse avec reprise sur erreur (Checkpointing).
        """
        search_path = os.path.join(directory_path, file_pattern)
        all_files = sorted(glob.glob(search_path))
        
        if not all_files:
            raise FileNotFoundError(f"Aucun fichier trouvé dans {directory_path}")

        # --- LOGIQUE DE REPRISE (RESUME) ---
        processed_files = set()
        if os.path.exists(self.cache_path):
            print(f"--- [REPRISE] Lecture du cache existant pour identifier les fichiers déjà traités ---")
            temp_df = pd.read_parquet(self.cache_path, columns=['Source_File'])
            processed_files = set(temp_df['Source_File'].unique())

            del temp_df # Libère la RAM

            if _skip_processing:
                print("⚠️ Mode SKIP activé : Aucun nouveau fichier ne sera traité, le cache existant sera utilisé tel quel.")
                return self._apply_temporal_logic(pd.read_parquet(self.cache_path))
            
        
        files_to_process = [f for f in all_files if os.path.basename(f) not in processed_files]
        
        if not files_to_process:
            print("Tous les fichiers sont déjà présents dans le cache.")
            return self._apply_temporal_logic(pd.read_parquet(self.cache_path))

        print(f"Total : {len(all_files)} | Déjà faits : {len(processed_files)} | Restants : {len(files_to_process)}")
        
        current_batch = []
        for i, file_path in enumerate(files_to_process):
            try:
                file_df = self._process_single_file(file_path)
                if not file_df.empty:
                    current_batch.append(file_df)
            except Exception as e:
                print(f"Erreur sur {file_path}: {e}")

            # --- SAUVEGARDE PAR SEUIL (BATCH) ---
            if (i + 1) % batch_size == 0 or (i + 1) == len(files_to_process):
                print(f" -> [{i+1}/{len(files_to_process)}] Écriture du checkpoint sur disque...")
                
                new_data = pd.concat(current_batch, ignore_index=True)
                
                if os.path.exists(self.cache_path):
                    # On concatène l'existant avec le nouveau
                    existing_cache = pd.read_parquet(self.cache_path)
                    final_cache = pd.concat([existing_cache, new_data], ignore_index=True)
                    final_cache.to_parquet(self.cache_path, index=False)
                else:
                    new_data.to_parquet(self.cache_path, index=False)
                
                # RESET pour libérer la RAM
                current_batch = []
                del new_data

        print(f"Traitement terminé. Chargement du cache complet pour calculs temporels...")
        long_df = pd.read_parquet(self.cache_path)
        return self._apply_temporal_logic(long_df)
    
    def _apply_temporal_logic(self, long_df):
        """
        Applique la séparation des sessions de marché et calcule les métriques temporelles
        de manière vectorisée et sécurisée (sans utiliser .apply).
        """
        print("Application de la logique de session et de rémanence...")
        
        # Sécurité : Si le dataframe est vide, on arrête les frais immédiatement
        if long_df.empty:
            return long_df
            
        # 1. Assignation des sessions de marché
        times = long_df['datetime'].dt.time
        dates = long_df['datetime'].dt.normalize()
        market_open, market_close = datetime.time(9, 30), datetime.time(16, 0)
        
        is_post_market = times >= market_close
        long_df['Trading_Date'] = dates
        # Si la news sort après 16h, elle impacte le jour ouvré suivant
        long_df.loc[is_post_market, 'Trading_Date'] = dates[is_post_market] + BDay(1)
        # On repousse les news du week-end au lundi
        long_df['Trading_Date'] = long_df['Trading_Date'] + BDay(0)
        
        long_df['Session'] = 'Overnight'
        is_intraday = (times >= market_open) & (times < market_close)
        # Intraday = entre 9h30 et 16h, uniquement les jours de semaine (0 à 4)
        long_df.loc[is_intraday & (long_df['datetime'].dt.dayofweek < 5), 'Session'] = 'Intraday'
        
        # 2. Agrégation par Jour/Session/Pays/Concept
        print("Agrégation quotidienne des signaux...")
        final_long = long_df.groupby(['Trading_Date', 'Concept', 'Country', 'Session'])['Raw_Signal'].sum().reset_index()
        
        # 3. Calcul des métriques de rémanence (EMA, Z-Score) par Vectorisation
        print("Calcul des métriques temporelles (EMA5, Z20)...")
        
        # Tri obligatoire pour que les fenêtres glissantes soient dans le bon ordre chronologique
        final_long = final_long.sort_values(['Concept', 'Country', 'Session', 'Trading_Date'])
        
        # EMA
        final_long['EMA5'] = final_long.groupby(['Concept', 'Country', 'Session'])['Raw_Signal'].transform(
            lambda x: x.ewm(span=5, adjust=False).mean()
        )
        
        # Z-Score
        def calc_zscore(x):
            rolling = x.rolling(window=20, min_periods=5)
            return (x - rolling.mean()) / rolling.std()
            
        final_long['Z20'] = final_long.groupby(['Concept', 'Country', 'Session'])['Raw_Signal'].transform(calc_zscore)
        
        # On remplace les NaN générés par les divisions par zéro ou les fenêtres vides par 0
        return final_long.fillna(0)

# ==========================================
# EXEMPLE D'UTILISATION
# ==========================================
if __name__ == "__main__":
    indicator = LatentSpaceIndicator()
    indicator.define_concept("Tech", ["technology", "semiconductors", "artificial intelligence"])
    indicator.define_sentiment("Positive", ["massive growth and record profits"])
    indicator.define_sentiment("Negative", ["bankruptcy, crisis, and market crash"])
    
    # On cible uniquement les USA (US) et la Chine (CH)
    # Note : En FIPS, la Chine est CH, les US sont US, la France est FR, la Russie RS.
    pipeline = GdeltBacktestPipeline(indicator, target_countries=['US', 'CH'])
    
    # df_final = pipeline.process_parquet_file("gdelt_data.parquet")