import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import requests
import io
import gzip
import zipfile
import json
from typing import Dict, Any, Tuple
import logging
import csv
import re
from datetime import datetime
import geopandas as gpd
import fiona  # Ajouté pour gérer les options de lecture Shapefile
import os
import chardet
import tempfile
import shutil

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialisation de l'état de session
if 'datasets' not in st.session_state:
    st.session_state['datasets'] = {}
if 'explorations' not in st.session_state:
    st.session_state['explorations'] = {}
if 'sources' not in st.session_state:
    st.session_state['sources'] = {}
if 'json_data' not in st.session_state:
    st.session_state['json_data'] = {}

class DataLoader:
    def detect_delimiter(self, content: str) -> str:
        """Détecte le délimiteur d'un fichier CSV/TXT."""
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(content[:1024])
            return dialect.delimiter
        except:
            first_lines = content.splitlines()[:5]
            for line in first_lines:
                if ',' in line and ';' not in line:
                    return ','
                elif ';' in line:
                    return ';'
                elif '\t' in line:
                    return '\t'
            return ','

    def load(self, source: any, source_type: str, skip_header: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """Charge un fichier local ou distant en intégralité avec gestion des formats compressés et Shapefiles."""
        try:
            content_bytes = None
            file_ext = None
            content_type = "application/octet-stream"
            total_size = 0
            temp_dir = None

            # Récupération des données
            if source_type == "url":
                with requests.head(source, timeout=10) as head_response:
                    head_response.raise_for_status()
                    total_size = int(head_response.headers.get("Content-Length", 0))
                    content_type = head_response.headers.get("Content-Type", "application/octet-stream").lower()

                progress_bar = st.progress(0)
                status_text = st.empty()
                with st.spinner("Téléchargement en cours..."):
                    response = requests.get(source, timeout=15)
                    response.raise_for_status()
                    content_bytes = response.content

                downloaded_size = len(content_bytes)
                if total_size > 0:
                    progress = min(int((downloaded_size / total_size) * 100), 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Téléchargement terminé : {progress}% ({downloaded_size / 1024:.1f}/{total_size / 1024:.1f} Ko)")
                else:
                    status_text.text(f"Téléchargement terminé : {downloaded_size / 1024:.1f} Ko (taille totale inconnue)")

                file_ext = source.split('.')[-1].lower() if '.' in source.split('/')[-1] else None

            elif source_type == "file":
                if isinstance(source, list):  # Cas de fichiers multiples (Shapefile)
                    temp_dir = tempfile.mkdtemp()
                    shp_file = None
                    for uploaded_file in source:
                        file_name = uploaded_file.name
                        file_ext_temp = file_name.split('.')[-1].lower()
                        with open(os.path.join(temp_dir, file_name), 'wb') as f:
                            f.write(uploaded_file.read())
                        if file_ext_temp == 'shp':
                            shp_file = file_name
                    if not shp_file:
                        raise ValueError("Aucun fichier .shp trouvé parmi les fichiers téléversés.")
                    content_bytes = None  # Pas besoin de content_bytes pour Shapefile multiple
                    file_ext = 'shp'
                    total_size = sum(uploaded_file.size for uploaded_file in source)
                    status_text = st.empty()
                    status_text.text(f"Fichiers locaux chargés : {total_size / 1024:.1f} Ko")
                else:  # Cas d'un seul fichier
                    content_bytes = source.read()
                    if not content_bytes:
                        raise ValueError("Le fichier téléversé est vide ou illisible")
                    file_ext = source.name.split('.')[-1].lower() if '.' in source.name else None
                    total_size = len(content_bytes)
                    status_text = st.empty()
                    status_text.text(f"Fichier local chargé : {total_size / 1024:.1f} Ko")

            else:
                raise ValueError(f"Type de source '{source_type}' non supporté")

            # Détection du format si non défini
            if not file_ext or file_ext not in ['csv', 'txt', 'json', 'geojson', 'xlsx', 'gz', 'zip', 'shp']:
                if 'text/csv' in content_type:
                    file_ext = 'csv'
                elif 'application/json' in content_type:
                    file_ext = 'json'
                elif 'application/geo+json' in content_type:
                    file_ext = 'geojson'
                elif 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type:
                    file_ext = 'xlsx'
                elif content_bytes and content_bytes.startswith(b'\x1f\x8b'):
                    file_ext = 'gz'
                elif content_bytes and content_bytes.startswith(b'PK\x03\x04'):
                    file_ext = 'zip'
                elif content_bytes and content_bytes[:4] == b'x\x01\x01\x0b':
                    file_ext = 'shp'
                else:
                    detected = chardet.detect(content_bytes[:1024]) if content_bytes else {'confidence': 0}
                    if detected['confidence'] > 0.7 and detected['encoding']:
                        file_ext = 'csv'
                    else:
                        raise ValueError("Format non détectable ou contenu non textuel")

            # Gestion des fichiers compressés et Shapefiles
            processed_content = content_bytes

            if file_ext == 'gz':
                if content_bytes.startswith(b'\x1f\x8b'):
                    with st.spinner("Décompression GZIP..."):
                        processed_content = gzip.decompress(content_bytes)
                    file_ext = (source.split('.')[-2].lower() if source_type == "url" else source.name.split('.')[-2].lower()) if '.' in source else 'csv'
                    st.info(f"Fichier GZIP décompressé, format interne détecté : {file_ext.upper()}")
                else:
                    st.warning("Fichier marqué comme GZIP mais non compressé. Chargement direct.")
                    file_ext = 'csv'

            elif file_ext == 'zip':
                with st.spinner("Décompression ZIP..."):
                    temp_dir = tempfile.mkdtemp()
                    with zipfile.ZipFile(io.BytesIO(content_bytes)) as z:
                        z.extractall(temp_dir)
                        compatible_files = [f for f in z.namelist() if f.split('.')[-1].lower() in ['csv', 'txt', 'json', 'geojson', 'xlsx', 'shp']]
                        if not compatible_files:
                            raise ValueError("Aucun fichier compatible (CSV, TXT, JSON, GeoJSON, XLSX, SHP) trouvé dans le ZIP.")
                        file_name = compatible_files[0]
                        processed_content = open(os.path.join(temp_dir, file_name), 'rb').read()
                        file_ext = file_name.split('.')[-1].lower()
                        st.info(f"Fichier ZIP décompressé, fichier traité : {file_name} (format : {file_ext.upper()})")

            # Gestion des Shapefiles (.shp)
            if file_ext == 'shp':
                with st.spinner("Chargement du Shapefile..."):
                    if not temp_dir:  # Cas d'un .shp seul (URL ou fichier unique)
                        temp_dir = tempfile.mkdtemp()
                        shp_path = os.path.join(temp_dir, 'data.shp')
                        with open(shp_path, 'wb') as f:
                            f.write(processed_content)
                        if source_type == "url":
                            base_url = source.rsplit('.', 1)[0]
                            for ext in ['shx', 'dbf']:
                                try:
                                    assoc_url = f"{base_url}.{ext}"
                                    response = requests.get(assoc_url, timeout=15)
                                    response.raise_for_status()
                                    with open(os.path.join(temp_dir, f'data.{ext}'), 'wb') as f:
                                        f.write(response.content)
                                except requests.RequestException:
                                    st.warning(f"Fichier associé .{ext} non trouvé à distance.")
                    else:  # Cas de fichiers multiples ou ZIP
                        shp_path = os.path.join(temp_dir, 'data.shp' if source_type == "url" else shp_file)
                    
                    # Utiliser fiona.Env pour activer SHAPE_RESTORE_SHX
                    try:
                        with fiona.Env(SHAPE_RESTORE_SHX='YES'):
                            gdf = gpd.read_file(shp_path)
                        if gdf.empty:
                            raise ValueError("Le fichier Shapefile est vide ou invalide.")
                        st.info(f"Shapefile chargé avec {len(gdf)} entités.")
                        df = gdf.drop(columns=['geometry']).assign(
                            latitude=gdf.geometry.centroid.y,
                            longitude=gdf.geometry.centroid.x
                        )
                        return df, None
                    finally:
                        if temp_dir and os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)

            # Détection de l'encodage pour les fichiers textuels
            content_str = None
            if file_ext in ['csv', 'txt', 'json', 'geojson']:
                try:
                    detected = chardet.detect(processed_content)
                    encoding = detected['encoding'] if detected['confidence'] > 0.7 else 'utf-8'
                    st.info(f"Encodage détecté : {encoding} (confiance : {detected['confidence']:.2%})")
                    content_str = processed_content.decode(encoding)
                except UnicodeDecodeError:
                    st.warning("Échec du décodage avec l'encodage détecté. Tentative avec latin-1.")
                    content_str = processed_content.decode('latin-1', errors='replace')

            # Traitement selon le format
            st.info(f"Chargement intégral du contenu au format : {file_ext.upper()}")
            if file_ext == 'csv':
                delimiter = self.detect_delimiter(content_str)
                skiprows = 1 if skip_header else 0
                df = pd.read_csv(io.StringIO(content_str), delimiter=delimiter, skiprows=skiprows)
                return df, None
            elif file_ext == 'xlsx':
                df = pd.read_excel(io.BytesIO(processed_content))
                return df, None
            elif file_ext == 'txt':
                delimiter = self.detect_delimiter(content_str)
                skiprows = 1 if skip_header else 0
                df = pd.read_csv(io.StringIO(content_str), delimiter=delimiter, skiprows=skiprows)
                return df, None
            elif file_ext in ['json', 'geojson']:
                json_data = json.loads(content_str)
                if file_ext == 'geojson' and "type" in json_data and json_data["type"] in ["FeatureCollection", "Feature"]:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp_file:
                        tmp_file.write(processed_content)
                        tmp_file.flush()
                        gdf = gpd.read_file(tmp_file.name)
                    os.unlink(tmp_file.name)
                    if gdf.empty:
                        raise ValueError("Le fichier GeoJSON est vide ou invalide.")
                    return gdf.drop(columns=['geometry']).assign(
                        latitude=gdf.geometry.centroid.y,
                        longitude=gdf.geometry.centroid.x
                    ), None
                elif isinstance(json_data, list):
                    return pd.json_normalize(json_data), None
                return None, json_data
            else:
                raise ValueError(f"Format non supporté : {file_ext}")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement : {str(e)}")
            logger.error(f"Erreur détaillée : {str(e)}", exc_info=True)
            return None, None
        finally:
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
            if 'temp_dir' in locals() and temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

class DataExplorer:
    @staticmethod
    def explore(df: pd.DataFrame) -> Dict[str, Any]:
        exploration = {
            "metadata": df.dtypes.to_dict(),
            "duplicates": df.duplicated().sum(),
            "missing_values": df.isnull().sum().to_dict(),
            "description": df.describe(),
            "outliers": {}
        }
        for col in df.select_dtypes(include=np.number).columns:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
            exploration["outliers"][col] = len(outliers)
        return exploration

def correlation(df: pd.DataFrame, var1: str, var2: str) -> Dict[str, float]:
    results = {}
    if df is None or var1 not in df.columns or var2 not in df.columns:
        st.error("Colonnes invalides ou données absentes pour la corrélation.")
        return results
    try:
        df_clean = df[[var1, var2]].dropna()
        if df[var1].dtype in [np.float64, np.int64] and df[var2].dtype in [np.float64, np.int64]:
            results["pearson"] = stats.pearsonr(df_clean[var1], df_clean[var2])[0]
            results["spearman"] = stats.spearmanr(df_clean[var1], df_clean[var2])[0]
        else:
            st.warning(f"Les colonnes {var1} ou {var2} ne sont pas numériques.")
    except Exception as e:
        st.error(f"Erreur lors du calcul de la corrélation : {str(e)}")
    return results

def multivariate_tests(df: pd.DataFrame, group_var: str, value_var: str) -> Dict[str, Any]:
    results = {}
    if df is None or group_var not in df.columns or value_var not in df.columns:
        st.error("Colonnes invalides ou données absentes pour les tests multivariables.")
        return results
    try:
        if df[group_var].dtype not in [object, "category"] or df[value_var].dtype not in [np.float64, np.int64]:
            st.warning(f"{group_var} doit être qualitatif et {value_var} quantitatif.")
            return results
        groups = [group[1][value_var].dropna() for group in df.groupby(group_var)]
        if len(groups) > 1:
            results["anova"] = stats.f_oneway(*groups)
            results["kruskal"] = stats.kruskal(*groups)
            ss_total = df[value_var].var() * (len(df[value_var]) - 1)
            ss_between = sum(len(g) * (g.mean() - df[value_var].mean())**2 for g in groups)
            results["eta_squared"] = ss_between / ss_total
            results["kurtosis"] = df[value_var].kurtosis()
            results["skewness"] = df[value_var].skew()
        else:
            st.warning(f"Pas assez de groupes dans {group_var} pour les tests.")
    except Exception as e:
        st.error(f"Erreur lors des tests multivariables : {str(e)}")
    return results

class DataPipeline:
    def __init__(self):
        self.loader = DataLoader()
        self.explorer = DataExplorer()
    
    def process(self, source: str, source_type: str, name: str, skip_header: bool = False) -> Tuple[pd.DataFrame, Dict, Dict]:
        with st.spinner(f"Chargement intégral du dataset '{name}'..."):
            data, json_data = self.loader.load(source, source_type, skip_header)
            if data is not None:
                exploration = self.explorer.explore(data)
                return data, exploration, None
            return None, None, json_data

# Interface Streamlit
def main():
    st.title("🔍 Exploration et Analyse en Quelques Clics")
    pipeline = DataPipeline()
    
    # Sidebar pour le chargement des datasets
    with st.sidebar:
        st.header("📥 Chargement des datasets")
        st.info("Ajoutez des datasets (CSV, Excel, JSON, GeoJSON, TXT, SHP). Formats .gz et .zip supportés. Pour Shapefiles locaux, téléversez .shp, .shx, .dbf ensemble.")
        
        if 'num_datasets' not in st.session_state:
            st.session_state['num_datasets'] = 1
        
        num_datasets = st.number_input("Nombre de datasets", min_value=1, value=st.session_state['num_datasets'], step=1, key="num_datasets_input")
        st.session_state['num_datasets'] = num_datasets
        
        for i in range(num_datasets):
            st.subheader(f"Dataset {i+1}")
            source_type = st.selectbox(f"Type {i+1}", ["URL", "Fichier local"], key=f"source_type_{i}")
            skip_header = st.checkbox(f"Ignorer la première ligne {i+1} (CSV/TXT)", value=False, key=f"skip_header_{i}")
            
            if source_type == "URL":
                source = st.text_input(f"URL {i+1}", "", key=f"source_url_{i}")
                name = st.text_input(f"Nom {i+1}", f"dataset_{i+1}", key=f"name_{i}")
            else:
                uploaded_files = st.file_uploader(f"Fichier {i+1}", type=["csv", "xlsx", "json", "geojson", "txt", "gz", "zip", "shp", "shx", "dbf"], 
                                                accept_multiple_files=True, key=f"upload_{i}")
                if uploaded_files:
                    if len(uploaded_files) == 1:
                        source = uploaded_files[0]
                        name = st.text_input(f"Nom {i+1}", uploaded_files[0].name.split('.')[0], key=f"name_{i}")
                    else:
                        source = uploaded_files  # Liste de fichiers pour Shapefile
                        shp_file = next((f for f in uploaded_files if f.name.endswith('.shp')), None)
                        name = st.text_input(f"Nom {i+1}", shp_file.name.split('.')[0] if shp_file else "dataset", key=f"name_{i}")
                else:
                    source = None
                    name = None
            
            if st.button(f"Charger {i+1}", key=f"load_{i}") and source and name:
                data, exploration, json_data = pipeline.process(source, "url" if source_type == "URL" else "file", name, skip_header)
                if data is not None:
                    st.session_state['datasets'][name] = data
                    st.session_state['explorations'][name] = exploration
                    st.session_state['json_data'][name] = None
                    st.session_state['sources'][name] = (source, source_type, skip_header)
                    st.success(f"✅ Dataset '{name}' chargé intégralement avec succès ({len(data)} lignes) !")
                else:
                    st.session_state['json_data'][name] = json_data
                    if json_data:
                        st.session_state['json_keys'] = list(json_data.keys())
                        st.session_state['sources'][name] = (source, source_type, skip_header)
                        st.info(f"📋 JSON '{name}' chargé. Sélectionnez une clé.")

            if name in st.session_state['json_data'] and st.session_state['json_data'][name] is not None:
                selected_key = st.selectbox(f"Clé pour {name}", st.session_state['json_keys'], key=f"json_key_select_{i}")
                if st.button(f"Traiter la clé", key=f"process_json_key_{i}"):
                    with st.spinner(f"Traitement de la clé '{selected_key}'..."):
                        json_data = st.session_state['json_data'][name]
                        if selected_key in json_data:
                            data = pd.json_normalize(json_data[selected_key])
                            st.session_state['datasets'][name] = data
                            st.session_state['explorations'][name] = pipeline.explorer.explore(data)
                            st.session_state['json_data'][name] = None
                            st.success(f"✅ Données extraites de la clé '{selected_key}' !")
                        else:
                            st.error(f"❌ La clé '{selected_key}' n'existe pas.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Actualiser", key="refresh"):
                with st.spinner("Actualisation des datasets..."):
                    for name, (source, source_type, skip_header) in st.session_state['sources'].items():
                        if source_type == "url":
                            data, exploration, json_data = pipeline.process(source, source_type, name, skip_header)
                        else:
                            data, exploration, json_data = pipeline.process(source, "file", name, skip_header)
                        if data is not None:
                            st.session_state['datasets'][name] = data
                            st.session_state['explorations'][name] = exploration
                            st.session_state['json_data'][name] = None
                        else:
                            st.session_state['json_data'][name] = json_data
                    st.success("✅ Datasets actualisés !")
        with col2:
            if st.button("🗑️ Réinitialiser", key="reset"):
                with st.spinner("Réinitialisation en cours..."):
                    st.session_state.clear()
                    st.session_state['datasets'] = {}
                    st.session_state['explorations'] = {}
                    st.session_state['sources'] = {}
                    st.session_state['json_data'] = {}
                    st.session_state['num_datasets'] = 1
                    st.rerun()

    # Pré-traitement des données
    if st.session_state['datasets']:
        st.subheader("⚙️ Pré-traitement des données")
        st.info("Sélectionnez les datasets à traiter. Les modifications sont appliquées en temps réel.")
        
        dataset_names = list(st.session_state['datasets'].keys())
        selected_datasets = st.multiselect("Datasets", dataset_names, default=dataset_names[0] if dataset_names else None, key="select_datasets")
        
        if selected_datasets:
            with st.expander("🔄 Convertir les types"):
                col_to_convert = st.multiselect("Colonnes", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="convert_cols")
                type_to_convert = st.selectbox("Type", 
                                              ["Entier (int)", "Décimal (float)", "Catégorie (category)", 
                                               "Date (datetime)", "Timestamp vers Date", "Extraire Mois", 
                                               "Extraire Jour de la semaine", "Extraire Jour du mois", "Extraire Heure", "Extraire Année"], 
                                              key="convert_type")
                if st.button("Appliquer", key="apply_convert"):
                    with st.spinner("Conversion en cours..."):
                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            for col in col_to_convert:
                                if col in data.columns:
                                    try:
                                        if type_to_convert == "Entier (int)":
                                            data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
                                        elif type_to_convert == "Décimal (float)":
                                            data[col] = pd.to_numeric(data[col], errors='coerce')
                                        elif type_to_convert == "Catégorie (category)":
                                            data[col] = data[col].astype('category')
                                        elif type_to_convert == "Date (datetime)":
                                            data[col] = pd.to_datetime(data[col], errors='coerce')
                                        elif type_to_convert == "Timestamp vers Date":
                                            data[col] = pd.to_datetime(data[col], unit='s', errors='coerce')
                                        elif type_to_convert == "Extraire Mois":
                                            data[col] = pd.to_datetime(data[col], errors='coerce').dt.month
                                        elif type_to_convert == "Extraire Jour de la semaine":
                                            data[col] = pd.to_datetime(data[col], errors='coerce').dt.day_name()
                                        elif type_to_convert == "Extraire Jour du mois":
                                            data[col] = pd.to_datetime(data[col], errors='coerce').dt.day
                                        elif type_to_convert == "Extraire Heure":
                                            data[col] = pd.to_datetime(data[col], errors='coerce').dt.hour
                                        elif type_to_convert == "Extraire Année":
                                            data[col] = pd.to_datetime(data[col], errors='coerce').dt.year
                                        st.success(f"✅ '{col}' converti en {type_to_convert} pour '{ds}'")
                                    except Exception as e:
                                        st.error(f"❌ Erreur lors de la conversion de '{col}' dans '{ds}' : {str(e)}")
                            st.session_state['datasets'][ds] = data
                            st.session_state['explorations'][ds] = pipeline.explorer.explore(data)

            with st.expander("🧹 Nettoyer les valeurs"):
                col_to_clean = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="clean_col")
                action = st.selectbox("Action", ["Supprimer des caractères", "Remplacer des caractères"], key="clean_action")
                pattern = st.text_input("Motif (regex)", key="clean_pattern")
                replacement = st.text_input("Remplacement", "", key="clean_replace") if action == "Remplacer des caractères" else ""
                if st.button("Appliquer", key="apply_clean"):
                    with st.spinner("Nettoyage en cours..."):
                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            if col_to_clean in data.columns:
                                try:
                                    if action == "Supprimer des caractères":
                                        data[col_to_clean] = data[col_to_clean].apply(lambda x: re.sub(pattern, '', str(x)) if pd.notnull(x) else x)
                                        st.success(f"✅ '{pattern}' supprimé dans '{col_to_clean}' pour '{ds}'")
                                    else:
                                        data[col_to_clean] = data[col_to_clean].apply(lambda x: re.sub(pattern, replacement, str(x)) if pd.notnull(x) else x)
                                        st.success(f"✅ '{pattern}' remplacé par '{replacement}' dans '{col_to_clean}' pour '{ds}'")
                                    st.session_state['datasets'][ds] = data
                                    st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                except Exception as e:
                                    st.error(f"❌ Erreur lors du nettoyage de '{col_to_clean}' dans '{ds}' : {str(e)}")

            with st.expander("🗑️ Supprimer des colonnes"):
                cols_to_drop = st.multiselect("Colonnes", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="drop_cols")
                if st.button("Supprimer", key="apply_drop"):
                    with st.spinner("Suppression des colonnes en cours..."):
                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            cols_present = [col for col in cols_to_drop if col in data.columns]
                            if cols_present:
                                try:
                                    data = data.drop(columns=cols_present)
                                    st.session_state['datasets'][ds] = data
                                    st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                    st.success(f"✅ Colonnes {cols_present} supprimées pour '{ds}'")
                                except Exception as e:
                                    st.error(f"❌ Erreur lors de la suppression des colonnes dans '{ds}' : {str(e)}")

            with st.expander("🚫 Supprimer des lignes"):
                col_to_filter = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="filter_col")
                filter_expr = st.text_input("Expression (ex. 'Station_1' ou regex)", key="filter_expr")
                filter_type = st.selectbox("Type", ["Valeur exacte", "Regex"], key="filter_type")
                if st.button("Supprimer", key="apply_filter"):
                    with st.spinner("Suppression des lignes en cours..."):
                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            if col_to_filter in data.columns:
                                try:
                                    if filter_type == "Valeur exacte":
                                        data = data[data[col_to_filter] != filter_expr]
                                    else:
                                        data = data[~data[col_to_filter].apply(lambda x: bool(re.search(filter_expr, str(x)) if pd.notnull(x) else False))]
                                    st.session_state['datasets'][ds] = data
                                    st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                    st.success(f"✅ Lignes contenant '{filter_expr}' dans '{col_to_filter}' supprimées pour '{ds}'")
                                except Exception as e:
                                    st.error(f"❌ Erreur lors de la suppression des lignes dans '{ds}' : {str(e)}")

            with st.expander("➕ Créer une colonne"):
                new_col_name = st.text_input("Nom", key="new_col_name")
                base_col = st.selectbox("Base", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="base_col")
                new_col_action = st.selectbox("Action", ["Copie avec conversion", "Copie avec nettoyage"], key="new_col_action")
                if new_col_action == "Copie avec conversion":
                    new_col_type = st.selectbox("Type", 
                                               ["Entier (int)", "Décimal (float)", "Catégorie (category)", 
                                                "Date (datetime)", "Timestamp vers Date", "Extraire Mois", 
                                                "Extraire Jour de la semaine", "Extraire Jour du mois", "Extraire Heure", "Extraire Année"], 
                                               key="new_col_type")
                else:
                    new_col_pattern = st.text_input("Motif", key="new_col_pattern")
                    new_col_replace = st.text_input("Remplacement", "", key="new_col_replace")
                if st.button("Créer", key="apply_new_col"):
                    with st.spinner("Création de la colonne en cours..."):
                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            if base_col in data.columns:
                                try:
                                    if new_col_action == "Copie avec conversion":
                                        if new_col_type == "Entier (int)":
                                            data[new_col_name] = pd.to_numeric(data[base_col], errors='coerce').astype('Int64')
                                        elif new_col_type == "Décimal (float)":
                                            data[new_col_name] = pd.to_numeric(data[base_col], errors='coerce')
                                        elif new_col_type == "Catégorie (category)":
                                            data[new_col_name] = data[base_col].astype('category')
                                        elif new_col_type == "Date (datetime)":
                                            data[new_col_name] = pd.to_datetime(data[base_col], errors='coerce')
                                        elif new_col_type == "Timestamp vers Date":
                                            data[new_col_name] = pd.to_datetime(data[base_col], unit='s', errors='coerce')
                                        elif new_col_type == "Extraire Mois":
                                            data[new_col_name] = pd.to_datetime(data[base_col], errors='coerce').dt.month
                                        elif new_col_type == "Extraire Jour de la semaine":
                                            data[new_col_name] = pd.to_datetime(data[base_col], errors='coerce').dt.day_name()
                                        elif new_col_type == "Extraire Jour du mois":
                                            data[new_col_name] = pd.to_datetime(data[base_col], errors='coerce').dt.day
                                        elif new_col_type == "Extraire Heure":
                                            data[new_col_name] = pd.to_datetime(data[base_col], errors='coerce').dt.hour
                                        elif new_col_type == "Extraire Année":
                                            data[new_col_name] = pd.to_datetime(data[base_col], errors='coerce').dt.year
                                        st.success(f"✅ '{new_col_name}' créé avec {new_col_type} pour '{ds}'")
                                    else:
                                        if new_col_replace:
                                            data[new_col_name] = data[base_col].apply(lambda x: re.sub(new_col_pattern, new_col_replace, str(x)) if pd.notnull(x) else x)
                                            st.success(f"✅ '{new_col_name}' créé avec remplacement pour '{ds}'")
                                        else:
                                            data[new_col_name] = data[base_col].apply(lambda x: re.sub(new_col_pattern, '', str(x)) if pd.notnull(x) else x)
                                            st.success(f"✅ '{new_col_name}' créé avec suppression pour '{ds}'")
                                    st.session_state['datasets'][ds] = data
                                    st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                except Exception as e:
                                    st.error(f"❌ Erreur lors de la création de '{new_col_name}' dans '{ds}' : {str(e)}")

            with st.expander("🕳️ Traiter les valeurs manquantes"):
                col_to_fill = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="fill_col")
                fill_method = st.selectbox("Méthode", ["Supprimer les lignes", "Remplacer par moyenne (numérique)", "Remplacer par mode (catégorique)", "Plus proche voisin"], key="fill_method")
                if st.button("Traiter", key="apply_fill"):
                    with st.spinner("Traitement des valeurs manquantes en cours..."):
                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            if col_to_fill in data.columns:
                                try:
                                    if fill_method == "Supprimer les lignes":
                                        data = data.dropna(subset=[col_to_fill])
                                        st.success(f"✅ Lignes avec valeurs manquantes supprimées dans '{col_to_fill}' pour '{ds}'")
                                    elif fill_method == "Remplacer par moyenne (numérique)":
                                        if data[col_to_fill].dtype in [np.float64, np.int64]:
                                            data[col_to_fill] = data[col_to_fill].fillna(data[col_to_fill].mean())
                                            st.success(f"✅ Valeurs manquantes remplacées par la moyenne dans '{col_to_fill}' pour '{ds}'")
                                        else:
                                            st.warning(f"⚠️ '{col_to_fill}' n'est pas numérique dans '{ds}'.")
                                    elif fill_method == "Remplacer par mode (catégorique)":
                                        if data[col_to_fill].dtype in [object, "category"]:
                                            mode = data[col_to_fill].mode()
                                            data[col_to_fill] = data[col_to_fill].fillna(mode[0] if not mode.empty else None)
                                            st.success(f"✅ Valeurs manquantes remplacées par le mode dans '{col_to_fill}' pour '{ds}'")
                                        else:
                                            st.warning(f"⚠️ '{col_to_fill}' n'est pas catégorique dans '{ds}'.")
                                    elif fill_method == "Plus proche voisin":
                                        data[col_to_fill] = data[col_to_fill].interpolate(method='nearest').ffill().bfill()
                                        st.success(f"✅ Valeurs manquantes interpolées dans '{col_to_fill}' pour '{ds}'")
                                    st.session_state['datasets'][ds] = data
                                    st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                except Exception as e:
                                    st.error(f"❌ Erreur lors du traitement des valeurs manquantes dans '{ds}' : {str(e)}")

            with st.expander("⚠️ Traiter les valeurs aberrantes"):
                col_to_outlier = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].select_dtypes(include=np.number).columns) for ds in selected_datasets]), key="outlier_col")
                outlier_method = st.selectbox("Méthode", ["Supprimer (IQR)", "Remplacer par médiane", "Limiter (IQR)"], key="outlier_method")
                if st.button("Traiter", key="apply_outlier"):
                    with st.spinner("Traitement des valeurs aberrantes en cours..."):
                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            if col_to_outlier in data.columns and data[col_to_outlier].dtype in [np.float64, np.int64]:
                                try:
                                    Q1, Q3 = data[col_to_outlier].quantile([0.25, 0.75])
                                    IQR = Q3 - Q1
                                    lower_bound = Q1 - 1.5 * IQR
                                    upper_bound = Q3 + 1.5 * IQR
                                    if outlier_method == "Supprimer (IQR)":
                                        data = data[(data[col_to_outlier] >= lower_bound) & (data[col_to_outlier] <= upper_bound)]
                                        st.success(f"✅ Aberrants supprimés dans '{col_to_outlier}' pour '{ds}'")
                                    elif outlier_method == "Remplacer par médiane":
                                        data[col_to_outlier] = np.where((data[col_to_outlier] < lower_bound) | (data[col_to_outlier] > upper_bound), 
                                                                        data[col_to_outlier].median(), data[col_to_outlier])
                                        st.success(f"✅ Aberrants remplacés par médiane dans '{col_to_outlier}' pour '{ds}'")
                                    elif outlier_method == "Limiter (IQR)":
                                        data[col_to_outlier] = data[col_to_outlier].clip(lower=lower_bound, upper=upper_bound)
                                        st.success(f"✅ Aberrants limités dans '{col_to_outlier}' pour '{ds}'")
                                    st.session_state['datasets'][ds] = data
                                    st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                except Exception as e:
                                    st.error(f"❌ Erreur lors du traitement des aberrants dans '{ds}' : {str(e)}")
                            else:
                                st.warning(f"⚠️ '{col_to_outlier}' n’est pas numérique ou absente dans '{ds}'.")

            with st.expander("🔗 Effectuer une jointure"):
                if len(dataset_names) > 1:
                    left_ds = st.selectbox("Dataset principal", selected_datasets, key="join_left_ds")
                    right_ds = st.selectbox("Dataset secondaire", dataset_names, key="join_right_ds")
                    col_main = st.selectbox("Colonne principale", st.session_state['datasets'][left_ds].columns, key="join_col_main")
                    col_second = st.selectbox("Colonne secondaire", st.session_state['datasets'][right_ds].columns, key="join_col_second")
                    join_type = st.selectbox("Type", ["inner", "left", "right", "outer"], key="join_type")
                    if st.button("Joindre", key="apply_join"):
                        with st.spinner("Jointure en cours..."):
                            try:
                                data = st.session_state['datasets'][left_ds].merge(st.session_state['datasets'][right_ds], how=join_type, left_on=col_main, right_on=col_second)
                                new_name = f"{left_ds}_joined_{right_ds}"
                                st.session_state['datasets'][new_name] = data
                                st.session_state['explorations'][new_name] = pipeline.explorer.explore(data)
                                st.success(f"✅ Jointure entre '{left_ds}' et '{right_ds}' effectuée ! Nouveau dataset : '{new_name}'")
                            except Exception as e:
                                st.error(f"❌ Erreur lors de la jointure : {str(e)}")

    # Exploration et analyses
    if st.session_state['datasets']:
        st.subheader("📊 Exploration et Analyses")
        st.info("Sélectionnez un dataset pour explorer ou analyser. Téléchargez vos résultats en CSV si besoin.")
        
        dataset_to_analyze = st.selectbox("Dataset à analyser", list(st.session_state['datasets'].keys()), key="analyze_dataset")
        if dataset_to_analyze:
            data = st.session_state['datasets'][dataset_to_analyze]
            exploration = st.session_state['explorations'][dataset_to_analyze]
            
            st.write("**Aperçu des données**")
            st.dataframe(data.head(), use_container_width=True)
            
            with st.expander("📈 Exploration"):
                st.write(f"**Nom du dataset :** {dataset_to_analyze}")
                st.write("**Métadonnées :**")
                st.dataframe(pd.DataFrame(exploration["metadata"].items(), columns=["Colonne", "Type"]), use_container_width=True)
                st.write(f"**Doublons :** {exploration['duplicates']}")
                st.write("**Valeurs manquantes :**")
                missing_df = pd.DataFrame(exploration["missing_values"].items(), columns=["Colonne", "Nb manquants"])
                if missing_df["Nb manquants"].sum() > 0:
                    st.dataframe(missing_df[missing_df["Nb manquants"] > 0], use_container_width=True)
                else:
                    st.write("Aucune valeur manquante")
                st.write("**Statistiques descriptives :**")
                st.dataframe(exploration["description"], use_container_width=True)
                st.write("**Valeurs aberrantes (IQR) :**")
                st.dataframe(pd.DataFrame(exploration["outliers"].items(), columns=["Colonne", "Nb aberrants"]), use_container_width=True)
            
            csv = data.to_csv(index=False)
            st.download_button(label="💾 Télécharger en CSV", data=csv, file_name=f"{dataset_to_analyze}.csv", mime="text/csv", key="download_csv")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Histogrammes", "🔗 Corrélations", "📉 Tests multivariables", "🗺️ Cartes spatiales"])
            
            quant_cols = data.select_dtypes(include=[np.number]).columns
            qual_cols = data.select_dtypes(include=['object', 'category']).columns
            
            with tab1:
                if len(quant_cols) > 0:
                    col_hist = st.selectbox("Colonne", quant_cols, key="hist_select")
                    with st.spinner("Génération de l’histogramme..."):
                        fig_hist = px.histogram(data, x=col_hist, title=f"Distribution de {col_hist}", nbins=50)
                        st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.write("Aucune colonne quantitative disponible.")
            
            with tab2:
                if len(quant_cols) > 1:
                    with st.form(key='corr_form'):
                        col1, col2 = st.columns(2)
                        with col1:
                            var1 = st.selectbox("Variable 1", quant_cols, key="var1_select")
                        with col2:
                            var2 = st.selectbox("Variable 2", quant_cols, index=1, key="var2_select")
                        submit_corr = st.form_submit_button(label="Calculer")
                    
                    if submit_corr:
                        with st.spinner("Calcul de la corrélation..."):
                            corr = correlation(data, var1, var2)
                            st.write("### Résultats")
                            pearson = corr.get('pearson', 'N/A')
                            spearman = corr.get('spearman', 'N/A')
                            st.write(f"**Pearson :** {pearson:.3f}" if isinstance(pearson, (int, float)) else f"**Pearson :** {pearson}")
                            st.write(f"**Spearman :** {spearman:.3f}" if isinstance(spearman, (int, float)) else f"**Spearman :** {spearman}")
                            if 'pearson' in corr and 'spearman' in corr:
                                fig_scatter = px.scatter(data, x=var1, y=var2, trendline="ols", title=f"Corrélation : {var1} vs {var2}")
                                st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.write("Pas assez de colonnes quantitatives.")
            
            with tab3:
                if len(qual_cols) > 0 and len(quant_cols) > 0:
                    with st.form(key='multi_form'):
                        col1, col2 = st.columns(2)
                        with col1:
                            group_var = st.selectbox("Variable qualitative", qual_cols, key="group_select")
                        with col2:
                            value_var = st.selectbox("Variable quantitative", quant_cols, key="value_select")
                        submit_multi = st.form_submit_button(label="Effectuer")
                    
                    if submit_multi:
                        with st.spinner("Calcul des tests multivariables..."):
                            multi = multivariate_tests(data, group_var, value_var)
                            st.write("### Résultats")
                            anova = multi.get('anova', [None, None])
                            kruskal = multi.get('kruskal', [None, None])
                            eta_squared = multi.get('eta_squared', 'N/A')
                            kurtosis = multi.get('kurtosis', 'N/A')
                            skewness = multi.get('skewness', 'N/A')
                            st.write(f"**ANOVA :** F={anova[0]:.2f}" if anova[0] is not None else "**ANOVA :** N/A", 
                                     f", p={anova[1]:.4f}" if anova[1] is not None else ", p=N/A")
                            st.write(f"**Kruskal-Wallis :** H={kruskal[0]:.2f}" if kruskal[0] is not None else "**Kruskal-Wallis :** N/A", 
                                     f", p={kruskal[1]:.4f}" if kruskal[1] is not None else ", p=N/A")
                            st.write(f"**Eta² :** {eta_squared:.4f}" if isinstance(eta_squared, (int, float)) else f"**Eta² :** {eta_squared}")
                            st.write(f"**Kurtosis :** {kurtosis:.2f}" if isinstance(kurtosis, (int, float)) else f"**Kurtosis :** {kurtosis}")
                            st.write(f"**Skewness :** {skewness:.2f}" if isinstance(skewness, (int, float)) else f"**Skewness :** {skewness}")
                            if 'anova' in multi:
                                fig_box = px.box(data, x=group_var, y=value_var, title=f"Distribution de {value_var} par {group_var}")
                                st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.write("Pas assez de colonnes qualitatives/quantitatives.")
            
            with tab4:
                if 'latitude' in data.columns and 'longitude' in data.columns:
                    map_col = st.selectbox("Taille/Couleur", ["Aucune"] + list(data.columns), key="map_col")
                    map_size = st.checkbox("Taille", key="map_size") if map_col != "Aucune" else False
                    with st.spinner("Génération de la carte..."):
                        fig_map = px.scatter_mapbox(
                            data, lat="latitude", lon="longitude", 
                            hover_name="nom_station" if "nom_station" in data.columns else None,
                            size=map_col if map_size and map_col != "Aucune" else None,
                            color=map_col if not map_size and map_col != "Aucune" else None,
                            zoom=10, height=600, title=f"Carte ({dataset_to_analyze})"
                        )
                        fig_map.update_layout(mapbox_style="open-street-map")
                        st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.write("Colonnes 'latitude' et 'longitude' requises.")

if __name__ == "__main__":
    main()