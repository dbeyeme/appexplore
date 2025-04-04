import shutil
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
from typing import Dict, Any, Tuple, Optional
import logging
import csv
import re
import geopandas as gpd
import os
import chardet
import tempfile
import sqlalchemy
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import google.generativeai as genai
from fpdf import FPDF
import time
import unicodedata
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib

from pathlib import Path
from urllib.parse import urlparse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(user)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler("activity.log", maxBytes=10**6, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

user_id = "user_default"
extra = {"user": user_id}
logger = logging.LoggerAdapter(logger, extra)

# Initialisation de l’état de session

def initialize_session_state():
    keys = ['datasets', 'explorations', 'sources', 'json_data', 'chat_history', 
            'test_results', 'test_interpretations', 'models', 'preprocessed_data', 
            'backup_states']
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = {} if key != 'chat_history' else []
            if key == 'backup_states':
                st.session_state[key] = []

# Configuration de l’API Gemini
API_KEY = "AIzaSyBJLhpSfKsbxgVEJwYmPSEZmaVlKt5qNlI"
try:
    genai.configure(api_key=API_KEY)
    client = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    logger.error(f"Erreur configuration Gemini API: {str(e)}")
    st.error("Erreur de configuration de l'API Gemini. Veuillez vérifier la clé API.")


# CSS pour l'interface
st.markdown("""
    <style>
    .stApp {background-color: #0A0F1A; font-family: 'Helvetica', 'Arial', sans-serif; color: #E6E6E6;}
    h1, h2, h3 {color: #FFFFFF; font-weight: 500;}
    .stButton > button {background-color: #FFFFFF; color: #0A0F1A; border: 1px solid #333333; padding: 10px 20px; border-radius: 20px; font-size: 14px; transition: background-color 0.3s ease, transform 0.2s ease; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);}
    .stButton > button:hover {background-color: #E0E0E0; transform: scale(1.05);}
    .chat-message {background-color: #1A2634; padding: 20px; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 3px 6px rgba(0, 0, 0, 0.4); color: #E6E6E6; border-left: 4px solid #1DA1F2;}
    .section-box {background-color: #1A2634; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

class DataLoader:
    def __init__(self):
        self.supported_extensions = ['csv', 'txt', 'xlsx', 'xls', 'json', 'geojson', 'gz', 'zip', 'shp']

    def detect_delimiter(self, content: str) -> str:

        try:
            if not content.strip():
                raise ValueError("Fichier vide")
            sniffer = csv.Sniffer()

            return sniffer.sniff(content[:1024]).delimiter
        except:
            for delim in [',', ';', '\t', '|']:
                if delim in content[:1024]:
                    return delim
            return ','

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))

    def load(self, source, source_type, skip_header=False, header_row=None, selected_sheets=None):
        """
        Charge les données à partir d'une source (URL, fichier, ou base de données).
        """
        temp_dir = tempfile.mkdtemp()
        try:
            if source_type == "url":
                headers = {'User-Agent': 'Mozilla/5.0'}  # Pour éviter les blocages par certains serveurs
                response = requests.get(source, headers=headers, stream=True)
                response.raise_for_status()
                content_bytes = response.content

                # Détection de l'extension à partir de l'URL
                parsed_url = urlparse(source)
                path = parsed_url.path
                file_ext = os.path.splitext(path)[1][1:].lower()  # Extrait l'extension (ex. 'csv', 'xlsx')

                # Si pas d'extension dans l'URL, utiliser Content-Type
                if not file_ext:
                    content_type = response.headers.get('Content-Type', '').lower()
                    logger.info(f"Pas d'extension dans l'URL, Content-Type détecté : {content_type}")
                    if 'csv' in content_type or 'text/plain' in content_type:
                        file_ext = 'csv'
                    elif 'excel' in content_type or 'spreadsheet' in content_type or 'openxml' in content_type:
                        file_ext = 'xlsx'
                    elif 'json' in content_type:
                        file_ext = 'json'
                    elif 'geojson' in content_type:
                        file_ext = 'geojson'
                    elif 'zip' in content_type:
                        file_ext = 'zip'
                    elif 'gzip' in content_type:
                        file_ext = 'gz'
                    else:
                        logger.warning(f"Type MIME non reconnu : {content_type}. Tentative avec contenu brut.")
                        # Tentative basée sur les premiers octets du contenu
                        if content_bytes.startswith(b'PK'):
                            file_ext = 'xlsx'  # Début d'une archive ZIP (xlsx ou zip)
                        elif content_bytes.startswith(b'{'):
                            file_ext = 'json'
                        else:
                            file_ext = 'csv'  # Par défaut, essayer CSV pour les données textuelles

                logger.info(f"Type de fichier détecté : {file_ext}")
                return self._process_file_content(content_bytes, file_ext, temp_dir, skip_header, header_row, selected_sheets)

            elif source_type == "file":
                if isinstance(source, (io.BytesIO, io.StringIO)):
                    content_bytes = source.getvalue() if isinstance(source, io.BytesIO) else source.getvalue().encode()
                    file_ext = os.path.splitext(source.name)[1][1:].lower()
                else:
                    with open(source, 'rb') as f:
                        content_bytes = f.read()
                    file_ext = os.path.splitext(source)[1][1:].lower()
                return self._process_file_content(content_bytes, file_ext, temp_dir, skip_header, header_row, selected_sheets)

            elif source_type == "db":
                db_url, query = source
                return self._load_db(db_url, query), None

        except Exception as e:
            logger.error(f"Erreur chargement: {str(e)}", exc_info=True)
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


    def _handle_shapefile(self, source, temp_dir):
        shp_file = None
        for uploaded_file in source:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            if uploaded_file.name.endswith('.shp'):
                shp_file = file_path
        if not shp_file:
            raise ValueError("Fichier .shp manquant")
        gdf = gpd.read_file(shp_file)
        return gdf.drop(columns=['geometry']).assign(
            latitude=gdf.geometry.centroid.y, 
            longitude=gdf.geometry.centroid.x
        ), None

    def _handle_zip(self, content_bytes, temp_dir):
        with zipfile.ZipFile(io.BytesIO(content_bytes)) as zf:
            zf.testzip()  # Vérifie l'intégrité du ZIP
            extracted_files = zf.namelist()
            if not extracted_files:
                raise ValueError("Archive ZIP vide")
            # Prend le premier fichier pour simplifier (peut être étendu pour plusieurs fichiers)
            first_file = extracted_files[0]
            file_ext = first_file.split('.')[-1].lower()
            with zf.open(first_file) as f:
                content_bytes = f.read()
            return self._process_file_content(content_bytes, file_ext, temp_dir, False, None, None)

    def _handle_database(self, source):
        engine = sqlalchemy.create_engine(source[0])
        return pd.read_sql(source[1], engine), None

    def _process_file_content(self, content_bytes, file_ext, temp_dir, 
                            skip_header, header_row, selected_sheets):
        encoding = chardet.detect(content_bytes[:1024])['encoding'] or 'utf-8'
        content_str = content_bytes.decode(encoding, errors='replace')

        if file_ext in ['csv', 'txt']:
            return self._load_csv_txt(content_str, skip_header, header_row)
        elif file_ext in ['xlsx', 'xls']:
            return self._load_excel(content_bytes, header_row, selected_sheets)
        elif file_ext in ['json', 'geojson']:
            return self._load_json(content_str)
        raise ValueError(f"Type de fichier non supporté: {file_ext}")

    def _load_csv_txt(self, content_str, skip_header, header_row):
        delimiter = self.detect_delimiter(content_str)
        try:
            df = pd.read_csv(
                io.StringIO(content_str), 
                delimiter=delimiter, 
                skiprows=[0] if skip_header else None, 
                header=header_row,
                engine='python',
                on_bad_lines='warn'
            )
            return df, None
        except Exception as e:
            logger.warning(f"Erreur lecture CSV/TXT: {str(e)}")
            return self._recover_csv(content_str, delimiter)

    def _recover_csv(self, content_str, delimiter):
        lines = [line for line in content_str.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Aucune donnée valide dans le fichier CSV/TXT")
        max_cols = max(len(line.split(delimiter)) for line in lines)
        return pd.read_csv(
            io.StringIO(content_str),
            delimiter=delimiter,
            names=range(max_cols),
            engine='python',
            on_bad_lines='skip'
        ), None

    def _load_excel(self, content_bytes, header_row, selected_sheets):
        # Vérification initiale de l'intégrité du fichier Excel (format ZIP)
        try:
            with zipfile.ZipFile(io.BytesIO(content_bytes)) as zf:
                # Vérifie la présence du manifeste obligatoire pour un fichier .xlsx
                if '[Content_Types].xml' not in zf.namelist():
                    raise ValueError("Fichier Excel invalide : manifeste [Content_Types].xml manquant.")
                # Test optionnel de l'intégrité, avec gestion d'erreur
                try:
                    zf.testzip()
                except Exception as zip_err:
                    logger.warning(f"Échec du test d'intégrité ZIP ({str(zip_err)}), tentative de lecture malgré tout.")
        except zipfile.BadZipFile as e:
            logger.error(f"Fichier Excel corrompu ou invalide: {str(e)}")
            raise ValueError(f"Le fichier Excel semble corrompu ou n'est pas un fichier .xlsx/.xls valide : {str(e)}")

        # Tentative de chargement avec openpyxl (pour .xlsx)
        try:
            xl = pd.ExcelFile(io.BytesIO(content_bytes), engine='openpyxl')
            if selected_sheets:
                if len(selected_sheets) > 1:
                    data = {sheet: pd.read_excel(io.BytesIO(content_bytes), sheet_name=sheet, 
                                            header=header_row, engine='openpyxl') 
                        for sheet in selected_sheets}
                    return pd.concat(data.values(), ignore_index=True), content_bytes
                return pd.read_excel(io.BytesIO(content_bytes), sheet_name=selected_sheets[0], 
                                header=header_row, engine='openpyxl'), content_bytes
            return pd.read_excel(io.BytesIO(content_bytes), header=header_row, 
                            engine='openpyxl'), content_bytes
        except Exception as e:
            logger.warning(f"Échec avec openpyxl: {str(e)}. Tentative avec xlrd.")
            # Tentative avec xlrd (pour .xls ou comme secours)
            try:
                xl = pd.ExcelFile(io.BytesIO(content_bytes), engine='xlrd')
                if selected_sheets:
                    if len(selected_sheets) > 1:
                        data = {sheet: pd.read_excel(io.BytesIO(content_bytes), sheet_name=sheet, 
                                                header=header_row, engine='xlrd') 
                            for sheet in selected_sheets}
                        return pd.concat(data.values(), ignore_index=True), content_bytes
                    return pd.read_excel(io.BytesIO(content_bytes), sheet_name=selected_sheets[0], 
                                    header=header_row, engine='xlrd'), content_bytes
                return pd.read_excel(io.BytesIO(content_bytes), header_row=header_row, 
                                engine='xlrd'), content_bytes
            except Exception as fallback_e:
                logger.error(f"Échec avec xlrd également: {str(fallback_e)}")
                raise ValueError(f"Impossible de lire le fichier Excel avec openpyxl ou xlrd : {str(e)}. Le fichier peut être corrompu ou incompatible.")
        
    
    
    def _load_json(self, content_str):
        try:
            json_data = json.loads(content_str)
            if "type" in json_data and json_data["type"] in ["FeatureCollection", "Feature"]:
                gdf = gpd.read_file(io.StringIO(content_str))
                return gdf.drop(columns=['geometry']).assign(
                    latitude=gdf.geometry.centroid.y,
                    longitude=gdf.geometry.centroid.x
                ), None
            return pd.json_normalize(json_data), None
        except json.JSONDecodeError as e:
            logger.error(f"Erreur JSON: {str(e)}")
            raise ValueError(f"Le fichier JSON est invalide ou mal formé : {str(e)}")

    def _analyze_and_interpret(self, test_type: str, results: Dict, error: str = None, is_user_error: bool = False) -> str:
        try:
            if error:
                prompt = f"{'Erreur utilisateur' if is_user_error else 'Erreur interne'}: '{error}'. Explique clairement."
                response = client.generate_content(prompt)
                return response.text.strip()
            else:
                prompt = f"Interprète ces résultats {test_type} : {json.dumps(results)}. Explication simple."
                response = client.generate_content(prompt)
                return response.text.strip()
        except Exception as e:


class DataExplorer:
    @staticmethod
    def explore(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            raise ValueError("Dataset vide")
        return {
            "metadata": df.dtypes.to_dict(),
            "duplicates": df.duplicated().sum(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percent": (df.isnull().sum() / len(df) * 100).to_dict(),
            "description": df.describe(include='all'),

            "outliers": {col: len(df[(df[col] < df[col].quantile(0.25) - 1.5*(df[col].quantile(0.75) - df[col].quantile(0.25))) | 
                                   (df[col] > df[col].quantile(0.75) + 1.5*(df[col].quantile(0.75) - df[col].quantile(0.25)))][col])
                        for col in df.select_dtypes(include=np.number).columns},
            "unique_values": {col: df[col].nunique() for col in df.columns},
            "sample": df.head().to_dict()

        }

def correlation(df: pd.DataFrame, var1: str, var2: str) -> Dict[str, float]:
    results = {}

    if var1 not in df.columns or var2 not in df.columns:
        results["error"] = "Variable(s) non trouvée(s)"
        return results
    if var1 == var2:
        results["error"] = "Les variables doivent être différentes"
        return results
    df_clean = df[[var1, var2]].dropna()
    if len(df_clean) < 2:
        results["error"] = "Pas assez de données valides"
        return results

    if df_clean[var1].dtype in [np.float64, np.int64] and df_clean[var2].dtype in [np.float64, np.int64]:
        try:
            results["pearson"] = stats.pearsonr(df_clean[var1], df_clean[var2])[0]
            results["spearman"] = stats.spearmanr(df_clean[var1], df_clean[var2])[0]
            results["kendall"] = stats.kendalltau(df_clean[var1], df_clean[var2])[0]
        except Exception as e:

            results["error"] = f"Erreur calcul: {str(e)}"
    else:
        results["error"] = "Variables doivent être numériques"

    return results

def chi2_test(df: pd.DataFrame, var1: str, var2: str) -> Dict[str, Any]:
    results = {}

    if var1 not in df.columns or var2 not in df.columns or var1 == var2:
        results["error"] = "Variables invalides ou identiques"
        return results
    contingency_table = pd.crosstab(df[var1], df[var2])
    if contingency_table.empty or contingency_table.size == 0:
        results["error"] = "Pas assez de données croisées"

        return results
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        results.update({"chi2": chi2, "p_value": p, "dof": dof, "expected": expected.tolist()})
    except Exception as e:

        results["error"] = f"Erreur Chi²: {str(e)}"

    return results

def multivariate_tests(df: pd.DataFrame, group_var: str, value_var: str) -> Dict[str, Any]:
    results = {}
    if group_var not in df.columns or value_var not in df.columns:

        results["error"] = "Variable(s) non trouvée(s)"
        return results
    groups = [group[1][value_var].dropna() for group in df.groupby(group_var)]
    if len(groups) <= 1 or df[value_var].dtype not in [np.float64, np.int64]:
        results["error"] = "Données insuffisantes ou non numériques"

        return results
    try:
        results["anova"] = stats.f_oneway(*groups)
        results["kruskal"] = stats.kruskal(*groups)
        results["levene"] = stats.levene(*groups)
        ss_total = df[value_var].var() * (len(df[value_var]) - 1)
        ss_between = sum(len(g) * (g.mean() - df[value_var].mean())**2 for g in groups)
        results["eta_squared"] = ss_between / ss_total
        results["kurtosis"] = df[value_var].kurtosis()
        results["skewness"] = df[value_var].skew()
    except Exception as e:

        results["error"] = f"Erreur tests: {str(e)}"

    return results

def correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    return df[numeric_cols].corr(method=method)

def chi2_matrix(df: pd.DataFrame) -> pd.DataFrame:
    qual_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(qual_cols) < 2:
        return None
    chi2_pvals = pd.DataFrame(index=qual_cols, columns=qual_cols)
    for col1 in qual_cols:
        for col2 in qual_cols:
            result = chi2_test(df, col1, col2)
            chi2_pvals.loc[col1, col2] = result.get('p_value', 1.0) if col1 != col2 else 1.0
    return chi2_pvals.astype(float)

def save_matrix_to_image(matrix: pd.DataFrame, title: str, filename: str, method: str = None):
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, cmap='Blues' if method == 'chi2' else 'RdBu_r',
                    vmin=0 if method == 'chi2' else -1, vmax=1, fmt='.3f', annot_kws={"size": 10})
        plt.title(title)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Erreur sauvegarde matrice: {str(e)}")

        raise

def analyze_and_interpret(test_type: str, results: Dict, error: str = None, is_user_error: bool = False) -> str:
    try:
        if error:

            prompt = f"{'Erreur utilisateur' if is_user_error else 'Erreur interne'}: '{error}'. Explique clairement."
            response = client.generate_content(prompt)
            return response.text.strip()
        else:
            prompt = f"Interprète ces résultats {test_type} : {json.dumps(results)}. Explication simple."
            response = client.generate_content(prompt)
            return response.text.strip()
    except Exception as e:
        return f"Erreur interprétation: {str(e)}"

def create_temp_pdf(dataset_name: str, data: pd.DataFrame, exploration: Dict, 
                   corr_mat: pd.DataFrame, chi2_mat: pd.DataFrame, test_results: Dict, 
                   interpretations: Dict, temp_dir: str) -> str:

    temp_pdf_path = os.path.join(temp_dir, f"temp_{dataset_name}_{int(time.time())}.pdf")
    doc = SimpleDocTemplate(temp_pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Exploration de : {dataset_name}", styles['Title']),
        Spacer(1, 12),
        Paragraph("Métadonnées", styles['Heading2']),
        Paragraph("<br/>".join([f"{k}: {v}" for k, v in exploration["metadata"].items()]), styles['Normal']),
        Spacer(1, 12),
        Paragraph(f"Doublons : {exploration['duplicates']}", styles['Normal']),
        Paragraph(f"Valeurs uniques : <br/>{'<br/>'.join([f'{k}: {v}' for k, v in exploration['unique_values'].items()])}", styles['Normal']),
        Spacer(1, 12),
        Paragraph("Valeurs manquantes", styles['Heading2']),
        Paragraph("<br/>".join([f"{k}: {v} ({exploration['missing_percent'][k]:.2f}%)" for k, v in exploration["missing_values"].items()]), styles['Normal']),
        Spacer(1, 12),
        Paragraph("Statistiques descriptives", styles['Heading2']),
        Paragraph(str(exploration["description"]).replace('\n', '<br/>'), styles['Normal']),
        Spacer(1, 12),
        Paragraph("Valeurs aberrantes (IQR)", styles['Heading2']),
        Paragraph("<br/>".join([f"{k}: {v}" for k, v in exploration["outliers"].items()]), styles['Normal']),
        Spacer(1, 12),
        Paragraph("Résultats des tests", styles['Heading2'])
    ]
    for key, result in test_results.items():
        if key == "correlation":
            story.append(Paragraph(f"Corrélation {result['var1']} vs {result['var2']}:<br/>Pearson: {result.get('pearson', 'N/A'):.3f}<br/>Spearman: {result.get('spearman', 'N/A'):.3f}<br/>Kendall: {result.get('kendall', 'N/A'):.3f}", styles['Normal']))
            story.append(Paragraph(f"Interprétation : {interpretations.get(key, 'N/A')}", styles['Normal']))
        elif key == "chi2":
            story.append(Paragraph(f"Chi² {result['var1']} vs {result['var2']}:<br/>Chi²: {result.get('chi2', 'N/A'):.2f}<br/>p-valeur: {result.get('p_value', 'N/A'):.4f}", styles['Normal']))
            story.append(Paragraph(f"Interprétation : {interpretations.get(key, 'N/A')}", styles['Normal']))
        elif key == "multivariate":
            story.append(Paragraph(f"Tests {result['group_var']} vs {result['value_var']}:<br/>ANOVA F={result['anova'][0]:.2f}, p={result['anova'][1]:.4f}<br/>Kruskal H={result['kruskal'][0]:.2f}, p={result['kruskal'][1]:.4f}", styles['Normal']))
            story.append(Paragraph(f"Interprétation : {interpretations.get(key, 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 12))

    if corr_mat is not None:
        corr_img = os.path.join(temp_dir, f"corr_{dataset_name}.png")
        save_matrix_to_image(corr_mat, "Matrice de Corrélation", corr_img)
        story.append(Image(corr_img, width=400, height=300))
    if chi2_mat is not None:
        chi2_img = os.path.join(temp_dir, f"chi2_{dataset_name}.png")
        save_matrix_to_image(chi2_mat, "Matrice Chi²", chi2_img, method='chi2')
        story.append(Image(chi2_img, width=400, height=300))
    

    doc.build(story)
    logger.info(f"PDF temporaire généré à {temp_pdf_path}")
    return temp_pdf_path


def generate_gemini_report(pdf_path: str) -> str:
    try:
        with open(pdf_path, "rb") as f:
            response = client.generate_content([
                {"mime_type": "application/pdf", "data": f.read()},
                "Analyse et résume ce rapport en français dans un style clair, structuré et professionnel."
            ])

        logger.info("Rapport Gemini généré avec succès")
        return re.sub(r'[#*]+', '', response.text)
    except Exception as e:
        logger.error(f"Erreur génération rapport: {str(e)}")
        return f"Erreur génération rapport: {str(e)}"


def display_progressive_report(report: str) -> str:
    lines = report.split('\n')
    chat_container = st.empty()
    current_text = ""
    for i, line in enumerate(lines):
        if line.strip():
            is_title = (i == 0) or (i > 0 and not lines[i-1].strip()) or (i < len(lines)-1 and not lines[i+1].strip())
            substituted_line = re.sub(r'(\d+\.?\d*)', r'<span style="color: #FF4500; font-weight: bold;">\1</span>', line)
            formatted_line = (
                f'<div style="color: #E6E6E6; font-weight: bold; font-size: 18px; margin-bottom: 10px;">{line}</div>' 
                if is_title 
                else f'<div style="color: #E6E6E6; font-size: 14px; margin-bottom: 5px;">{substituted_line}</div>'
            )
            current_text += formatted_line
            chat_container.markdown(f'<div class="chat-message">{current_text}</div>', unsafe_allow_html=True)
            time.sleep(0.2)
    return report

def create_downloadable_pdf(dataset_name: str, report_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    normalized_text = unicodedata.normalize('NFKD', report_text).encode('latin-1', 'replace').decode('latin-1')
    for line in normalized_text.split('\n'):
        pdf.cell(0, 10, line, ln=True)
    return pdf.output(dest="S").encode('latin-1')

def safe_dataframe_display(df: pd.DataFrame):
    df_safe = df.copy()
    for col in df_safe.select_dtypes(include=['object']).columns:
        df_safe[col] = df_safe[col].astype(str)
    return df_safe


def preprocess_data(df: pd.DataFrame, features: list, target: str, encoding_method: str, 
                   normalization_method: str, modeling_type: str, model_type: str) -> Tuple[pd.DataFrame, pd.Series, list, Any]:

    X = df[features].copy()
    y = df[target].copy() if modeling_type != "Clustering" else None
    transformer = None

    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    processed_features = []

    for col in X.columns:
        if col in categorical_cols:
            if encoding_method == "Label Encoding":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                processed_features.append(col)
            elif encoding_method == "One-Hot Encoding":
                dummies = pd.get_dummies(X[col], prefix=col)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                processed_features.extend(dummies.columns)
            elif encoding_method == "Exclude":
                X = X.drop(columns=[col])
        else:
            processed_features.append(col)

    if normalization_method != "None" and numeric_cols.any():
        if normalization_method == "StandardScaler":
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        elif normalization_method == "RobustScaler":
            scaler = RobustScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        elif normalization_method == "MinMaxScaler":
            scaler = MinMaxScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    if modeling_type == "Regression" and y is not None and y.dtype == 'object':

        raise ValueError("La variable cible doit être numérique pour la régression")

    elif modeling_type == "Classification" and y is not None and y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

    X = X.dropna()
    if modeling_type != "Clustering" and y is not None:
        y = y.loc[X.index]

    if modeling_type == "Regression" and "Polynomial" in model_type:
        transformer = PolynomialFeatures(degree=2)
        X = transformer.fit_transform(X)
        processed_features = transformer.get_feature_names_out(input_features=processed_features).tolist()

    return X, y, processed_features, transformer


def train_model(df: pd.DataFrame, features: list, target: str, modeling_type: str, 
                model_type: str, encoding_method: str, normalization_method: str) -> Tuple[Any, Dict, list, Any]:
    try:
        X, y, processed_features, transformer = preprocess_data(df, features, target if modeling_type != "Clustering" else None, 
                                                              encoding_method, normalization_method, modeling_type, model_type)
        
        if len(X) < 2:
            raise ValueError("Pas assez de données après prétraitement")

        if modeling_type == "Regression":
            if y.dtype not in [np.float64, np.int64]:
                raise ValueError("La cible doit être numérique pour la régression")
            model_options = {
                "Linear Regression": LinearRegression(),
                "Polynomial Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBoost": xgb.XGBRegressor(),
                "Stacking": StackingRegressor(estimators=[("lr", LinearRegression()), ("rf", RandomForestRegressor()), ("xgb", xgb.XGBRegressor())], 
                                            final_estimator=LinearRegression()),
                "Voting": VotingRegressor(estimators=[("lr", LinearRegression()), ("rf", RandomForestRegressor()), ("xgb", xgb.XGBRegressor())])
            }
            model = model_options[model_type]

            model.fit(X, y)
            y_pred = model.predict(X)
            metrics = {"R²": r2_score(y, y_pred), "MSE": mean_squared_error(y, y_pred)}

        elif modeling_type == "Classification":
            if y.dtype in [np.float64, np.int64] and y.nunique() > 10:

                raise ValueError("La cible doit être catégorique pour la classification")
            model_options = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "Logistic Regression": LogisticRegression(),
                "Stacking": StackingClassifier(estimators=[("dt", DecisionTreeClassifier()), ("rf", RandomForestClassifier()), 
                                                         ("xgb", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))], 
                                              final_estimator=LogisticRegression()),
                "Voting": VotingClassifier(estimators=[("dt", DecisionTreeClassifier()), ("rf", RandomForestClassifier()), 
                                                      ("xgb", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))], voting='soft')
            }
            model = model_options[model_type]

            model.fit(X, y)
            y_pred = model.predict(X)
            metrics = {"Accuracy": accuracy_score(y, y_pred)}

        elif modeling_type == "Clustering":

            model = KMeans(n_clusters=3) if model_type == "K-Means" else None

            model.fit(X)
            y_pred = model.predict(X)
            metrics = {"Inertia": model.inertia_}


        logger.info(f"Modèle {model_type} ({modeling_type}) entraîné avec succès")

        return model, metrics, processed_features, transformer

    except Exception as e:
        error_msg = str(e)

        logger.error(f"Erreur entraînement: {error_msg}", exc_info=True)
        st.error(analyze_and_interpret("training", None, error_msg, is_user_error="Erreur de l'utilisateur" in error_msg))

        return None, {}, [], None

class DataPipeline:
    def __init__(self):
        self.loader = DataLoader()
        self.explorer = DataExplorer()


    def process(self, source: str, source_type: str, name: str, skip_header: bool = False, 
                header_row: int = None, selected_sheets=None) -> Tuple[pd.DataFrame, Dict, bytes]:
        try:
            data, raw_data = self.loader.load(source, source_type, skip_header, header_row, selected_sheets)
            if data is not None:
                exploration = self.explorer.explore(data)
                return data, exploration, raw_data
            return None, None, raw_data
        except Exception as e:
            logger.error(f"Erreur pipeline: {str(e)}")
            st.error(f"Erreur traitement {name}: {str(e)}")
            return None, None, None

def backup_state():
    st.session_state['backup_states'].append({
        'datasets': st.session_state['datasets'].copy(),
        'explorations': st.session_state['explorations'].copy(),
        'timestamp': time.time()
    })
    if len(st.session_state['backup_states']) > 5:
        st.session_state['backup_states'].pop(0)

def restore_state(timestamp):
    for state in st.session_state['backup_states']:
        if state['timestamp'] == timestamp:
            st.session_state['datasets'] = state['datasets'].copy()
            st.session_state['explorations'] = state['explorations'].copy()
            return True
    return False

def main():
    initialize_session_state()

    pipeline = DataPipeline()

    st.title("🔍 Exploration, Analyse et Modélisation de Données")

    with st.sidebar:
        st.header("📥 Chargement des Données")

        source_type = st.selectbox("Source", ["URL", "Fichiers Locaux", "Base de Données"], key="source_type_select")

        if source_type == "URL":
            num_datasets = st.number_input("Nombre de datasets", min_value=1, max_value=10, value=1, key="num_datasets_input")
            for i in range(int(num_datasets)):
                with st.expander(f"Dataset {i+1}"):
                    source = st.text_input("URL", key=f"source_url_{i}")
                    name = st.text_input("Nom", f"dataset_{i+1}", key=f"name_input_{i}")
                    skip_header = st.checkbox("Ignorer première ligne", key=f"skip_header_{i}")
                    header_row = st.number_input("Ligne en-tête", min_value=0, key=f"header_row_{i}")
                    preview = st.button("👁️ Prévisualiser", key=f"preview_url_{i}")
                    load = st.button("📤 Charger", key=f"load_url_{i}")
                    
                    if preview and source:
                        try:
                            df, _ = pipeline.loader.load(source, "url", skip_header, header_row)
                            if df is not None:
                                st.dataframe(df.head())
                        except Exception as e:
                            st.error(f"Erreur prévisualisation: {str(e)}")
                    
                    if load and source and name:
                        backup_state()
                        data, exploration, raw_data = pipeline.process(source, "url", name, skip_header, header_row)
                        if data is not None:
                            st.session_state['datasets'][name] = data
                            st.session_state['explorations'][name] = exploration
                            st.session_state['sources'][name] = (source, "url", skip_header, header_row)
                            st.session_state['test_results'][name] = {}
                            st.session_state['test_interpretations'][name] = {}
                            st.success(f"✅ '{name}' chargé ({len(data)} lignes)")
                            logger.info(f"Dataset {name} chargé avec succès depuis URL")

        elif source_type == "Fichiers Locaux":
            uploaded_files = st.file_uploader("Fichiers", accept_multiple_files=True,
                                            type=['csv', 'xlsx', 'xls', 'json', 'geojson', 'txt', 'gz', 'zip', 'shp', 'shx', 'dbf'],
                                            key="file_uploader")
            skip_header = st.checkbox("Ignorer la première ligne (CSV/TXT)", key="skip_header_files")
            header_row = st.number_input("Ligne comme en-tête", min_value=0, value=0, key="header_row_files")
            if uploaded_files:
                for file in uploaded_files:
                    name = st.text_input(f"Nom pour {file.name}", file.name.split('.')[0], key=f"name_file_{file.name}")
                    if file.name.endswith(('.xlsx', '.xls')):
                        xl = pd.ExcelFile(file)
                        sheets = st.multiselect(f"Feuilles {file.name}", xl.sheet_names, default=[xl.sheet_names[0]], key=f"sheets_{file.name}")
                    else:
                        sheets = None
                    if st.button(f"📤 Charger {file.name}", key=f"load_file_{file.name}"):
                        backup_state()
                        data, exploration, raw_data = pipeline.process(file, "file", name, skip_header, header_row, sheets)
                        if data is not None:
                            st.session_state['datasets'][name] = data
                            st.session_state['explorations'][name] = exploration
                            st.session_state['sources'][name] = (raw_data, "file", skip_header, header_row, file.name)
                            st.session_state['test_results'][name] = {}
                            st.session_state['test_interpretations'][name] = {}
                            st.success(f"✅ '{name}' chargé ({len(data)} lignes)")
                            logger.info(f"Dataset {name} chargé avec succès depuis fichier local")

        elif source_type == "Base de Données":
            db_url = st.text_input("URL de la base", key="db_url_input")
            query = st.text_area("Requête SQL", "SELECT * FROM table_name", key="db_query_input")
            name = st.text_input("Nom du dataset", "db_dataset", key="db_name_input")
            if st.button("📤 Charger", key="load_db"):
                backup_state()

                data, exploration, raw_data = pipeline.process((db_url, query), "db", name)
                if data is not None:
                    st.session_state['datasets'][name] = data
                    st.session_state['explorations'][name] = exploration
                    st.session_state['sources'][name] = (query, "db", False, None)
                    st.session_state['test_results'][name] = {}
                    st.session_state['test_interpretations'][name] = {}
                    st.success(f"✅ '{name}' chargé ({len(data)} lignes)")

                    logger.info(f"Dataset {name} chargé avec succès depuis DB")


        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:

            if st.button("🔄 Actualiser", key="refresh_datasets"):

                logger.info("Actualisation des datasets")
                for name, (source, source_type, skip_header, header_row, *extra) in st.session_state['sources'].items():
                    if source_type == "file":
                        file_source = io.BytesIO(source)
                        file_source.name = extra[0] if extra else name
                        data, exploration, raw_data = pipeline.process(file_source, source_type, name, skip_header, header_row)
                        if name in st.session_state['datasets']:
                            data = st.session_state['datasets'][name]  # Conserver les données traitées
                    else:
                        data, exploration, raw_data = pipeline.process(source, source_type, name, skip_header, header_row)
                    if data is not None:
                        st.session_state['datasets'][name] = data
                        st.session_state['explorations'][name] = pipeline.explorer.explore(data)
                        st.session_state['test_results'][name] = {}
                        st.session_state['test_interpretations'][name] = {}

                st.success("✅ Datasets actualisés")
        with col2:
            if st.button("🗑️ Réinitialiser", key="reset_app"):
                logger.info("Réinitialisation de l’application")
                st.session_state.clear()
                initialize_session_state()
                st.rerun()

        if st.session_state['backup_states']:
            st.header("🔄 Historique")
            backup_times = [state['timestamp'] for state in st.session_state['backup_states']]
            selected_backup = st.selectbox("Restaurer état", 
                                         options=[time.ctime(t) for t in backup_times], 
                                         key="restore_state_select")
            if st.button("Restaurer", key="restore_state_button"):
                if restore_state(backup_times[[time.ctime(t) for t in backup_times].index(selected_backup)]):
                    st.success("État restauré")


        st.header("⚙️ Gestion des Datasets")
        if st.session_state['datasets']:
            dataset_names = list(st.session_state['datasets'].keys())

            selected_datasets = st.multiselect("Datasets à traiter", dataset_names, default=dataset_names[0], key="datasets_to_process")

            if selected_datasets:
                with st.expander("🔄 Conversion des Types"):
                    col_to_convert = st.multiselect("Colonnes", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="convert_columns")
                    type_to_convert = st.selectbox("Type", ["Entier (int)", "Décimal (float)", "Catégorie (category)", "Date (datetime)", "Timestamp vers Date", 
                                                          "Extraire Mois", "Extraire Jour de la semaine", "Extraire Jour du mois", "Extraire Heure", "Extraire Année"],
                                                  key="convert_type_select")
                    if st.button("✅ Appliquer", key="apply_conversion_types"):
                        backup_state()

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
                                        st.session_state['datasets'][ds] = data
                                        st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                        st.success(f"✅ Conversion appliquée pour '{ds}'")

                                    except Exception as e:
                                        st.error(analyze_and_interpret("conversion", None, str(e), True))

                with st.expander("🧹 Nettoyage des Valeurs"):
                    col_to_clean = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="clean_column_select")
                    action = st.selectbox("Action", ["Supprimer des caractères", "Remplacer des caractères"], key="clean_action_select")
                    pattern = st.text_input("Motif (regex)", key="clean_pattern_input")
                    replacement = st.text_input("Remplacement", "", key="clean_replacement_input") if action == "Remplacer des caractères" else ""
                    if st.button("✅ Appliquer", key="apply_clean_values"):
                        backup_state()

                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            if col_to_clean in data.columns:
                                if action == "Supprimer des caractères":
                                    data[col_to_clean] = data[col_to_clean].apply(lambda x: re.sub(pattern, '', str(x)) if pd.notnull(x) else x)
                                else:
                                    data[col_to_clean] = data[col_to_clean].apply(lambda x: re.sub(pattern, replacement, str(x)) if pd.notnull(x) else x)
                                st.session_state['datasets'][ds] = data
                                st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                st.success(f"✅ Nettoyage appliqué pour '{ds}'")


                with st.expander("🗑️ Suppression de Colonnes"):
                    cols_to_drop = st.multiselect("Colonnes", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="drop_columns_multiselect")
                    if st.button("✅ Supprimer", key="apply_drop_columns"):
                        backup_state()

                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            data.drop(columns=[col for col in cols_to_drop if col in data.columns], inplace=True)
                            st.session_state['datasets'][ds] = data
                            st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                            st.success(f"✅ Colonnes supprimées pour '{ds}'")


                with st.expander("🚫 Suppression de Lignes"):
                    col_to_filter = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="filter_column_select")
                    filter_expr = st.text_input("Expression (ex. 'Station_1')", key="filter_expr_input")
                    filter_type = st.selectbox("Type", ["Valeur exacte", "Regex"], key="filter_type_select")
                    if st.button("✅ Supprimer", key="apply_drop_rows"):
                        backup_state()

                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            if col_to_filter in data.columns:
                                if filter_type == "Valeur exacte":
                                    data = data[data[col_to_filter] != filter_expr]
                                else:
                                    data = data[~data[col_to_filter].apply(lambda x: bool(re.search(filter_expr, str(x)) if pd.notnull(x) else False))]
                                st.session_state['datasets'][ds] = data
                                st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                st.success(f"✅ Lignes supprimées pour '{ds}'")


                with st.expander("➕ Création de Colonne"):
                    new_col_name = st.text_input("Nom", key="new_col_name_input")
                    base_col = st.selectbox("Base", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="new_col_base_select")
                    new_col_action = st.selectbox("Action", ["Copie avec conversion", "Copie avec nettoyage"], key="new_col_action_select")
                    if new_col_action == "Copie avec conversion":
                        new_col_type = st.selectbox("Type", ["Entier (int)", "Décimal (float)", "Catégorie (category)", "Date (datetime)", "Timestamp vers Date", 
                                                            "Extraire Mois", "Extraire Jour de la semaine", "Extraire Jour du mois", "Extraire Heure", "Extraire Année"],
                                                   key="new_col_type_select")
                    else:
                        new_col_pattern = st.text_input("Motif", key="new_col_pattern_input")
                        new_col_replace = st.text_input("Remplacement", "", key="new_col_replace_input")
                    if st.button("✅ Créer", key="apply_create_column"):
                        backup_state()

                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            if base_col in data.columns:
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
                                else:
                                    data[new_col_name] = data[base_col].apply(lambda x: re.sub(new_col_pattern, new_col_replace or '', str(x)) if pd.notnull(x) else x)
                                st.session_state['datasets'][ds] = data
                                st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                st.success(f"✅ '{new_col_name}' créé pour '{ds}'")


                with st.expander("🕳️ Traitement des Valeurs Manquantes"):
                    cols_to_fill = st.multiselect("Colonnes", ["Toutes"] + list(set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets])), key="fill_missing_columns")
                    fill_method = st.selectbox("Méthode", ["Supprimer les lignes", "Supprimer toutes lignes vides", "Remplacer par moyenne", "Remplacer par mode", "Plus proche voisin"],
                                              key="fill_method_select")
                    if st.button("✅ Traiter", key="apply_fill_missing"):
                        backup_state()

                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            if "Toutes" in cols_to_fill and fill_method == "Supprimer toutes lignes vides":
                                data.dropna(how='all', inplace=True)
                            else:
                                target_cols = [col for col in cols_to_fill if col != "Toutes" and col in data.columns]
                                for col in target_cols:
                                    if fill_method == "Supprimer les lignes":
                                        data.dropna(subset=[col], inplace=True)
                                    elif fill_method == "Remplacer par moyenne" and data[col].dtype in [np.float64, np.int64]:
                                        data[col].fillna(data[col].mean(), inplace=True)
                                    elif fill_method == "Remplacer par mode":
                                        mode = data[col].mode()
                                        data[col].fillna(mode[0] if not mode.empty else None, inplace=True)
                                    elif fill_method == "Plus proche voisin":
                                        data[col] = data[col].interpolate(method='nearest').ffill().bfill()
                            st.session_state['datasets'][ds] = data
                            st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                            st.success(f"✅ Valeurs manquantes traitées pour '{ds}'")


                with st.expander("⚠️ Traitement des Valeurs Aberrantes"):
                    col_to_outlier = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].select_dtypes(include=np.number).columns) for ds in selected_datasets]),
                                                 key="outlier_column_select")
                    outlier_method = st.selectbox("Méthode", ["Supprimer (IQR)", "Remplacer par médiane", "Limiter (IQR)"], key="outlier_method_select")
                    if st.button("✅ Traiter", key="apply_handle_outliers"):
                        backup_state()

                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            if col_to_outlier in data.columns and data[col_to_outlier].dtype in [np.float64, np.int64]:
                                Q1, Q3 = data[col_to_outlier].quantile([0.25, 0.75])
                                IQR = Q3 - Q1
                                lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                                if outlier_method == "Supprimer (IQR)":
                                    data = data[(data[col_to_outlier] >= lower_bound) & (data[col_to_outlier] <= upper_bound)]
                                elif outlier_method == "Remplacer par médiane":
                                    data[col_to_outlier] = np.where((data[col_to_outlier] < lower_bound) | (data[col_to_outlier] > upper_bound), data[col_to_outlier].median(), data[col_to_outlier])
                                elif outlier_method == "Limiter (IQR)":
                                    data[col_to_outlier] = data[col_to_outlier].clip(lower=lower_bound, upper=upper_bound)
                                st.session_state['datasets'][ds] = data
                                st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                st.success(f"✅ Aberrants traités pour '{ds}'")


                with st.expander("🔗 Jointure de Datasets"):
                    if len(dataset_names) > 1:
                        left_ds = st.selectbox("Dataset principal", selected_datasets, key="join_left_ds_select")
                        right_ds = st.selectbox("Dataset secondaire", dataset_names, key="join_right_ds_select")
                        col_main = st.selectbox("Colonne principale", st.session_state['datasets'][left_ds].columns, key="join_main_col_select")
                        col_second = st.selectbox("Colonne secondaire", st.session_state['datasets'][right_ds].columns, key="join_second_col_select")
                        join_type = st.selectbox("Type", ["inner", "left", "right", "outer"], key="join_type_select")
                        if st.button("✅ Joindre", key="apply_join_datasets"):
                            backup_state()

                            data = st.session_state['datasets'][left_ds].merge(st.session_state['datasets'][right_ds], how=join_type, left_on=col_main, right_on=col_second)
                            new_name = f"{left_ds}_joined_{right_ds}"
                            st.session_state['datasets'][new_name] = data
                            st.session_state['explorations'][new_name] = pipeline.explorer.explore(data)
                            st.session_state['test_results'][new_name] = {}
                            st.session_state['test_interpretations'][new_name] = {}
                            st.success(f"✅ Jointure effectuée : '{new_name}'")

                # Nouvel expander pour Numérisation et Normalisation
                with st.expander("🔢 Numérisation et Normalisation"):
                    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
                    
                    encoding_method = st.selectbox("Méthode de numérisation", ["Label Encoding", "One-Hot Encoding", "Exclude"], key="encoding_method_select")
                    normalization_method = st.selectbox("Méthode de normalisation", ["None", "StandardScaler", "RobustScaler", "MinMaxScaler"], key="normalization_method_select")
                    columns_to_process = st.multiselect("Colonnes à traiter", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="preprocess_columns_multiselect")
                    
                    if st.button("✅ Appliquer", key="apply_preprocessing"):
                        backup_state()
                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            
                            # Numérisation (Encoding)
                            if encoding_method != "Exclude" and columns_to_process:
                                for col in columns_to_process:
                                    if col in data.columns and data[col].dtype in ['object', 'category']:
                                        if encoding_method == "Label Encoding":
                                            le = LabelEncoder()
                                            data[col] = le.fit_transform(data[col].astype(str))
                                        elif encoding_method == "One-Hot Encoding":
                                            ohe = OneHotEncoder(sparse_output=False, drop='first')
                                            encoded_data = ohe.fit_transform(data[[col]].astype(str))
                                            encoded_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                                            data = pd.concat([data.drop(columns=[col]), pd.DataFrame(encoded_data, columns=encoded_cols, index=data.index)], axis=1)
                            
                            # Normalisation
                            if normalization_method != "None" and columns_to_process:
                                numeric_cols = [col for col in columns_to_process if col in data.columns and data[col].dtype in [np.float64, np.int64]]
                                if numeric_cols:
                                    if normalization_method == "StandardScaler":
                                        scaler = StandardScaler()
                                        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                                    elif normalization_method == "RobustScaler":
                                        scaler = RobustScaler()
                                        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                                    elif normalization_method == "MinMaxScaler":
                                        scaler = MinMaxScaler()
                                        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                            
                            st.session_state['datasets'][ds] = data
                            st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                            st.success(f"✅ Numérisation et normalisation appliquées pour '{ds}'")

    if st.session_state['datasets']:
        st.markdown("<div class='section-box'><h2>📋 Aperçu des Données</h2>", unsafe_allow_html=True)
        selected_preview = st.selectbox("Choisir un dataset", list(st.session_state['datasets'].keys()), key="preview_dataset_select")
        st.dataframe(safe_dataframe_display(st.session_state['datasets'][selected_preview].head(10)), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-box'><h2>📊 Exploration, Visualisation et Modélisation</h2>", unsafe_allow_html=True)


        data = st.session_state['datasets'][dataset_to_analyze]
        exploration = st.session_state['explorations'][dataset_to_analyze]
        test_results = st.session_state['test_results'][dataset_to_analyze]
        test_interpretations = st.session_state['test_interpretations'][dataset_to_analyze]

        st.write("**Aperçu des données**")
        st.dataframe(safe_dataframe_display(data.head()), use_container_width=True)

        with st.expander("📈 Exploration"):


            st.write(f"**Nom du dataset :** {dataset_to_analyze}")
            st.write("**Métadonnées :**")
            st.dataframe(pd.DataFrame(exploration["metadata"].items(), columns=["Colonne", "Type"]))
            st.write(f"**Doublons :** {exploration['duplicates']}")
            st.write("**Valeurs uniques :**")
            st.dataframe(pd.DataFrame(exploration["unique_values"].items(), columns=["Colonne", "Nb unique"]))
            st.write("**Valeurs manquantes :**")

            missing_df = pd.DataFrame({"Colonne": exploration["missing_values"].keys(), "Nb manquants": exploration["missing_values"].values(), 
                                     "Pourcentage (%)": [f"{v:.2f}" for v in exploration["missing_percent"].values()]})

            st.dataframe(missing_df[missing_df["Nb manquants"] > 0]) if missing_df["Nb manquants"].sum() > 0 else st.write("Aucune valeur manquante")
            st.write("**Statistiques descriptives :**")
            st.dataframe(exploration["description"])
            st.write("**Valeurs aberrantes (IQR) :**")
            st.dataframe(pd.DataFrame(exploration["outliers"].items(), columns=["Colonne", "Nb aberrants"]))


        st.download_button(label="💾 Télécharger en CSV", data=data.to_csv(index=False), file_name=f"{dataset_to_analyze}.csv", mime="text/csv", key="download_csv_button")


        tab1, tab2, tab3, tab4 = st.tabs([
            "🎨 Premiers Pas : Découvrez Vos Données",
            "🔍 Zoom Sur les Relations",
            "🧩 Les Liens en un Coup d’Œil",
            "🤖 Prédictions et Magie IA"
        ])
        quant_cols = data.select_dtypes(include=[np.number]).columns
        qual_cols = data.select_dtypes(include=['object', 'category']).columns

        with tab1:

            if len(quant_cols) > 0:
                col_hist = st.selectbox("Colonne pour histogramme", quant_cols, key="hist_column_select")
                fig_hist = px.histogram(data, x=col_hist, title=f"Distribution de {col_hist}", nbins=50, marginal="box")
                st.plotly_chart(fig_hist, use_container_width=True)
            if len(qual_cols) > 0:
                qual_col = st.selectbox("Variable qualitative", qual_cols, key="bar_qual_column_select")

                fig_bar = px.histogram(data, x=qual_col, title=f"Répartition de {qual_col}", color=qual_col)
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:

            if len(quant_cols) > 1:
                with st.form(key='corr_form'):
                    col1, col2 = st.columns(2)
                    with col1:

                        var1 = st.selectbox("Variable 1", quant_cols, key="corr_var1_select")
                    with col2:
                        var2 = st.selectbox("Variable 2", quant_cols, index=1, key="corr_var2_select")

                    submit_corr = st.form_submit_button(label="✅ Calculer")
                if submit_corr:
                    corr = correlation(data, var1, var2)
                    test_results["correlation"] = {"var1": var1, "var2": var2, **corr}
                    if "error" in corr:

                        st.error(analyze_and_interpret("correlation", None, corr["error"], True))

                    else:
                        st.write(f"**Pearson :** {corr.get('pearson', 'N/A'):.3f}")
                        st.write(f"**Spearman :** {corr.get('spearman', 'N/A'):.3f}")
                        st.write(f"**Kendall :** {corr.get('kendall', 'N/A'):.3f}")
                        interpretation = analyze_and_interpret("correlation", corr)
                        test_interpretations["correlation"] = interpretation
                        st.write(f"**Interprétation :** {interpretation}")

                        fig_scatter = px.scatter(data, x=var1, y=var2, trendline="ols", title=f"Corrélation : {var1} vs {var2}")

                        st.plotly_chart(fig_scatter, use_container_width=True)

            if len(qual_cols) > 0 and len(quant_cols) > 0:
                with st.form(key='multi_form'):
                    col1, col2 = st.columns(2)
                    with col1:

                        group_var = st.selectbox("Variable qualitative", qual_cols, key="multi_group_var_select")
                    with col2:
                        value_var = st.selectbox("Variable quantitative", quant_cols, key="multi_value_var_select")

                    submit_multi = st.form_submit_button(label="✅ Effectuer")
                if submit_multi:
                    multi = multivariate_tests(data, group_var, value_var)
                    test_results["multivariate"] = {"group_var": group_var, "value_var": value_var, **multi}
                    if "error" in multi:

                        st.error(analyze_and_interpret("multivariate", None, multi["error"], True))
                    else:
                        st.write(f"**ANOVA :** F={multi.get('anova')[0]:.2f}, p={multi.get('anova')[1]:.4f}")
                        st.write(f"**Kruskal-Wallis :** H={multi.get('kruskal')[0]:.2f}, p={multi.get('kruskal')[1]:.4f}")
                        st.write(f"**Levene :** F={multi.get('levene')[0]:.2f}, p={multi.get('levene')[1]:.4f}")

                        st.write(f"**Eta² :** {multi.get('eta_squared', 'N/A'):.4f}")
                        st.write(f"**Kurtosis :** {multi.get('kurtosis', 'N/A'):.2f}")
                        st.write(f"**Skewness :** {multi.get('skewness', 'N/A'):.2f}")
                        interpretation = analyze_and_interpret("multivariate", multi)
                        test_interpretations["multivariate"] = interpretation
                        st.write(f"**Interprétation :** {interpretation}")
                        fig_box = px.box(data, x=group_var, y=value_var, title=f"Distribution de {value_var} par {group_var}", color=group_var)
                        st.plotly_chart(fig_box, use_container_width=True)

            if len(qual_cols) > 1:
                with st.form(key='chi2_form'):
                    col1, col2 = st.columns(2)
                    with col1:

                        chi_var1 = st.selectbox("Variable qualitative 1", qual_cols, key="chi2_var1_select")
                    with col2:
                        chi_var2 = st.selectbox("Variable qualitative 2", qual_cols, index=1, key="chi2_var2_select")

                    submit_chi2 = st.form_submit_button(label="✅ Effectuer")
                if submit_chi2 and chi_var1 != chi_var2:
                    chi2_results = chi2_test(data, chi_var1, chi_var2)
                    test_results["chi2"] = {"var1": chi_var1, "var2": chi_var2, **chi2_results}
                    if "error" in chi2_results:

                        st.error(analyze_and_interpret("chi2", None, chi2_results["error"], True))

                    else:
                        st.write(f"**Chi² :** {chi2_results.get('chi2', 'N/A'):.2f}")
                        st.write(f"**p-valeur :** {chi2_results.get('p_value', 'N/A'):.4f}")
                        st.write(f"**Degrés de liberté :** {chi2_results.get('dof', 'N/A')}")
                        interpretation = analyze_and_interpret("chi2", chi2_results)
                        test_interpretations["chi2"] = interpretation
                        st.write(f"**Interprétation :** {interpretation}")
                        fig_bar = px.histogram(data, x=chi_var1, color=chi_var2, barmode="group", title=f"{chi_var1} vs {chi_var2}")
                        st.plotly_chart(fig_bar, use_container_width=True)

            if 'latitude' in data.columns and 'longitude' in data.columns:

                map_col = st.selectbox("Taille/Couleur", ["Aucune"] + list(data.columns), key="map_col_select")
                map_size = st.checkbox("Taille", key="map_size_checkbox") if map_col != "Aucune" else False
                fig_map = px.scatter_mapbox(data, lat="latitude", lon="longitude", 
                                          hover_name="nom_station" if "nom_station" in data.columns else None,
                                          size=map_col if map_size and map_col != "Aucune" else None, 
                                          color=map_col if not map_size and map_col != "Aucune" else None,
                                          zoom=10, height=600, title=f"Carte ({dataset_to_analyze})")

                fig_map.update_layout(mapbox_style="dark")
                st.plotly_chart(fig_map, use_container_width=True)

        with tab3:

            if len(quant_cols) > 1:
                corr_method = st.selectbox("Méthode de corrélation", ["pearson", "spearman"], key="corr_method_select")
                corr_mat = correlation_matrix(data, method=corr_method)
                if corr_mat is not None:
                    fig_corr = px.imshow(corr_mat, text_auto=".2f", aspect="equal", title=f"Matrice de Corrélation ({corr_method.capitalize()})")
                    st.plotly_chart(fig_corr, use_container_width=True)
            if len(qual_cols) > 1:
                chi2_mat = chi2_matrix(data)
                if chi2_mat is not None:
                    fig_chi2 = px.imshow(chi2_mat, text_auto=".3f", aspect="equal", title="Matrice Chi² (p-valeurs)")
                    st.plotly_chart(fig_chi2, use_container_width=True)

        with tab4:
            if len(data.columns) > 1:
                features = st.multiselect("Variables explicatives", data.columns, key="model_features_multiselect")
                available_targets = [col for col in data.columns if col not in features]
                target = st.selectbox("Variable cible", available_targets, key="model_target_select") if "Clustering" not in st.session_state.get('modeling_type', '') else None
                modeling_type = st.selectbox("Type de modélisation", ["Regression", "Classification", "Clustering"], key="modeling_type_select")

                model_options = {
                    "Regression": ["Linear Regression", "Polynomial Regression", "Decision Tree", "Random Forest", "XGBoost", "Stacking", "Voting"],
                    "Classification": ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression", "Stacking", "Voting"],
                    "Clustering": ["K-Means"]
                }

                model_type = st.selectbox("Modèle", model_options[modeling_type], key="model_type_select")
                # Les options de numérisation et normalisation ont été déplacées, donc elles ne sont plus ici
                if st.button("✅ Entraîner", key="train_model"):
                    backup_state()
                    # Par défaut, utiliser "Label Encoding" et "None" comme valeurs si non prétraité avant
                    model, metrics, processed_features, transformer = train_model(data, features, target if modeling_type != "Clustering" else None, 
                                                                                modeling_type, model_type, "Label Encoding", "None")
                    if model:
                        st.session_state['models'][dataset_to_analyze] = (model, processed_features, target if modeling_type != "Clustering" else None, 
                                                                        model_type, modeling_type, transformer)
                        st.write("**Métriques :**")
                        for k, v in metrics.items():
                            st.write(f"{k}: {v:.4f}")
                        X_processed, y_processed, _, _ = preprocess_data(data, features, target if modeling_type != "Clustering" else None, 
                                                                       "Label Encoding", "None", modeling_type, model_type)
                        if modeling_type == "Regression":
                            fig_pred = px.scatter(x=y_processed, y=model.predict(X_processed), labels={"x": "Réel", "y": "Prédit"}, title=f"Prédictions ({model_type})")
                            st.plotly_chart(fig_pred)
                        elif modeling_type == "Classification":
                            fig_pred = px.scatter(x=y_processed, y=model.predict(X_processed), labels={"x": "Réel", "y": "Prédit"}, title=f"Prédictions ({model_type})")
                            st.plotly_chart(fig_pred)
                        elif modeling_type == "Clustering":
                            fig_cluster = px.scatter(data_frame=X_processed, x=processed_features[0], 
                                                   y=processed_features[1] if len(processed_features) > 1 else processed_features[0], 
                                                   color=model.predict(X_processed), title="Clusters K-Means")
                            st.plotly_chart(fig_cluster)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                            joblib.dump(model, tmp.name)
                            with open(tmp.name, "rb") as f:
                                st.download_button("💾 Sauvegarder le modèle", data=f, file_name=f"{dataset_to_analyze}_{model_type}.pkl", key="download_model_button")
                            os.remove(tmp.name)

        if st.button("📝 Générer un rapport", key="generate_report"):
            with tempfile.TemporaryDirectory() as temp_dir:

                corr_mat = correlation_matrix(data) if len(quant_cols) > 1 else None
                chi2_mat = chi2_matrix(data) if len(qual_cols) > 1 else None
                try:
                    temp_pdf = create_temp_pdf(dataset_to_analyze, data, exploration, corr_mat, chi2_mat, test_results, test_interpretations, temp_dir)
                    report = generate_gemini_report(temp_pdf)
                    st.session_state['chat_history'] = [("", report)]
                    st.success("Rapport généré avec succès !")

                except Exception as e:
                    st.error(f"Erreur génération rapport: {str(e)}")


                if st.session_state['chat_history']:
                    st.markdown("<div class='section-box'><h2>📜 Rapport Généré</h2>", unsafe_allow_html=True)
                    report = display_progressive_report(st.session_state['chat_history'][-1][1])
                    pdf_bytes = create_downloadable_pdf(dataset_to_analyze, report)
                    st.download_button(
                        label="💾 Télécharger le Rapport en PDF",
                        data=pdf_bytes,
                        file_name=f"rapport_{dataset_to_analyze}.pdf",
                        mime="application/pdf",

                        key="download_report_button"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

