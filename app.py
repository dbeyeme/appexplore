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
import geopandas as gpd
import fiona
import os
import chardet
import tempfile
import shutil
import sqlalchemy
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
#import google.generativeai as genai
from google import genai
from fpdf import FPDF
import time
import pyarrow as pa
import unicodedata  

# Configuration du logging avec fichier
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])
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
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'test_results' not in st.session_state:
    st.session_state['test_results'] = {}

# Configuration de l'API Gemini (remplacez par votre clé)
API_KEY = "AIzaSyBJLhpSfKsbxgVEJwYmPSEZmaVlKt5qNlI"  # Remplacez par une clé valide
genai.configure(api_key=API_KEY)
client = genai.GenerativeModel('gemini-1.5-flash')

# CSS personnalisé pour un rendu professionnel
st.markdown("""
    <style>
    .stApp {
        background-color: #0A0F1A;
        font-family: 'Helvetica', 'Arial', sans-serif;
        color: #E6E6E6;
    }
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 500;
    }
    .stButton > button {
        background-color: #FFFFFF;
        color: #0A0F1A;
        border: 1px solid #333333;
        padding: 10px 20px;
        border-radius: 20px;
        font-size: 14px;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:hover {
        background-color: #E0E0E0;
        transform: scale(1.05);
    }
    .chat-message {
        background-color: #1A2634;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.4);
        font-family: 'Helvetica', 'Arial', sans-serif;
        color: #E6E6E6;
        line-height: 1.6;
        border-left: 4px solid #1DA1F2;
    }
    .section-box {
        background-color: #1A2634;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

class DataLoader:
    def detect_delimiter(self, content: str) -> str:
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
        try:
            content_bytes = None
            file_ext = None
            content_type = "application/octet-stream"
            total_size = 0
            temp_dir = None

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
                    status_text.text(f"Téléchargement terminé : {downloaded_size / 1024:.1f} Ko")

                file_ext = source.split('.')[-1].lower() if '.' in source.split('/')[-1] else None

            elif source_type == "file":
                if isinstance(source, list):
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
                        raise ValueError("Aucun fichier .shp trouvé.")
                    file_ext = 'shp'
                    total_size = sum(uploaded_file.size for uploaded_file in source)
                    status_text = st.empty()
                    status_text.text(f"Fichiers locaux chargés : {total_size / 1024:.1f} Ko")
                else:
                    content_bytes = source.read()
                    file_ext = source.name.split('.')[-1].lower() if '.' in source.name else None
                    total_size = len(content_bytes)
                    status_text = st.empty()
                    status_text.text(f"Fichier local chargé : {total_size / 1024:.1f} Ko")

            elif source_type == "db":
                engine = sqlalchemy.create_engine(source[0])
                df = pd.read_sql(source[1], engine)
                return df, None

            else:
                raise ValueError(f"Type de source '{source_type}' non supporté")

            if source_type != "db":
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
                    else:
                        detected = chardet.detect(content_bytes[:1024]) if content_bytes else {'confidence': 0}
                        if detected['confidence'] > 0.7 and detected['encoding']:
                            file_ext = 'csv'
                        else:
                            raise ValueError("Format non détectable")

                processed_content = content_bytes

                if file_ext == 'gz':
                    with st.spinner("Décompression GZIP..."):
                        processed_content = gzip.decompress(content_bytes)
                    file_ext = (source.split('.')[-2].lower() if source_type == "url" else source.name.split('.')[-2].lower()) if '.' in source else 'csv'
                    st.info(f"Fichier GZIP décompressé : {file_ext.upper()}")

                elif file_ext == 'zip':
                    with st.spinner("Décompression ZIP..."):
                        temp_dir = tempfile.mkdtemp()
                        with zipfile.ZipFile(io.BytesIO(content_bytes)) as z:
                            z.extractall(temp_dir)
                            compatible_files = [f for f in z.namelist() if f.split('.')[-1].lower() in ['csv', 'txt', 'json', 'geojson', 'xlsx', 'shp']]
                            if not compatible_files:
                                raise ValueError("Aucun fichier compatible trouvé dans le ZIP.")
                            file_name = compatible_files[0]
                            processed_content = open(os.path.join(temp_dir, file_name), 'rb').read()
                            file_ext = file_name.split('.')[-1].lower()
                            st.info(f"Fichier ZIP décompressé : {file_name} ({file_ext.upper()})")

                if file_ext == 'shp':
                    with st.spinner("Chargement du Shapefile..."):
                        if not temp_dir:
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
                                        with open(os.path.join(temp_dir, f'data.{ext}'), 'wb') as f:
                                            f.write(response.content)
                                    except:
                                        st.warning(f"Fichier .{ext} non trouvé.")
                        else:
                            shp_path = os.path.join(temp_dir, shp_file)
                        with fiona.Env(SHAPE_RESTORE_SHX='YES'):
                            gdf = gpd.read_file(shp_path)
                        if gdf.empty:
                            raise ValueError("Shapefile vide ou invalide.")
                        st.info(f"Shapefile chargé : {len(gdf)} entités.")
                        df = gdf.drop(columns=['geometry']).assign(
                            latitude=gdf.geometry.centroid.y,
                            longitude=gdf.geometry.centroid.x
                        )
                        return df, None

                content_str = None
                if file_ext in ['csv', 'txt', 'json', 'geojson']:
                    detected = chardet.detect(processed_content)
                    encoding = detected['encoding'] if detected['confidence'] > 0.7 else 'utf-8'
                    content_str = processed_content.decode(encoding, errors='replace')

                st.info(f"Chargement au format : {file_ext.upper()}")
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
            st.error(f"❌ Erreur : {str(e)}")
            logger.error(f"Erreur lors du chargement : {str(e)}", exc_info=True)
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
            "missing_percent": (df.isnull().sum() / len(df) * 100).to_dict(),
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
        st.error("Colonnes invalides ou données absentes.")
        return results
    if var1 == var2:
        st.warning("Les variables doivent être différentes pour une corrélation.")
        return results
    try:
        df_clean = df[[var1, var2]].dropna()
        if df[var1].dtype in [np.float64, np.int64] and df[var2].dtype in [np.float64, np.int64]:
            results["pearson"] = stats.pearsonr(df_clean[var1], df_clean[var2])[0]
            results["spearman"] = stats.spearmanr(df_clean[var1], df_clean[var2])[0]
        else:
            st.warning(f"Les colonnes {var1} ou {var2} ne sont pas numériques.")
    except Exception as e:
        st.error(f"Erreur : {str(e)}")
    return results

def chi2_test(df: pd.DataFrame, var1: str, var2: str) -> Dict[str, Any]:
    results = {}
    if df is None or var1 not in df.columns or var2 not in df.columns:
        st.error("Colonnes invalides ou données absentes.")
        return results
    try:
        contingency_table = pd.crosstab(df[var1], df[var2])
        chi2, p, dof, _ = stats.chi2_contingency(contingency_table)
        results["chi2"] = chi2
        results["p_value"] = p
        results["dof"] = dof
    except Exception as e:
        st.error(f"Erreur : {str(e)}")
    return results

def multivariate_tests(df: pd.DataFrame, group_var: str, value_var: str) -> Dict[str, Any]:
    results = {}
    if df is None or group_var not in df.columns or value_var not in df.columns:
        st.error("Colonnes invalides ou données absentes.")
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
    except Exception as e:
        st.error(f"Erreur : {str(e)}")
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
            if col1 == col2:
                chi2_pvals.loc[col1, col2] = 1.0
            else:
                result = chi2_test(df, col1, col2)
                chi2_pvals.loc[col1, col2] = result.get('p_value', 1.0)
    return chi2_pvals.astype(float)

def save_matrix_to_image(matrix: pd.DataFrame, title: str, filename: str, method: str = None):
    plt.figure(figsize=(8, 6))
    if method == 'chi2':
        sns.heatmap(matrix, annot=True, cmap='Blues', vmin=0, vmax=1, fmt='.3f')
    else:
        sns.heatmap(matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f')
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_temp_pdf(dataset_name: str, data: pd.DataFrame, exploration: Dict, corr_mat: pd.DataFrame, chi2_mat: pd.DataFrame, test_results: Dict) -> str:
    temp_pdf = f"temp_{dataset_name}_{int(time.time())}.pdf"
    doc = SimpleDocTemplate(temp_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Exploration du dataset : {dataset_name}", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Métadonnées", styles['Heading2']))
    meta_text = "<br/>".join([f"{k}: {v}" for k, v in exploration["metadata"].items()])
    story.append(Paragraph(meta_text, styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Doublons : {exploration['duplicates']}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Valeurs manquantes", styles['Heading2']))
    missing_text = "<br/>".join([f"{k}: {v} ({exploration['missing_percent'][k]:.2f}%)" for k, v in exploration["missing_values"].items()])
    story.append(Paragraph(missing_text, styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Statistiques descriptives", styles['Heading2']))
    desc_text = exploration["description"].to_string()
    story.append(Paragraph(desc_text.replace('\n', '<br/>'), styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Valeurs aberrantes (IQR)", styles['Heading2']))
    outlier_text = "<br/>".join([f"{k}: {v}" for k, v in exploration["outliers"].items()])
    story.append(Paragraph(outlier_text, styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Résultats des tests statistiques", styles['Heading2']))
    if "correlation" in test_results:
        corr = test_results["correlation"]
        corr_text = f"Corrélation entre {corr['var1']} et {corr['var2']}:<br/>" \
                    f"Pearson: {corr['pearson']:.3f}<br/>Spearman: {corr['spearman']:.3f}"
        story.append(Paragraph(corr_text, styles['Normal']))
        story.append(Spacer(1, 12))
    
    if "chi2" in test_results:
        chi2 = test_results["chi2"]
        chi2_text = f"Test Chi² entre {chi2['var1']} et {chi2['var2']}:<br/>" \
                    f"Chi²: {chi2['chi2']:.2f}<br/>p-valeur: {chi2['p_value']:.4f}<br/>Degrés de liberté: {chi2['dof']}"
        story.append(Paragraph(chi2_text, styles['Normal']))
        story.append(Spacer(1, 12))
    
    if "multivariate" in test_results:
        multi = test_results["multivariate"]
        multi_text = f"Tests multivariables entre {multi['group_var']} et {multi['value_var']}:<br/>" \
                     f"ANOVA: F={multi['anova'][0]:.2f}, p={multi['anova'][1]:.4f}<br/>" \
                     f"Kruskal-Wallis: H={multi['kruskal'][0]:.2f}, p={multi['kruskal'][1]:.4f}<br/>" \
                     f"Eta²: {multi['eta_squared']:.4f}<br/>Kurtosis: {multi['kurtosis']:.2f}<br/>Skewness: {multi['skewness']:.2f}"
        story.append(Paragraph(multi_text, styles['Normal']))
        story.append(Spacer(1, 12))

    if corr_mat is not None:
        corr_img = f"temp_corr_{dataset_name}.png"
        save_matrix_to_image(corr_mat, "Matrice de Corrélation (Pearson)", corr_img)
        story.append(Image(corr_img, width=400, height=300))
        story.append(Spacer(1, 12))

    if chi2_mat is not None:
        chi2_img = f"temp_chi2_{dataset_name}.png"
        save_matrix_to_image(chi2_mat, "Matrice de Dépendance (p-valeurs Chi²)", chi2_img, method='chi2')
        story.append(Image(chi2_img, width=400, height=300))
        story.append(Spacer(1, 12))

    doc.build(story)
    return temp_pdf

def generate_gemini_report(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    
    prompt = """
    Analyse et résume ce rapport d'exploration de données en français. Fournis un résumé clair, structuré et informatif des principales conclusions, rédigé dans un style professionnel et naturel, comme si tu étais un analyste de données expérimenté. Évite un ton mécanique ou trop technique, et privilégie une présentation accessible et humaine. Structure le texte en paragraphes avec des titres pour chaque section. N'utilise pas les caractères Markdown comme ## ou **.
    """
    
    try:
        response = client.generate_content(
            contents=[
                {
                    "mime_type": "application/pdf",
                    "data": pdf_data
                },
                prompt
            ]
        )
        report = re.sub(r'[#*]+', '', response.text)
        return report
    except Exception as e:
        logger.error(f"Erreur lors de la génération Gemini : {str(e)}")
        return "Erreur lors de l'analyse Gemini. Vérifiez la clé API ou le fichier temporaire."

def display_progressive_report(report: str) -> str:
    """Affiche le rapport progressivement dans le tchat avec un style professionnel."""
    lines = report.split('\n')
    chat_container = st.empty()
    current_text = ""
    
    for i, line in enumerate(lines):
        if line.strip():  # Ignorer les lignes vides
            # Détecter les titres (lignes suivies ou précédées d'une ligne vide)
            is_title = (i == 0) or (i > 0 and not lines[i-1].strip()) or (i < len(lines)-1 and not lines[i+1].strip())
            if is_title:
                # Titre en bleu foncé (#1DA1F2), gras, taille plus grande
                formatted_line = f'<div style="color: #E6E6E6; font-weight: bold; font-size: 18px; margin-bottom: 10px;">{line}</div>'
            else:
                # Texte normal en blanc cassé, chiffres en orange (#FF4500) et gras
                formatted_line = re.sub(r'(\d+\.?\d*)', r'<span style="color: #FF4500; font-weight: bold;">\1</span>', line)
                formatted_line = f'<div style="color: #E6E6E6; font-size: 14px; margin-bottom: 5px;">{formatted_line}</div>'
            current_text += formatted_line
            chat_container.markdown(f'<div class="chat-message">{current_text}</div>', unsafe_allow_html=True)
            time.sleep(0.3)  # Délai de 0.3s pour un affichage plus fluide
    return report  # Retourne le texte brut pour le PDF

def create_downloadable_pdf(dataset_name: str, report_text: str) -> bytes:
    """Génère un PDF avec un rendu professionnel."""
    pdf = FPDF()
    pdf.add_page()
    
    # Utilisation de Helvetica (police standard intégrée à FPDF)
    pdf.set_font("Helvetica", size=12)
    
    if not report_text.strip():
        logger.error("Le texte du rapport est vide.")
        report_text = "Aucun contenu disponible pour ce rapport."
        st.error("Le rapport est vide.")

    # Normalisation du texte pour gérer les caractères Unicode
    normalized_text = unicodedata.normalize('NFKD', report_text).encode('ASCII', 'ignore').decode('ASCII')
    normalized_text = normalized_text.replace("'", "'").replace(""", '"').replace(""", '"')

    # En-tête du PDF
    pdf.set_text_color(29, 161, 242)  # Noir
    pdf.set_font("Helvetica", "B", size=12)
    pdf.cell(0, 10, f"Rapport d'analyse : {dataset_name}", ln=True, align="C")
    pdf.ln(10)

    # Contenu du rapport
    lines = normalized_text.split('\n')
    for i, line in enumerate(lines):
        if line.strip():
            is_title = (i == 0) or (i > 0 and not lines[i-1].strip()) or (i < len(lines)-1 and not lines[i+1].strip())
            if is_title:
                pdf.set_text_color(0, 0, 0)  # Bleu (#1DA1F2)
                pdf.set_font("Helvetica", "B", size=12)
                pdf.multi_cell(0, 10, line)
                pdf.ln(5)
            else:
                pdf.set_text_color(0, 0, 0)  # Noir
                pdf.set_font("Helvetica", size=10)
                # Mettre les chiffres en gras
                parts = re.split(r'(\d+\.?\d*)', line)
                for part in parts:
                    if re.match(r'\d+\.?\d*', part):
                        pdf.set_font("Helvetica", "B", size=10)
                        pdf.write(10, part)
                    else:
                        pdf.set_font("Helvetica", size=10)
                        pdf.write(10, part)
                pdf.ln(10)

    # Générer le PDF et s'assurer que le résultat est de type bytes
    pdf_output = pdf.output(dest="S")  # Retourne une chaîne ou un bytearray selon la version de FPDF
    
    # Convertir en bytes si nécessaire
    if isinstance(pdf_output, bytearray):
        pdf_output = bytes(pdf_output)
    elif isinstance(pdf_output, str):
        # Si c'est une chaîne, on l'encode en bytes (latin-1 est utilisé par FPDF en interne)
        pdf_output = pdf_output.encode('latin-1', errors='replace')
    
    logger.info("PDF généré avec succès.")
    return pdf_output

def safe_dataframe_display(df: pd.DataFrame):
    df_safe = df.copy()
    for col in df_safe.select_dtypes(include=['object']).columns:
        df_safe[col] = df_safe[col].astype(str)
    return df_safe

class DataPipeline:
    def __init__(self):
        self.loader = DataLoader()
        self.explorer = DataExplorer()
    
    def process(self, source: str, source_type: str, name: str, skip_header: bool = False) -> Tuple[pd.DataFrame, Dict, Dict]:
        with st.spinner(f"Chargement du dataset '{name}'..."):
            data, json_data = self.loader.load(source, source_type, skip_header)
            if data is not None:
                exploration = self.explorer.explore(data)
                return data, exploration, None
            return None, None, json_data

def main():
    st.title("🔍 Exploration et Analyse de Données")
    pipeline = DataPipeline()

    with st.sidebar:
        st.header("📥 Chargement des Données")
        st.info("Chargez vos données ici (ex. : CSV via URL, fichiers locaux ou requête SQL).")
        
        source_type = st.selectbox("Source", ["URL", "Fichiers Locaux", "Base de Données"], key="source_type_global")
        
        if source_type == "URL":
            num_datasets = st.number_input("Nombre de datasets", min_value=1, value=1, step=1, key="num_datasets_input")
            for i in range(num_datasets):
                st.subheader(f"Dataset {i+1}")
                source = st.text_input(f"URL {i+1}", "", key=f"source_url_{i}")
                name = st.text_input(f"Nom {i+1}", f"dataset_{i+1}", key=f"name_{i}")
                skip_header = st.checkbox(f"Ignorer la première ligne {i+1}", value=False, key=f"skip_header_{i}")
                if st.button(f"📤 Charger {i+1}", key=f"load_{i}") and source and name:
                    data, exploration, json_data = pipeline.process(source, "url", name, skip_header)
                    if data is not None:
                        st.session_state['datasets'][name] = data
                        st.session_state['explorations'][name] = exploration
                        st.session_state['sources'][name] = (source, "url", skip_header)
                        st.session_state['test_results'][name] = {}
                        st.success(f"✅ '{name}' chargé ({len(data)} lignes)")
                    elif json_data:
                        st.session_state['json_data'][name] = json_data
                        st.session_state['sources'][name] = (source, "url", skip_header)
                        st.info(f"📋 JSON '{name}' chargé.")

        elif source_type == "Fichiers Locaux":
            uploaded_files = st.file_uploader("Importer des fichiers", 
                                             type=["csv", "xlsx", "json", "geojson", "txt", "gz", "zip", "shp", "shx", "dbf"], 
                                             accept_multiple_files=True, key="multi_upload")
            skip_header = st.checkbox("Ignorer la première ligne (CSV/TXT)", value=False, key="multi_skip_header")
            if uploaded_files and st.button("📤 Charger Tous", key="load_all"):
                with st.spinner("Chargement des fichiers..."):
                    for uploaded_file in uploaded_files:
                        name = uploaded_file.name.split('.')[0]
                        data, exploration, json_data = pipeline.process(uploaded_file, "file", name, skip_header)
                        if data is not None:
                            st.session_state['datasets'][name] = data
                            st.session_state['explorations'][name] = exploration
                            st.session_state['sources'][name] = (uploaded_file, "file", skip_header)
                            st.session_state['test_results'][name] = {}
                            st.success(f"✅ '{name}' chargé ({len(data)} lignes)")
                        elif json_data:
                            st.session_state['json_data'][name] = json_data
                            st.session_state['sources'][name] = (uploaded_file, "file", skip_header)
                            st.info(f"📋 JSON '{name}' chargé.")

        elif source_type == "Base de Données":
            db_url = st.text_input("URL de la base (ex. sqlite:///path.db)", key="db_url")
            query = st.text_area("Requête SQL", "SELECT * FROM table_name", key="db_query")
            name = st.text_input("Nom du dataset", "db_dataset", key="db_name")
            if st.button("📤 Charger", key="load_db"):
                with st.spinner("Connexion à la base..."):
                    data, exploration, json_data = pipeline.process((db_url, query), "db", name)
                    if data is not None:
                        st.session_state['datasets'][name] = data
                        st.session_state['explorations'][name] = exploration
                        st.session_state['sources'][name] = (query, "db", False)
                        st.session_state['test_results'][name] = {}
                        st.success(f"✅ '{name}' chargé ({len(data)} lignes)")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Actualiser", key="refresh"):
                with st.spinner("Actualisation..."):
                    for name, (source, source_type, skip_header) in st.session_state['sources'].items():
                        data, exploration, json_data = pipeline.process(source, source_type, name, skip_header)
                        if data is not None:
                            st.session_state['datasets'][name] = data
                            st.session_state['explorations'][name] = exploration
                            st.session_state['json_data'][name] = None
                            st.session_state['test_results'][name] = {}
                        else:
                            st.session_state['json_data'][name] = json_data
                    st.success("✅ Datasets actualisés")
        with col2:
            if st.button("🗑️ Réinitialiser", key="reset"):
                with st.spinner("Réinitialisation..."):
                    st.session_state.clear()
                    st.session_state['datasets'] = {}
                    st.session_state['explorations'] = {}
                    st.session_state['sources'] = {}
                    st.session_state['json_data'] = {}
                    st.session_state['chat_history'] = []
                    st.session_state['test_results'] = {}
                    st.rerun()

    with st.sidebar:
        st.header("⚙️ Gestion des Datasets")
        st.info("Sélectionnez et modifiez vos datasets ici.")
        if st.session_state['datasets']:
            dataset_names = list(st.session_state['datasets'].keys())
            selected_datasets = st.multiselect("Datasets à traiter", dataset_names, default=dataset_names[0] if dataset_names else None, key="select_datasets")
            
            if selected_datasets:
                with st.expander("🔄 Conversion des Types"):
                    st.info("Convertissez le type de données d'une colonne.")
                    col_to_convert = st.multiselect("Colonnes", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="convert_cols")
                    type_to_convert = st.selectbox("Type", 
                                                  ["Entier (int)", "Décimal (float)", "Catégorie (category)", 
                                                   "Date (datetime)", "Timestamp vers Date", "Extraire Mois", 
                                                   "Extraire Jour de la semaine", "Extraire Jour du mois", "Extraire Heure", "Extraire Année"], 
                                                  key="convert_type")
                    if st.button("✅ Appliquer", key="apply_convert"):
                        with st.spinner("Conversion..."):
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
                                            st.error(f"❌ Erreur : {str(e)}")

                with st.expander("🧹 Nettoyage des Valeurs"):
                    st.info("Modifiez le contenu des colonnes.")
                    col_to_clean = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="clean_col")
                    action = st.selectbox("Action", ["Supprimer des caractères", "Remplacer des caractères"], key="clean_action")
                    pattern = st.text_input("Motif (regex)", key="clean_pattern")
                    replacement = st.text_input("Remplacement", "", key="clean_replace") if action == "Remplacer des caractères" else ""
                    if st.button("✅ Appliquer", key="apply_clean"):
                        with st.spinner("Nettoyage..."):
                            for ds in selected_datasets:
                                data = st.session_state['datasets'][ds].copy()
                                if col_to_clean in data.columns:
                                    try:
                                        if action == "Supprimer des caractères":
                                            data[col_to_clean] = data[col_to_clean].apply(lambda x: re.sub(pattern, '', str(x)) if pd.notnull(x) else x)
                                        else:
                                            data[col_to_clean] = data[col_to_clean].apply(lambda x: re.sub(pattern, replacement, str(x)) if pd.notnull(x) else x)
                                        st.session_state['datasets'][ds] = data
                                        st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                        st.success(f"✅ Nettoyage appliqué pour '{ds}'")
                                    except Exception as e:
                                        st.error(f"❌ Erreur : {str(e)}")

                with st.expander("🗑️ Suppression de Colonnes"):
                    st.info("Supprimez des colonnes inutiles.")
                    cols_to_drop = st.multiselect("Colonnes", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="drop_cols")
                    if st.button("✅ Supprimer", key="apply_drop"):
                        with st.spinner("Suppression..."):
                            for ds in selected_datasets:
                                data = st.session_state['datasets'][ds].copy()
                                cols_present = [col for col in cols_to_drop if col in data.columns]
                                if cols_present:
                                    data = data.drop(columns=cols_present)
                                    st.session_state['datasets'][ds] = data
                                    st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                    st.success(f"✅ Colonnes supprimées pour '{ds}'")

                with st.expander("🚫 Suppression de Lignes"):
                    st.info("Supprimez des lignes selon une condition.")
                    col_to_filter = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="filter_col")
                    filter_expr = st.text_input("Expression (ex. 'Station_1')", key="filter_expr")
                    filter_type = st.selectbox("Type", ["Valeur exacte", "Regex"], key="filter_type")
                    if st.button("✅ Supprimer", key="apply_filter"):
                        with st.spinner("Suppression..."):
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
                    st.info("Ajoutez une nouvelle colonne.")
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
                    if st.button("✅ Créer", key="apply_new_col"):
                        with st.spinner("Création..."):
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
                                        else:
                                            if new_col_replace:
                                                data[new_col_name] = data[base_col].apply(lambda x: re.sub(new_col_pattern, new_col_replace, str(x)) if pd.notnull(x) else x)
                                            else:
                                                data[new_col_name] = data[base_col].apply(lambda x: re.sub(new_col_pattern, '', str(x)) if pd.notnull(x) else x)
                                        st.session_state['datasets'][ds] = data
                                        st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                        st.success(f"✅ '{new_col_name}' créé pour '{ds}'")
                                    except Exception as e:
                                        st.error(f"❌ Erreur : {str(e)}")

                with st.expander("🕳️ Traitement des Valeurs Manquantes"):
                    st.info("Gérez les valeurs manquantes.")
                    col_to_fill = st.selectbox("Colonne", ["Toutes"] + list(set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets])), key="fill_col")
                    fill_method = st.selectbox("Méthode", ["Supprimer les lignes", "Supprimer toutes lignes vides", "Remplacer par moyenne", "Remplacer par mode", "Plus proche voisin"], key="fill_method")
                    if st.button("✅ Traiter", key="apply_fill"):
                        with st.spinner("Traitement..."):
                            for ds in selected_datasets:
                                data = st.session_state['datasets'][ds].copy()
                                if col_to_fill == "Toutes":
                                    if fill_method == "Supprimer toutes lignes vides":
                                        data = data.dropna(how='any')
                                    else:
                                        st.warning("Sélectionnez 'Supprimer toutes lignes vides' pour agir sur toutes les colonnes.")
                                elif col_to_fill in data.columns:
                                    if fill_method == "Supprimer les lignes":
                                        data = data.dropna(subset=[col_to_fill])
                                    elif fill_method == "Remplacer par moyenne":
                                        if data[col_to_fill].dtype in [np.float64, np.int64]:
                                            data[col_to_fill] = data[col_to_fill].fillna(data[col_to_fill].mean())
                                        else:
                                            st.warning(f"⚠️ '{col_to_fill}' n'est pas numérique.")
                                    elif fill_method == "Remplacer par mode":
                                        mode = data[col_to_fill].mode()
                                        data[col_to_fill] = data[col_to_fill].fillna(mode[0] if not mode.empty else None)
                                    elif fill_method == "Plus proche voisin":
                                        data[col_to_fill] = data[col_to_fill].interpolate(method='nearest').ffill().bfill()
                                st.session_state['datasets'][ds] = data
                                st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                st.success(f"✅ Valeurs manquantes traitées pour '{ds}'")

                with st.expander("⚠️ Traitement des Valeurs Aberrantes"):
                    st.info("Corrigez ou supprimez les valeurs extrêmes.")
                    col_to_outlier = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].select_dtypes(include=np.number).columns) for ds in selected_datasets]), key="outlier_col")
                    outlier_method = st.selectbox("Méthode", ["Supprimer (IQR)", "Remplacer par médiane", "Limiter (IQR)"], key="outlier_method")
                    if st.button("✅ Traiter", key="apply_outlier"):
                        with st.spinner("Traitement..."):
                            for ds in selected_datasets:
                                data = st.session_state['datasets'][ds].copy()
                                if col_to_outlier in data.columns and data[col_to_outlier].dtype in [np.float64, np.int64]:
                                    Q1, Q3 = data[col_to_outlier].quantile([0.25, 0.75])
                                    IQR = Q3 - Q1
                                    lower_bound = Q1 - 1.5 * IQR
                                    upper_bound = Q3 + 1.5 * IQR
                                    if outlier_method == "Supprimer (IQR)":
                                        data = data[(data[col_to_outlier] >= lower_bound) & (data[col_to_outlier] <= upper_bound)]
                                    elif outlier_method == "Remplacer par médiane":
                                        data[col_to_outlier] = np.where((data[col_to_outlier] < lower_bound) | (data[col_to_outlier] > upper_bound), 
                                                                        data[col_to_outlier].median(), data[col_to_outlier])
                                    elif outlier_method == "Limiter (IQR)":
                                        data[col_to_outlier] = data[col_to_outlier].clip(lower=lower_bound, upper=upper_bound)
                                    st.session_state['datasets'][ds] = data
                                    st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                                    st.success(f"✅ Aberrants traités pour '{ds}'")

                with st.expander("🔗 Jointure de Datasets"):
                    st.info("Fusionnez deux datasets.")
                    if len(dataset_names) > 1:
                        left_ds = st.selectbox("Dataset principal", selected_datasets, key="join_left_ds")
                        right_ds = st.selectbox("Dataset secondaire", dataset_names, key="join_right_ds")
                        col_main = st.selectbox("Colonne principale", st.session_state['datasets'][left_ds].columns, key="join_col_main")
                        col_second = st.selectbox("Colonne secondaire", st.session_state['datasets'][right_ds].columns, key="join_col_second")
                        join_type = st.selectbox("Type", ["inner", "left", "right", "outer"], key="join_type")
                        if st.button("✅ Joindre", key="apply_join"):
                            with st.spinner("Jointure..."):
                                data = st.session_state['datasets'][left_ds].merge(st.session_state['datasets'][right_ds], how=join_type, left_on=col_main, right_on=col_second)
                                new_name = f"{left_ds}_joined_{right_ds}"
                                st.session_state['datasets'][new_name] = data
                                st.session_state['explorations'][new_name] = pipeline.explorer.explore(data)
                                st.session_state['test_results'][new_name] = {}
                                st.success(f"✅ Jointure effectuée : '{new_name}'")

    if st.session_state['datasets']:
        st.markdown("<div class='section-box'><h2>📋 Aperçu des Données</h2>", unsafe_allow_html=True)
        selected_preview = st.selectbox("Choisir un dataset", list(st.session_state['datasets'].keys()), key="preview_dataset")
        st.dataframe(safe_dataframe_display(st.session_state['datasets'][selected_preview].head(10)), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-box'><h2>📊 Exploration et Visualisation</h2>", unsafe_allow_html=True)
        dataset_to_analyze = st.selectbox("Dataset à analyser", list(st.session_state['datasets'].keys()), key="analyze_dataset")
        if dataset_to_analyze:
            data = st.session_state['datasets'][dataset_to_analyze]
            exploration = st.session_state['explorations'][dataset_to_analyze]
            test_results = st.session_state['test_results'][dataset_to_analyze]
            
            st.write("**Aperçu des données**")
            st.dataframe(safe_dataframe_display(data.head()), use_container_width=True)
            
            with st.expander("📈 Exploration"):
                st.info("Découvrez les détails de votre dataset.")
                st.write(f"**Nom du dataset :** {dataset_to_analyze}")
                st.write("**Métadonnées :**")
                st.dataframe(safe_dataframe_display(pd.DataFrame(exploration["metadata"].items(), columns=["Colonne", "Type"])), use_container_width=True)
                st.write(f"**Doublons :** {exploration['duplicates']}")
                st.write("**Valeurs manquantes :**")
                missing_df = pd.DataFrame({
                    "Colonne": exploration["missing_values"].keys(),
                    "Nb manquants": exploration["missing_values"].values(),
                    "Pourcentage (%)": [f"{exploration['missing_percent'][col]:.2f}" for col in exploration["missing_values"]]
                })
                if missing_df["Nb manquants"].sum() > 0:
                    st.dataframe(safe_dataframe_display(missing_df[missing_df["Nb manquants"] > 0]), use_container_width=True)
                else:
                    st.write("Aucune valeur manquante")
                st.write("**Statistiques descriptives :**")
                st.dataframe(safe_dataframe_display(exploration["description"]), use_container_width=True)
                st.write("**Valeurs aberrantes (IQR) :**")
                st.dataframe(safe_dataframe_display(pd.DataFrame(exploration["outliers"].items(), columns=["Colonne", "Nb aberrants"])), use_container_width=True)
            
            csv = data.to_csv(index=False)
            st.download_button(label="💾 Télécharger en CSV", data=csv, file_name=f"{dataset_to_analyze}.csv", mime="text/csv", key="download_csv")
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Histogrammes", "🔗 Corrélations", "📉 Tests Multivariables", 
                                                               "🗺️ Cartes Spatiales", "📌 Qualitatives", "🔢 Matrice de Corrélation", 
                                                               "🔍 Matrice de Dépendance"])
            
            quant_cols = data.select_dtypes(include=[np.number]).columns
            qual_cols = data.select_dtypes(include=['object', 'category']).columns
            
            with tab1:
                st.info("Visualisez la répartition d’une colonne numérique.")
                if len(quant_cols) > 0:
                    col_hist = st.selectbox("Colonne", quant_cols, key="hist_select")
                    with st.spinner("Génération..."):
                        fig_hist = px.histogram(data, x=col_hist, title=f"Distribution de {col_hist}", nbins=50)
                        st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.write("Aucune colonne quantitative disponible.")
            
            with tab2:
                st.info("Analysez la corrélation entre deux variables numériques.")
                if len(quant_cols) > 1:
                    with st.form(key='corr_form'):
                        col1, col2 = st.columns(2)
                        with col1:
                            var1 = st.selectbox("Variable 1", quant_cols, key="var1_select")
                        with col2:
                            var2 = st.selectbox("Variable 2", quant_cols, index=1, key="var2_select")
                        submit_corr = st.form_submit_button(label="✅ Calculer")
                    
                    if submit_corr:
                        with st.spinner("Calcul..."):
                            corr = correlation(data, var1, var2)
                            test_results["correlation"] = {"var1": var1, "var2": var2, **corr}
                            st.session_state['test_results'][dataset_to_analyze] = test_results
                            st.write("### Résultats")
                            pearson = corr.get('pearson', 'N/A')
                            spearman = corr.get('spearman', 'N/A')
                            st.write(f"**Pearson :** {pearson:.3f}" if isinstance(pearson, float) else f"**Pearson :** {pearson}")
                            st.write(f"**Spearman :** {spearman:.3f}" if isinstance(spearman, float) else f"**Spearman :** {spearman}")
                            if 'pearson' in corr:
                                fig_scatter = px.scatter(data, x=var1, y=var2, trendline="ols", title=f"Corrélation : {var1} vs {var2}")
                                st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.write("Pas assez de colonnes quantitatives.")
            
            with tab3:
                st.info("Effectuez des tests statistiques.")
                if len(qual_cols) > 0 and len(quant_cols) > 0:
                    with st.form(key='multi_form'):
                        col1, col2 = st.columns(2)
                        with col1:
                            group_var = st.selectbox("Variable qualitative", qual_cols, key="group_select")
                        with col2:
                            value_var = st.selectbox("Variable quantitative", quant_cols, key="value_select")
                        submit_multi = st.form_submit_button(label="✅ Effectuer")
                    
                    if submit_multi:
                        with st.spinner("Calcul..."):
                            multi = multivariate_tests(data, group_var, value_var)
                            test_results["multivariate"] = {"group_var": group_var, "value_var": value_var, **multi}
                            st.session_state['test_results'][dataset_to_analyze] = test_results
                            st.write("### Résultats")
                            anova = multi.get('anova', [None, None])
                            kruskal = multi.get('kruskal', [None, None])
                            eta_squared = multi.get('eta_squared', 'N/A')
                            kurtosis = multi.get('kurtosis', 'N/A')
                            skewness = multi.get('skewness', 'N/A')
                            st.write(f"**ANOVA :** F={anova[0]:.2f}, p={anova[1]:.4f}" if anova[0] else "**ANOVA :** N/A")
                            st.write(f"**Kruskal-Wallis :** H={kruskal[0]:.2f}, p={kruskal[1]:.4f}" if kruskal[0] else "**Kruskal-Wallis :** N/A")
                            st.write(f"**Eta² :** {eta_squared:.4f}" if isinstance(eta_squared, float) else f"**Eta² :** {eta_squared}")
                            st.write(f"**Kurtosis :** {kurtosis:.2f}" if isinstance(kurtosis, float) else f"**Kurtosis :** {kurtosis}")
                            st.write(f"**Skewness :** {skewness:.2f}" if isinstance(skewness, float) else f"**Skewness :** {skewness}")
                            if 'anova' in multi:
                                fig_box = px.box(data, x=group_var, y=value_var, title=f"Distribution de {value_var} par {group_var}")
                                st.plotly_chart(fig_box, use_container_width=True)
                
                if len(qual_cols) > 1:
                    st.subheader("Test de Dépendance (Chi²)")
                    with st.form(key='chi2_form'):
                        col1, col2 = st.columns(2)
                        with col1:
                            chi_var1 = st.selectbox("Variable qualitative 1", qual_cols, key="chi_var1")
                        with col2:
                            chi_var2 = st.selectbox("Variable qualitative 2", qual_cols, index=1, key="chi_var2")
                        submit_chi2 = st.form_submit_button(label="✅ Effectuer")
                    if submit_chi2 and chi_var1 != chi_var2:
                        with st.spinner("Calcul..."):
                            chi2_results = chi2_test(data, chi_var1, chi_var2)
                            test_results["chi2"] = {"var1": chi_var1, "var2": chi_var2, **chi2_results}
                            st.session_state['test_results'][dataset_to_analyze] = test_results
                            st.write(f"**Chi² :** {chi2_results.get('chi2', 'N/A'):.2f}")
                            st.write(f"**p-valeur :** {chi2_results.get('p_value', 'N/A'):.4f}")
                            st.write(f"**Degrés de liberté :** {chi2_results.get('dof', 'N/A')}")
                            fig_bar = px.histogram(data, x=chi_var1, color=chi_var2, barmode="group", title=f"{chi_var1} vs {chi_var2}")
                            st.plotly_chart(fig_bar, use_container_width=True)
            
            with tab4:
                st.info("Affichez vos données géographiques.")
                if 'latitude' in data.columns and 'longitude' in data.columns:
                    map_col = st.selectbox("Taille/Couleur", ["Aucune"] + list(data.columns), key="map_col")
                    map_size = st.checkbox("Taille", key="map_size") if map_col != "Aucune" else False
                    with st.spinner("Génération..."):
                        fig_map = px.scatter_mapbox(
                            data, lat="latitude", lon="longitude", 
                            hover_name="nom_station" if "nom_station" in data.columns else None,
                            size=map_col if map_size and map_col != "Aucune" else None,
                            color=map_col if not map_size and map_col != "Aucune" else None,
                            zoom=10, height=600, title=f"Carte ({dataset_to_analyze})"
                        )
                        fig_map.update_layout(mapbox_style="dark")
                        st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.write("Colonnes 'latitude' et 'longitude' requises.")

            with tab5:
                st.info("Visualisez la répartition des catégories.")
                if len(qual_cols) > 0:
                    qual_col = st.selectbox("Variable qualitative", qual_cols, key="qual_select")
                    with st.spinner("Génération..."):
                        fig_bar = px.histogram(data, x=qual_col, title=f"Répartition de {qual_col}", color=qual_col)
                        st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.write("Aucune colonne qualitative disponible.")

            with tab6:
                st.info("Visualisez les corrélations entre variables numériques.")
                if len(quant_cols) > 1:
                    corr_method = st.selectbox("Méthode", ["pearson", "spearman"], key="corr_method")
                    with st.spinner("Génération..."):
                        corr_mat = correlation_matrix(data, method=corr_method)
                        if corr_mat is not None:
                            fig_corr = px.imshow(corr_mat, text_auto=".2f", aspect="equal", 
                                                title=f"Matrice de Corrélation ({corr_method.capitalize()})",
                                                color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                            st.plotly_chart(fig_corr, use_container_width=True)
                        else:
                            st.write("Pas assez de colonnes numériques.")
                else:
                    st.write("Pas assez de colonnes quantitatives.")

            with tab7:
                st.info("Visualisez les dépendances entre variables qualitatives.")
                if len(qual_cols) > 1:
                    with st.spinner("Génération..."):
                        chi2_mat = chi2_matrix(data)
                        if chi2_mat is not None:
                            fig_chi2 = px.imshow(chi2_mat, text_auto=".3f", aspect="equal", 
                                                title="Matrice de Dépendance (p-valeurs Chi²)",
                                                color_continuous_scale="Blues", zmin=0, zmax=1)
                            st.plotly_chart(fig_chi2, use_container_width=True)
                            st.write("**Légende :** p < 0.05 (bleu foncé) = dépendance significative.")
                        else:
                            st.write("Pas assez de colonnes qualitatives.")
                else:
                    st.write("Pas assez de colonnes qualitatives.")

            # Initialisation des chemins d'image pour éviter UnboundLocalError
            corr_img_path = None
            chi2_img_path = None

            if st.button("📝 Générer un rapport", key="generate_report"):
                with st.spinner("Création du fichier temporaire..."):
                    corr_mat = correlation_matrix(data, method='pearson') if len(quant_cols) > 1 else None
                    chi2_mat = chi2_matrix(data) if len(qual_cols) > 1 else None
                    temp_pdf = create_temp_pdf(dataset_to_analyze, data, exploration, corr_mat, chi2_mat, test_results)
                    logger.info(f"Fichier temporaire créé : {temp_pdf}")
                
                with st.spinner("Analyse des données et rédaction du rapport..."):
                    report = generate_gemini_report(temp_pdf)
                    st.session_state['chat_history'] = [("", report)]
                
                os.remove(temp_pdf)
                logger.info(f"Fichier temporaire supprimé : {temp_pdf}")

                corr_img_path = f"temp_corr_{dataset_to_analyze}.png" if corr_mat is not None and os.path.exists(f"temp_corr_{dataset_to_analyze}.png") else None
                chi2_img_path = f"temp_chi2_{dataset_to_analyze}.png" if chi2_mat is not None and os.path.exists(f"temp_chi2_{dataset_to_analyze}.png") else None

            if st.session_state['chat_history']:
                st.markdown("<h3>💬 Rapport d’analyse</h3>", unsafe_allow_html=True)
                report_text = display_progressive_report(st.session_state['chat_history'][0][1])
                
                pdf_data = create_downloadable_pdf(dataset_to_analyze, report_text)
                st.download_button(
                    label="📄 Télécharger le rapport en PDF",
                    data=pdf_data,
                    file_name=f"rapport_{dataset_to_analyze}.pdf",
                    mime="application/pdf",
                    key="download_report"
                )

                # Nettoyage des fichiers temporaires uniquement si définis et existants
                if corr_img_path and os.path.exists(corr_img_path):
                    os.remove(corr_img_path)
                    logger.info(f"Fichier temporaire supprimé : {corr_img_path}")
                if chi2_img_path and os.path.exists(chi2_img_path):
                    os.remove(chi2_img_path)
                    logger.info(f"Fichier temporaire supprimé : {chi2_img_path}")

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
