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
import google.generativeai as genai
from fpdf import FPDF
import time
import unicodedata
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(user)s - %(message)s',
                    handlers=[logging.handlers.RotatingFileHandler("activity.log", maxBytes=10**6, backupCount=5), logging.StreamHandler()])
logger = logging.getLogger()
logger.setLevel(logging.INFO)
user_id = "user_default"
extra = {"user": user_id}
logger = logging.LoggerAdapter(logger, extra)

# Initialisation de l’état de session
for key in ['datasets', 'explorations', 'sources', 'json_data', 'chat_history', 'test_results', 'test_interpretations', 'models', 'preprocessed_data']:
    if key not in st.session_state:
        st.session_state[key] = {} if key != 'chat_history' else []

# Configuration de l’API Gemini
API_KEY = "AIzaSyBJLhpSfKsbxgVEJwYmPSEZmaVlKt5qNlI" 
genai.configure(api_key=API_KEY)
client = genai.GenerativeModel('gemini-1.5-flash')

# CSS
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
    def detect_delimiter(self, content: str) -> str:
        if not content.strip():
            raise ValueError("Le fichier est vide ou ne contient aucune donnée valide.")
        sniffer = csv.Sniffer()
        try:
            return sniffer.sniff(content[:1024]).delimiter
        except:
            for delim in [',', ';', '\t', '|']:
                if delim in content[:1024]:
                    return delim
            return ','

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def load(self, source: any, source_type: str, skip_header: bool = False, header_row: int = None) -> Tuple[pd.DataFrame, Dict]:
        temp_dir = tempfile.mkdtemp()
        try:
            content_bytes, file_ext, total_size = None, None, 0
            status_text = st.empty()

            if source_type == "url":
                with requests.head(source, timeout=10) as head_response:
                    head_response.raise_for_status()
                    total_size = int(head_response.headers.get("Content-Length", 0))
                progress_bar = st.progress(0)
                with st.spinner("Téléchargement..."):
                    response = requests.get(source, timeout=15)
                    response.raise_for_status()
                    content_bytes = response.content
                downloaded_size = len(content_bytes)
                progress_bar.progress(min(int((downloaded_size / total_size) * 100), 100) if total_size else 100)
                file_ext = source.split('.')[-1].lower()

            elif source_type == "file":
                if isinstance(source, list):
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
                    total_size = sum(f.size for f in source)
                else:
                    content_bytes = source.read()
                    file_ext = source.name.split('.')[-1].lower()
                    total_size = len(content_bytes)
                status_text.text(f"Chargé : {total_size / 1024:.1f} Ko")

            elif source_type == "db":
                engine = sqlalchemy.create_engine(source[0])
                df = pd.read_sql(source[1], engine)
                return df, None

            else:
                raise ValueError(f"Type de source '{source_type}' non supporté")

            if source_type != "db":
                processed_content = content_bytes
                if file_ext == 'gz':
                    processed_content = gzip.decompress(content_bytes)
                    file_ext = source.split('.')[-2].lower() if source_type == "url" else source.name.split('.')[-2].lower()
                elif file_ext == 'zip':
                    with zipfile.ZipFile(io.BytesIO(content_bytes)) as z:
                        file_name = z.namelist()[0]
                        processed_content = z.read(file_name)
                        file_ext = file_name.split('.')[-1].lower()

                if file_ext == 'shp':
                    shp_path = os.path.join(temp_dir, 'data.shp' if source_type == "url" else shp_file)
                    with fiona.Env(SHAPE_RESTORE_SHX='YES'):
                        gdf = gpd.read_file(shp_path)
                    df = gdf.drop(columns=['geometry']).assign(latitude=gdf.geometry.centroid.y, longitude=gdf.geometry.centroid.x)
                    return df, None

                encoding = chardet.detect(processed_content[:1024])['encoding'] or 'utf-8'
                content_str = processed_content.decode(encoding, errors='replace')

                if file_ext in ['csv', 'txt']:
                    delimiter = self.detect_delimiter(content_str)
                    skiprows = [0] if skip_header else []
                    header = header_row if header_row is not None else (0 if not skip_header else None)
                    try:
                        df = pd.read_csv(io.StringIO(content_str), delimiter=delimiter, skiprows=skiprows, header=header, engine='python', on_bad_lines='warn')
                        if header_row is not None and header is None:
                            df.columns = df.iloc[0]
                            df = df.drop(0).reset_index(drop=True)
                    except Exception as e:
                        st.warning(f"Structure irrégulière détectée : {str(e)}. Tentative de correction...")
                        lines = content_str.splitlines()
                        if not any(line.strip() for line in lines):
                            raise ValueError("Fichier vide ou sans données exploitables.")
                        max_cols = max(len(line.split(delimiter)) for line in lines if line.strip())
                        df = pd.read_csv(io.StringIO(content_str), delimiter=delimiter, skiprows=skiprows, names=range(max_cols), engine='python', on_bad_lines='skip')
                    return df, None
                elif file_ext == 'xlsx':
                    df = pd.read_excel(io.BytesIO(processed_content), header=header_row, engine='openpyxl')
                    return df, None
                elif file_ext in ['json', 'geojson']:
                    try:
                        json_data = json.loads(content_str)
                        if "type" in json_data and json_data["type"] in ["FeatureCollection", "Feature"]:
                            gdf = gpd.read_file(io.StringIO(content_str))
                            return gdf.drop(columns=['geometry']).assign(latitude=gdf.geometry.centroid.y, longitude=gdf.geometry.centroid.x), None
                        return pd.json_normalize(json_data), None
                    except json.JSONDecodeError as e:
                        st.error(f"Erreur JSON : {str(e)}. Vérifiez le fichier.")
                        logger.error(f"Erreur JSON : {str(e)}")
                        return None, None
        except Exception as e:
            error_msg = f"Erreur de chargement : {str(e)}"
            logger.error(error_msg, exc_info=True)
            error_explanation = analyze_and_interpret("loading", None, error_msg, is_user_error=False)
            st.error(error_explanation)
            return None, None
        finally:
            if 'progress_bar' in locals():
                progress_bar.empty()
            status_text.empty()
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

class DataExplorer:
    @staticmethod
    def explore(df: pd.DataFrame) -> Dict[str, Any]:
        exploration = {
            "metadata": df.dtypes.to_dict(),
            "duplicates": df.duplicated().sum(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percent": (df.isnull().sum() / len(df) * 100).to_dict(),
            "description": df.describe(include='all'),
            "outliers": {},
            "unique_values": {col: df[col].nunique() for col in df.columns}
        }
        for col in df.select_dtypes(include=np.number).columns:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
            exploration["outliers"][col] = len(outliers)
        return exploration

def correlation(df: pd.DataFrame, var1: str, var2: str) -> Dict[str, float]:
    results = {}
    if var1 == var2 or var1 not in df.columns or var2 not in df.columns:
        st.warning("Variables invalides ou identiques.")
        return results
    df_clean = df[[var1, var2]].dropna()
    if df_clean[var1].dtype in [np.float64, np.int64] and df_clean[var2].dtype in [np.float64, np.int64]:
        try:
            results["pearson"] = stats.pearsonr(df_clean[var1], df_clean[var2])[0]
            results["spearman"] = stats.spearmanr(df_clean[var1], df_clean[var2])[0]
            results["kendall"] = stats.kendalltau(df_clean[var1], df_clean[var2])[0]
        except Exception as e:
            logger.error(f"Erreur dans correlation : {str(e)}", exc_info=True)
            results["error"] = "Pas assez de données valides pour calculer la corrélation."
    return results

def chi2_test(df: pd.DataFrame, var1: str, var2: str) -> Dict[str, Any]:
    results = {}
    if var1 == var2 or var1 not in df.columns or var2 not in df.columns:
        return results
    contingency_table = pd.crosstab(df[var1], df[var2])
    if contingency_table.size == 0 or contingency_table.empty:
        logger.warning(f"Tableau de contingence vide pour {var1} vs {var2}")
        results["error"] = f"Les variables '{var1}' et '{var2}' n'ont pas assez de données croisées pour effectuer le test Chi²."
        return results
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        results.update({"chi2": chi2, "p_value": p, "dof": dof, "expected": expected.tolist()})
    except Exception as e:
        logger.error(f"Erreur dans chi2_test : {str(e)}", exc_info=True)
        results["error"] = "Erreur lors du calcul du test Chi²."
    return results

def multivariate_tests(df: pd.DataFrame, group_var: str, value_var: str) -> Dict[str, Any]:
    results = {}
    if group_var not in df.columns or value_var not in df.columns:
        return results
    groups = [group[1][value_var].dropna() for group in df.groupby(group_var)]
    if len(groups) <= 1 or df[value_var].dtype not in [np.float64, np.int64]:
        logger.warning(f"Données insuffisantes ou invalides pour tests multivariés : {group_var} vs {value_var}")
        results["error"] = "Pas assez de groupes ou données non numériques."
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
        logger.error(f"Erreur dans multivariate_tests : {str(e)}", exc_info=True)
        results["error"] = "Erreur lors des tests multivariés."
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap='Blues' if method == 'chi2' else 'RdBu_r',
                vmin=0 if method == 'chi2' else -1, vmax=1, fmt='.3f', annot_kws={"size": 10})
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_and_interpret(test_type: str, results: Dict, error: str = None, is_user_error: bool = False) -> str:
    if error:
        if is_user_error:
            prompt = f"Erreur causée par l'utilisateur : '{error}'. Génère un message clair et compréhensible expliquant à l'utilisateur ce qu'il a mal fait."
        else:
            prompt = f"Erreur interne de l'outil : '{error}'. Génère un message clair et compréhensible pour un utilisateur non technique expliquant le problème."
        response = client.generate_content(prompt)
        return response.text.strip()
    else:
        prompt = f"Interprète ces résultats de test {test_type} : {json.dumps(results)}. Fournis un texte court et simple expliquant ce que cela signifie."
        response = client.generate_content(prompt)
        return response.text.strip()

def create_temp_pdf(dataset_name: str, data: pd.DataFrame, exploration: Dict, corr_mat: pd.DataFrame, chi2_mat: pd.DataFrame, test_results: Dict, interpretations: Dict) -> str:
    temp_pdf = f"temp_{dataset_name}_{int(time.time())}.pdf"
    doc = SimpleDocTemplate(temp_pdf, pagesize=letter)
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
        corr_img = f"temp_corr_{dataset_name}.png"
        save_matrix_to_image(corr_mat, "Matrice de Corrélation", corr_img)
        story.append(Image(corr_img, width=400, height=300))
    if chi2_mat is not None:
        chi2_img = f"temp_chi2_{dataset_name}.png"
        save_matrix_to_image(chi2_mat, "Matrice Chi²", chi2_img, method='chi2')
        story.append(Image(chi2_img, width=400, height=300))
    doc.build(story)
    return temp_pdf

def generate_gemini_report(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        response = client.generate_content([
            {"mime_type": "application/pdf", "data": f.read()},
            "Analyse et résume ce rapport en français dans un style clair, structuré et professionnel."
        ])
    return re.sub(r'[#*]+', '', response.text)

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

def preprocess_data(df: pd.DataFrame, features: list, target: str, encoding_method: str, normalization_method: str, modeling_type: str) -> Tuple[pd.DataFrame, pd.Series, list, Any]:
    X = df[features].copy()
    y = df[target].copy() if modeling_type != "Clustering" else None
    transformer = None

    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    processed_features = []

    for col in X.columns:
        if col in categorical_cols:
            try:
                X[col] = pd.to_numeric(X[col], errors='raise')
                processed_features.append(col)
            except ValueError:
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

    if modeling_type != "Clustering":
        if y.dtype == 'object' and encoding_method != "Exclude":
            try:
                y = pd.to_numeric(y, errors='raise')
            except ValueError:
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        X = X.dropna()
        y = y.loc[X.index]  # Alignement avec X tout en restant une Series

    if modeling_type == "Regression" and "Polynomial" in model_type:
        transformer = PolynomialFeatures(degree=2)
        X = transformer.fit_transform(X)
        processed_features = transformer.get_feature_names_out(input_features=processed_features).tolist()

    return X, y, processed_features, transformer

def train_model(df: pd.DataFrame, features: list, target: str, modeling_type: str, model_type: str, encoding_method: str, normalization_method: str) -> Tuple[Any, Dict, list, Any]:
    try:
        X, y, processed_features, transformer = preprocess_data(df, features, target if modeling_type != "Clustering" else None, encoding_method, normalization_method, modeling_type)
        
        # Vérification de la compatibilité de la cible
        if modeling_type == "Regression":
            if y.dtype not in [np.float64, np.int64]:
                raise ValueError("Erreur de l'utilisateur : La variable cible doit être numérique (ex. float ou int) pour la régression.")
        elif modeling_type == "Classification":
            if y.dtype in [np.float64, np.int64] and y.nunique() > 10:  # Seuil arbitraire pour éviter de confondre avec une variable continue
                raise ValueError("Erreur de l'utilisateur : La variable cible doit être catégorique (ex. texte ou peu de valeurs uniques) pour la classification.")
        
        if len(X) < 2:
            raise ValueError("Pas assez de données pour l’entraînement après prétraitement.")

        # Sélection du modèle selon le type de modélisation
        if modeling_type == "Regression":
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Polynomial Regression":
                model = LinearRegression()
            elif model_type == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_type == "Random Forest":
                model = RandomForestRegressor()
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor()
            model.fit(X, y)
            y_pred = model.predict(X)
            metrics = {"R²": r2_score(y, y_pred), "MSE": mean_squared_error(y, y_pred)}

        elif modeling_type == "Classification":
            if model_type == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_type == "Random Forest":
                model = RandomForestClassifier()
            elif model_type == "XGBoost":
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            elif model_type == "Logistic Regression":
                model = LogisticRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            metrics = {"Accuracy": accuracy_score(y, y_pred)}

        elif modeling_type == "Clustering":
            if model_type == "K-Means":
                model = KMeans(n_clusters=3)
            model.fit(X)
            y_pred = model.predict(X)
            metrics = {"Inertia": model.inertia_}

        return model, metrics, processed_features, transformer

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Erreur lors de l’entraînement : {error_msg}", exc_info=True)
        error_explanation = analyze_and_interpret("training", None, error_msg, is_user_error="Erreur de l'utilisateur" in error_msg)
        st.error(error_explanation)
        return None, {}, [], None

class DataPipeline:
    def __init__(self):
        self.loader = DataLoader()
        self.explorer = DataExplorer()

    def process(self, source: str, source_type: str, name: str, skip_header: bool = False, header_row: int = None) -> Tuple[pd.DataFrame, Dict, Dict]:
        data, json_data = self.loader.load(source, source_type, skip_header, header_row)
        if data is not None:
            exploration = self.explorer.explore(data)
            return data, exploration, None
        return None, None, json_data

def main():
    st.title("🔍 Exploration, Analyse et Modélisation de Données")
    pipeline = DataPipeline()

    with st.sidebar:
        st.header("📥 Chargement des Données")
        st.info("Chargez vos données ici (URL, fichiers locaux, ou base de données).")
        source_type = st.selectbox("Source", ["URL", "Fichiers Locaux", "Base de Données"], key="source_type_global", help="Choisissez la source des données.")

        if source_type == "URL":
            num_datasets = st.number_input("Nombre de datasets", min_value=1, value=1, step=1, key="num_datasets_input")
            for i in range(num_datasets):
                st.subheader(f"Dataset {i+1}")
                source = st.text_input(f"URL {i+1}", "", key=f"source_url_{i}", help="Exemple : https://example.com/data.csv")
                name = st.text_input(f"Nom {i+1}", f"dataset_{i+1}", key=f"name_{i}")
                skip_header = st.checkbox(f"Ignorer la première ligne {i+1}", value=False, key=f"skip_header_{i}")
                header_row = st.number_input(f"Ligne comme en-tête {i+1}", min_value=0, value=0, step=1, key=f"header_row_{i}", help="0 pour la première ligne.")
                if st.button(f"📤 Charger {i+1}", key=f"load_{i}") and source and name:
                    logger.info(f"Chargement de {name} depuis URL {source}")
                    data, exploration, json_data = pipeline.process(source, "url", name, skip_header, header_row)
                    if data is not None:
                        st.session_state['datasets'][name] = data
                        st.session_state['explorations'][name] = exploration
                        st.session_state['sources'][name] = (source, "url", skip_header, header_row)
                        st.session_state['test_results'][name] = {}
                        st.session_state['test_interpretations'][name] = {}
                        st.success(f"✅ '{name}' chargé ({len(data)} lignes)")
                        logger.info(f"Dataset {name} chargé avec succès ({len(data)} lignes)")
                    elif json_data:
                        st.session_state['json_data'][name] = json_data
                        st.session_state['sources'][name] = (source, "url", skip_header, header_row)
                        st.info(f"📋 JSON '{name}' chargé.")

        elif source_type == "Fichiers Locaux":
            uploaded_files = st.file_uploader("Importer des fichiers", type=["csv", "xlsx", "json", "geojson", "txt", "gz", "zip", "shp", "shx", "dbf"], 
                                             accept_multiple_files=True, key="multi_upload", help="Formats supportés : CSV, Excel, JSON, etc.")
            skip_header = st.checkbox("Ignorer la première ligne (CSV/TXT)", value=False, key="multi_skip_header")
            header_row = st.number_input("Ligne comme en-tête", min_value=0, value=0, step=1, key="multi_header_row")
            if uploaded_files and st.button("📤 Charger Tous", key="load_all"):
                for uploaded_file in uploaded_files:
                    name = uploaded_file.name.split('.')[0]
                    logger.info(f"Chargement de {name} depuis fichier local")
                    data, exploration, json_data = pipeline.process(uploaded_file, "file", name, skip_header, header_row)
                    if data is not None:
                        st.session_state['datasets'][name] = data
                        st.session_state['explorations'][name] = exploration
                        st.session_state['sources'][name] = (uploaded_file, "file", skip_header, header_row)
                        st.session_state['test_results'][name] = {}
                        st.session_state['test_interpretations'][name] = {}
                        st.success(f"✅ '{name}' chargé ({len(data)} lignes)")
                        logger.info(f"Dataset {name} chargé avec succès ({len(data)} lignes)")
                    elif json_data:
                        st.session_state['json_data'][name] = json_data
                        st.session_state['sources'][name] = (uploaded_file, "file", skip_header, header_row)
                        st.info(f"📋 JSON '{name}' chargé.")

        elif source_type == "Base de Données":
            db_url = st.text_input("URL de la base", key="db_url", help="Exemple : sqlite:///path.db")
            query = st.text_area("Requête SQL", "SELECT * FROM table_name", key="db_query")
            name = st.text_input("Nom du dataset", "db_dataset", key="db_name")
            if st.button("📤 Charger", key="load_db"):
                logger.info(f"Chargement de {name} depuis DB avec requête {query}")
                data, exploration, json_data = pipeline.process((db_url, query), "db", name)
                if data is not None:
                    st.session_state['datasets'][name] = data
                    st.session_state['explorations'][name] = exploration
                    st.session_state['sources'][name] = (query, "db", False, None)
                    st.session_state['test_results'][name] = {}
                    st.session_state['test_interpretations'][name] = {}
                    st.success(f"✅ '{name}' chargé ({len(data)} lignes)")
                    logger.info(f"Dataset {name} chargé avec succès ({len(data)} lignes)")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Actualiser", key="refresh"):
                logger.info("Actualisation des datasets")
                for name, (source, source_type, skip_header, header_row) in st.session_state['sources'].items():
                    data, exploration, json_data = pipeline.process(source, source_type, name, skip_header, header_row)
                    if data is not None:
                        st.session_state['datasets'][name] = data
                        st.session_state['explorations'][name] = exploration
                        st.session_state['test_results'][name] = {}
                        st.session_state['test_interpretations'][name] = {}
                    st.success("✅ Datasets actualisés")
                    logger.info("Datasets actualisés avec succès")
        with col2:
            if st.button("🗑️ Réinitialiser", key="reset"):
                logger.info("Réinitialisation de l’application")
                st.session_state.clear()
                for key in ['datasets', 'explorations', 'sources', 'json_data', 'chat_history', 'test_results', 'test_interpretations', 'models', 'preprocessed_data']:
                    st.session_state[key] = {} if key != 'chat_history' else []
                st.rerun()

        st.header("⚙️ Gestion des Datasets")
        st.info("Sélectionnez et modifiez vos datasets ici.")
        if st.session_state['datasets']:
            dataset_names = list(st.session_state['datasets'].keys())
            selected_datasets = st.multiselect("Datasets à traiter", dataset_names, default=dataset_names[0], key="select_datasets", help="Sélectionnez un ou plusieurs datasets.")

            if selected_datasets:
                with st.expander("🔄 Conversion des Types"):
                    st.info("Convertissez les types de données.")
                    col_to_convert = st.multiselect("Colonnes", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="convert_cols", help="Choisissez les colonnes à convertir.")
                    type_to_convert = st.selectbox("Type", ["Entier (int)", "Décimal (float)", "Catégorie (category)", "Date (datetime)", "Timestamp vers Date", "Extraire Mois", "Extraire Jour de la semaine", "Extraire Jour du mois", "Extraire Heure", "Extraire Année"], key="convert_type")
                    if st.button("✅ Appliquer", key="apply_convert"):
                        logger.info(f"Conversion de type pour {selected_datasets}")
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
                                        logger.info(f"Conversion appliquée pour {ds} sur {col} en {type_to_convert}")
                                    except Exception as e:
                                        error_msg = f"Erreur de conversion : {str(e)}"
                                        logger.error(error_msg)
                                        error_explanation = analyze_and_interpret("conversion", None, error_msg, is_user_error=True)
                                        st.error(error_explanation)

                with st.expander("🧹 Nettoyage des Valeurs"):
                    st.info("Modifiez le contenu des colonnes.")
                    col_to_clean = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="clean_col")
                    action = st.selectbox("Action", ["Supprimer des caractères", "Remplacer des caractères"], key="clean_action")
                    pattern = st.text_input("Motif (regex)", key="clean_pattern", help="Exemple : [0-9] pour supprimer les chiffres.")
                    replacement = st.text_input("Remplacement", "", key="clean_replace") if action == "Remplacer des caractères" else ""
                    if st.button("✅ Appliquer", key="apply_clean"):
                        logger.info(f"Nettoyage pour {selected_datasets}")
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
                                logger.info(f"Nettoyage appliqué pour {ds} sur {col_to_clean}")

                with st.expander("🗑️ Suppression de Colonnes"):
                    cols_to_drop = st.multiselect("Colonnes", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="drop_cols")
                    if st.button("✅ Supprimer", key="apply_drop"):
                        logger.info(f"Suppression de colonnes pour {selected_datasets}")
                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            data.drop(columns=[col for col in cols_to_drop if col in data.columns], inplace=True)
                            st.session_state['datasets'][ds] = data
                            st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                            st.success(f"✅ Colonnes supprimées pour '{ds}'")
                            logger.info(f"Colonnes supprimées pour {ds}")

                with st.expander("🚫 Suppression de Lignes"):
                    col_to_filter = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="filter_col")
                    filter_expr = st.text_input("Expression (ex. 'Station_1')", key="filter_expr")
                    filter_type = st.selectbox("Type", ["Valeur exacte", "Regex"], key="filter_type")
                    if st.button("✅ Supprimer", key="apply_filter"):
                        logger.info(f"Suppression de lignes pour {selected_datasets}")
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
                                logger.info(f"Lignes supprimées pour {ds}")

                with st.expander("➕ Création de Colonne"):
                    new_col_name = st.text_input("Nom", key="new_col_name")
                    base_col = st.selectbox("Base", set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets]), key="base_col")
                    new_col_action = st.selectbox("Action", ["Copie avec conversion", "Copie avec nettoyage"], key="new_col_action")
                    if new_col_action == "Copie avec conversion":
                        new_col_type = st.selectbox("Type", ["Entier (int)", "Décimal (float)", "Catégorie (category)", "Date (datetime)", "Timestamp vers Date", "Extraire Mois", "Extraire Jour de la semaine", "Extraire Jour du mois", "Extraire Heure", "Extraire Année"], key="new_col_type")
                    else:
                        new_col_pattern = st.text_input("Motif", key="new_col_pattern")
                        new_col_replace = st.text_input("Remplacement", "", key="new_col_replace")
                    if st.button("✅ Créer", key="apply_new_col"):
                        logger.info(f"Création de colonne pour {selected_datasets}")
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
                                logger.info(f"Colonne {new_col_name} créée pour {ds}")

                with st.expander("🕳️ Traitement des Valeurs Manquantes"):
                    col_to_fill = st.selectbox("Colonne", ["Toutes"] + list(set().union(*[set(st.session_state['datasets'][ds].columns) for ds in selected_datasets])), key="fill_col")
                    fill_method = st.selectbox("Méthode", ["Supprimer les lignes", "Supprimer toutes lignes vides", "Remplacer par moyenne", "Remplacer par mode", "Plus proche voisin"], key="fill_method")
                    if st.button("✅ Traiter", key="apply_fill"):
                        logger.info(f"Traitement des valeurs manquantes pour {selected_datasets}")
                        for ds in selected_datasets:
                            data = st.session_state['datasets'][ds].copy()
                            if col_to_fill == "Toutes" and fill_method == "Supprimer toutes lignes vides":
                                data.dropna(how='all', inplace=True)
                            elif col_to_fill in data.columns:
                                if fill_method == "Supprimer les lignes":
                                    data.dropna(subset=[col_to_fill], inplace=True)
                                elif fill_method == "Remplacer par moyenne" and data[col_to_fill].dtype in [np.float64, np.int64]:
                                    data[col_to_fill].fillna(data[col_to_fill].mean(), inplace=True)
                                elif fill_method == "Remplacer par mode":
                                    mode = data[col_to_fill].mode()
                                    data[col_to_fill].fillna(mode[0] if not mode.empty else None, inplace=True)
                                elif fill_method == "Plus proche voisin":
                                    data[col_to_fill] = data[col_to_fill].interpolate(method='nearest').ffill().bfill()
                            st.session_state['datasets'][ds] = data
                            st.session_state['explorations'][ds] = pipeline.explorer.explore(data)
                            st.success(f"✅ Valeurs manquantes traitées pour '{ds}'")
                            logger.info(f"Valeurs manquantes traitées pour {ds}")

                with st.expander("⚠️ Traitement des Valeurs Aberrantes"):
                    col_to_outlier = st.selectbox("Colonne", set().union(*[set(st.session_state['datasets'][ds].select_dtypes(include=np.number).columns) for ds in selected_datasets]), key="outlier_col")
                    outlier_method = st.selectbox("Méthode", ["Supprimer (IQR)", "Remplacer par médiane", "Limiter (IQR)"], key="outlier_method")
                    if st.button("✅ Traiter", key="apply_outlier"):
                        logger.info(f"Traitement des aberrants pour {selected_datasets}")
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
                                logger.info(f"Aberrants traités pour {ds}")

                with st.expander("🔗 Jointure de Datasets"):
                    if len(dataset_names) > 1:
                        left_ds = st.selectbox("Dataset principal", selected_datasets, key="join_left_ds")
                        right_ds = st.selectbox("Dataset secondaire", dataset_names, key="join_right_ds")
                        col_main = st.selectbox("Colonne principale", st.session_state['datasets'][left_ds].columns, key="join_col_main")
                        col_second = st.selectbox("Colonne secondaire", st.session_state['datasets'][right_ds].columns, key="join_col_second")
                        join_type = st.selectbox("Type", ["inner", "left", "right", "outer"], key="join_type")
                        if st.button("✅ Joindre", key="apply_join"):
                            logger.info(f"Jointure de {left_ds} et {right_ds}")
                            data = st.session_state['datasets'][left_ds].merge(st.session_state['datasets'][right_ds], how=join_type, left_on=col_main, right_on=col_second)
                            new_name = f"{left_ds}_joined_{right_ds}"
                            st.session_state['datasets'][new_name] = data
                            st.session_state['explorations'][new_name] = pipeline.explorer.explore(data)
                            st.session_state['test_results'][new_name] = {}
                            st.session_state['test_interpretations'][new_name] = {}
                            st.success(f"✅ Jointure effectuée : '{new_name}'")
                            logger.info(f"Jointure effectuée : {new_name}")

    if st.session_state['datasets']:
        st.markdown("<div class='section-box'><h2>📋 Aperçu des Données</h2>", unsafe_allow_html=True)
        selected_preview = st.selectbox("Choisir un dataset", list(st.session_state['datasets'].keys()), key="preview_dataset")
        st.dataframe(safe_dataframe_display(st.session_state['datasets'][selected_preview].head(10)), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-box'><h2>📊 Exploration, Visualisation et Modélisation</h2>", unsafe_allow_html=True)
        dataset_to_analyze = st.selectbox("Dataset à analyser", list(st.session_state['datasets'].keys()), key="analyze_dataset")
        data = st.session_state['datasets'][dataset_to_analyze]
        exploration = st.session_state['explorations'][dataset_to_analyze]
        test_results = st.session_state['test_results'][dataset_to_analyze]
        test_interpretations = st.session_state['test_interpretations'][dataset_to_analyze]

        st.write("**Aperçu des données**")
        st.dataframe(safe_dataframe_display(data.head()), use_container_width=True)

        with st.expander("📈 Exploration"):
            st.info("Découvrez les détails de votre dataset.")
            st.write(f"**Nom du dataset :** {dataset_to_analyze}")
            st.write("**Métadonnées :**")
            st.dataframe(pd.DataFrame(exploration["metadata"].items(), columns=["Colonne", "Type"]))
            st.write(f"**Doublons :** {exploration['duplicates']}")
            st.write("**Valeurs uniques :**")
            st.dataframe(pd.DataFrame(exploration["unique_values"].items(), columns=["Colonne", "Nb unique"]))
            st.write("**Valeurs manquantes :**")
            missing_df = pd.DataFrame({"Colonne": exploration["missing_values"].keys(), "Nb manquants": exploration["missing_values"].values(), "Pourcentage (%)": [f"{v:.2f}" for v in exploration["missing_percent"].values()]})
            st.dataframe(missing_df[missing_df["Nb manquants"] > 0]) if missing_df["Nb manquants"].sum() > 0 else st.write("Aucune valeur manquante")
            st.write("**Statistiques descriptives :**")
            st.dataframe(exploration["description"])
            st.write("**Valeurs aberrantes (IQR) :**")
            st.dataframe(pd.DataFrame(exploration["outliers"].items(), columns=["Colonne", "Nb aberrants"]))

        st.download_button(label="💾 Télécharger en CSV", data=data.to_csv(index=False), file_name=f"{dataset_to_analyze}.csv", mime="text/csv", key="download_csv")

        tab1, tab2, tab3, tab4 = st.tabs([
            "🎨 Premiers Pas : Découvrez Vos Données",
            "🔍 Zoom Sur les Relations",
            "🧩 Les Liens en un Coup d’Œil",
            "🤖 Prédictions et Magie IA"
        ])
        quant_cols = data.select_dtypes(include=[np.number]).columns
        qual_cols = data.select_dtypes(include=['object', 'category']).columns

        with tab1:
            st.subheader("🎨 Premiers Pas : Découvrez Vos Données")
            if len(quant_cols) > 0:
                col_hist = st.selectbox("Colonne pour histogramme", quant_cols, key="hist_select")
                fig_hist = px.histogram(data, x=col_hist, title=f"Distribution de {col_hist}", nbins=50, marginal="box")
                st.plotly_chart(fig_hist, use_container_width=True)
            if len(qual_cols) > 0:
                qual_col = st.selectbox("Variable qualitative", qual_cols, key="qual_select")
                fig_bar = px.histogram(data, x=qual_col, title=f"Répartition de {qual_col}", color=qual_col)
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            st.subheader("🔍 Zoom Sur les Relations")
            if len(quant_cols) > 1:
                with st.form(key='corr_form'):
                    col1, col2 = st.columns(2)
                    with col1:
                        var1 = st.selectbox("Variable 1", quant_cols, key="var1_select")
                    with col2:
                        var2 = st.selectbox("Variable 2", quant_cols, index=1, key="var2_select")
                    submit_corr = st.form_submit_button(label="✅ Calculer", help="Calcule la force de la relation entre deux variables numériques (ex. température et humidité).")
                if submit_corr:
                    corr = correlation(data, var1, var2)
                    test_results["correlation"] = {"var1": var1, "var2": var2, **corr}
                    if "error" in corr:
                        error_msg = analyze_and_interpret("correlation", None, corr["error"], is_user_error=True)
                        st.error(error_msg)
                    else:
                        st.write(f"**Pearson :** {corr.get('pearson', 'N/A'):.3f}")
                        st.write(f"**Spearman :** {corr.get('spearman', 'N/A'):.3f}")
                        st.write(f"**Kendall :** {corr.get('kendall', 'N/A'):.3f}")
                        interpretation = analyze_and_interpret("correlation", corr)
                        test_interpretations["correlation"] = interpretation
                        st.write(f"**Interprétation :** {interpretation}")
                        fig_scatter = px.scatter(data, x=var1, y=var2, trendline="ols", title=f"Corrélation : {var1} vs {var2}", color=var1)
                        st.plotly_chart(fig_scatter, use_container_width=True)

            if len(qual_cols) > 0 and len(quant_cols) > 0:
                with st.form(key='multi_form'):
                    col1, col2 = st.columns(2)
                    with col1:
                        group_var = st.selectbox("Variable qualitative", qual_cols, key="group_select")
                    with col2:
                        value_var = st.selectbox("Variable quantitative", quant_cols, key="value_select")
                    submit_multi = st.form_submit_button(label="✅ Effectuer", help="Compare une variable numérique (ex. ventes) entre groupes (ex. régions).")
                if submit_multi:
                    multi = multivariate_tests(data, group_var, value_var)
                    test_results["multivariate"] = {"group_var": group_var, "value_var": value_var, **multi}
                    if "error" in multi:
                        error_msg = analyze_and_interpret("multivariate", None, multi["error"], is_user_error=True)
                        st.error(error_msg)
                    else:
                        st.write(f"**ANOVA :** F={multi.get('anova', [None, None])[0]:.2f}, p={multi.get('anova', [None, None])[1]:.4f}")
                        st.write(f"**Kruskal-Wallis :** H={multi.get('kruskal', [None, None])[0]:.2f}, p={multi.get('kruskal', [None, None])[1]:.4f}")
                        st.write(f"**Levene :** F={multi.get('levene', [None, None])[0]:.2f}, p={multi.get('levene', [None, None])[1]:.4f}")
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
                        chi_var1 = st.selectbox("Variable qualitative 1", qual_cols, key="chi_var1")
                    with col2:
                        chi_var2 = st.selectbox("Variable qualitative 2", qual_cols, index=1, key="chi_var2")
                    submit_chi2 = st.form_submit_button(label="✅ Effectuer", help="Vérifie si deux variables catégoriques (ex. genre et préférence) sont liées.")
                if submit_chi2 and chi_var1 != chi_var2:
                    chi2_results = chi2_test(data, chi_var1, chi_var2)
                    test_results["chi2"] = {"var1": chi_var1, "var2": chi_var2, **chi2_results}
                    if "error" in chi2_results:
                        error_msg = analyze_and_interpret("chi2", None, chi2_results["error"], is_user_error=True)
                        st.error(error_msg)
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
                map_col = st.selectbox("Taille/Couleur", ["Aucune"] + list(data.columns), key="map_col")
                map_size = st.checkbox("Taille", key="map_size") if map_col != "Aucune" else False
                fig_map = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_name="nom_station" if "nom_station" in data.columns else None,
                                            size=map_col if map_size and map_col != "Aucune" else None, color=map_col if not map_size and map_col != "Aucune" else None,
                                            zoom=10, height=600, title=f"Carte ({dataset_to_analyze})")
                fig_map.update_layout(mapbox_style="dark")
                st.plotly_chart(fig_map, use_container_width=True)

        with tab3:
            st.subheader("🧩 Les Liens en un Coup d’Œil")
            if len(quant_cols) > 1:
                corr_method = st.selectbox("Méthode", ["pearson", "spearman"], key="corr_method", help="Pearson mesure les relations linéaires, Spearman les relations monotones.")
                corr_mat = correlation_matrix(data, method=corr_method)
                if corr_mat is not None:
                    fig_corr = px.imshow(corr_mat, text_auto=".2f", aspect="equal", title=f"Matrice de Corrélation ({corr_method.capitalize()})", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                    st.plotly_chart(fig_corr, use_container_width=True)

            if len(qual_cols) > 1:
                try:
                    chi2_mat = chi2_matrix(data)
                    if chi2_mat is not None:
                        fig_chi2 = px.imshow(chi2_mat, text_auto=".3f", aspect="equal", title="Matrice de Dépendance (p-valeurs Chi²)", color_continuous_scale="Blues", zmin=0, zmax=1)
                        st.plotly_chart(fig_chi2, use_container_width=True)
                except Exception as e:
                    logger.error(f"Erreur dans chi2_matrix : {str(e)}", exc_info=True)
                    error_msg = analyze_and_interpret("chi2_matrix", None, str(e), is_user_error=False)
                    st.error(error_msg)

        with tab4:
            st.subheader("🤖 Prédictions et Magie IA")
            if len(data.columns) > 1:
                features = st.multiselect("Variables explicatives", data.columns, key="model_features")
                target = st.selectbox("Variable cible", data.columns, key="model_target") if "Clustering" not in st.session_state.get('modeling_type', '') else None
                modeling_type = st.selectbox("Type de modélisation", ["Regression", "Classification", "Clustering"], key="modeling_type")
                model_options = {
                    "Regression": ["Linear Regression", "Polynomial Regression", "Decision Tree", "Random Forest", "XGBoost"],
                    "Classification": ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression"],
                    "Clustering": ["K-Means"]
                }
                model_type = st.selectbox("Modèle", model_options[modeling_type], key="model_type")
                encoding_method = st.selectbox("Méthode de numérisation", ["Label Encoding", "One-Hot Encoding", "Exclude"], key="encoding_method", help="Gérer les colonnes non numériques")
                normalization_method = st.selectbox("Méthode de normalisation", ["None", "StandardScaler", "RobustScaler", "MinMaxScaler"], key="normalization_method", help="Normaliser les données numériques")
                if st.button("✅ Entraîner", key="train_model"):
                    logger.info(f"Entraînement de {model_type} ({modeling_type}) sur {dataset_to_analyze}")
                    model, metrics, processed_features, transformer = train_model(data, features, target if modeling_type != "Clustering" else None, modeling_type, model_type, encoding_method, normalization_method)
                    if model:
                        st.session_state['models'][dataset_to_analyze] = (model, processed_features, target if modeling_type != "Clustering" else None, model_type, modeling_type, transformer)
                        st.session_state['preprocessed_data'][dataset_to_analyze] = (data[features], data[target] if modeling_type != "Clustering" else None, encoding_method, normalization_method)
                        st.write("**Métriques :**")
                        for k, v in metrics.items():
                            st.write(f"{k}: {v:.4f}")
                        try:
                            X_processed, y_processed, _, _ = preprocess_data(data, features, target if modeling_type != "Clustering" else None, encoding_method, normalization_method, modeling_type)
                            if modeling_type == "Regression":
                                fig_pred = px.scatter(x=y_processed, y=model.predict(X_processed), labels={"x": "Réel", "y": "Prédit"}, title=f"Prédictions ({model_type})")
                                st.plotly_chart(fig_pred)
                            elif modeling_type == "Classification":
                                fig_pred = px.scatter(x=y_processed, y=model.predict(X_processed), labels={"x": "Réel", "y": "Prédit"}, title=f"Prédictions ({model_type})")
                                st.plotly_chart(fig_pred)
                            elif modeling_type == "Clustering":
                                fig_cluster = px.scatter(data_frame=X_processed, x=processed_features[0], y=processed_features[1] if len(processed_features) > 1 else processed_features[0], color=model.predict(X_processed), title="Clusters K-Means")
                                st.plotly_chart(fig_cluster)
                            st.download_button("💾 Sauvegarder le modèle", data=joblib.dump(model, f"{dataset_to_analyze}_{model_type}.pkl")[0], file_name=f"{dataset_to_analyze}_{model_type}.pkl", key="download_model")
                            logger.info(f"Modèle {model_type} entraîné pour {dataset_to_analyze}")
                        except Exception as e:
                            error_msg = f"Erreur lors de la visualisation des prédictions : {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            error_explanation = analyze_and_interpret("prediction_visualization", None, error_msg, is_user_error=False)
                            st.error(error_explanation)

        if st.button("📝 Générer un rapport", key="generate_report"):
            logger.info(f"Génération de rapport pour {dataset_to_analyze}")
            corr_mat = correlation_matrix(data) if len(quant_cols) > 1 else None
            chi2_mat = chi2_matrix(data) if len(qual_cols) > 1 else None
            temp_pdf = create_temp_pdf(dataset_to_analyze, data, exploration, corr_mat, chi2_mat, test_results, test_interpretations)
            report = generate_gemini_report(temp_pdf)
            st.session_state['chat_history'] = [("", report)]
            os.remove(temp_pdf)
            if corr_mat is not None and os.path.exists(f"temp_corr_{dataset_to_analyze}.png"):
                os.remove(f"temp_corr_{dataset_to_analyze}.png")
            if chi2_mat is not None and os.path.exists(f"temp_chi2_{dataset_to_analyze}.png"):
                os.remove(f"temp_chi2_{dataset_to_analyze}.png")
            logger.info(f"Rapport généré pour {dataset_to_analyze}")

        if st.session_state['chat_history']:
            st.markdown("<h3>💬 Rapport d’analyse</h3>", unsafe_allow_html=True)
            report_text = display_progressive_report(st.session_state['chat_history'][0][1])
            pdf_data = create_downloadable_pdf(dataset_to_analyze, report_text)
            st.download_button(label="📄 Télécharger le rapport en PDF", data=pdf_data, file_name=f"rapport_{dataset_to_analyze}.pdf", mime="application/pdf", key="download_report")

        st.session_state['test_results'][dataset_to_analyze] = test_results
        st.session_state['test_interpretations'][dataset_to_analyze] = test_interpretations
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()