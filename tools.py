import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import contextlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def describe_dataset(df: pd.DataFrame) -> str:
    info = []
    info.append("### Colunas e tipos de dados:")
    info.append(str(df.dtypes))
    info.append("\n### Valores ausentes por coluna:")
    info.append(str(df.isnull().sum()))
    return "\n".join(info)

def plot_histogram(df: pd.DataFrame, col: str, bins: int = 50) -> str:
    plt.figure(figsize=(8,5))
    sns.histplot(df[col].dropna(), bins=bins, kde=True)
    plt.title(f"Histograma de {col}")
    path = os.path.join(ARTIFACTS_DIR, f"hist_{col}.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_boxplot(df: pd.DataFrame, col: str) -> str:
    plt.figure(figsize=(6,5))
    sns.boxplot(y=df[col].dropna())
    plt.title(f"Boxplot de {col}")
    path = os.path.join(ARTIFACTS_DIR, f"box_{col}.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    plt.figure(figsize=(10,8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Mapa de Correlação")
    path = os.path.join(ARTIFACTS_DIR, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    return path

def detect_outliers(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.DataFrame:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return outliers[[col]].head(10)

def cluster_kmeans(df: pd.DataFrame, cols: list, n_clusters: int = 3) -> str:
    data = df[cols].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    data["Cluster"] = labels

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=data[cols[0]], y=data[cols[1]], hue=labels, palette="tab10")
    plt.title(f"KMeans Clustering ({cols[0]} vs {cols[1]})")
    path = os.path.join(ARTIFACTS_DIR, "kmeans.png")
    plt.savefig(path)
    plt.close()
    return path

def run_python_code(df: pd.DataFrame, code: str) -> str:
    """Executa código Python seguro, com acesso ao df."""
    local_vars = {"df": df}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {}, local_vars)
    except Exception as e:
        return f"Erro na execução do código: {e}\n\nCódigo gerado:\n{code}"
    output = stdout.getvalue()
    if "result" in local_vars:
        return f"{output}\nResultado: {local_vars['result']}"
    return output or "Código executado sem saída."
