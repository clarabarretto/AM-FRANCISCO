"""Questão 1 (v2): execução do KCM-K-GH sobre o conjunto Ionosphere.

Versão aprimorada do q1.py com:
  - Padronização opcional via parâmetro `scale`.
  - Inicialização de s² baseada na variância dos dados quando scale=False.
  - Tratamento de clusters vazios (reinicialização a partir do maior cluster).
  - Salvamento automático de figuras no diretório Resultados/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

# Configurações de estilo para publicação (LaTeX)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})
sns.set_theme(style="whitegrid", rc={"font.family": "serif"})
np.set_printoptions(precision=4, suppress=True)

# Constantes do dataset
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
COLUMN_NAMES = [f"V{i}" for i in range(1, 35)] + ["class"]

# Diretório para salvar figuras
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Resultados")


def load_ionosphere():
    """Carrega o dataset Ionosphere e retorna as features e rótulos binários."""
    df = pd.read_csv(URL, header=None, names=COLUMN_NAMES)
    df["class_num"] = df["class"].map({"g": 1, "b": 0})
    X = df.drop(columns=["class", "class_num"]).values.astype(float)
    y_true = df["class_num"].values
    return df, X, y_true


def gaussian_kernel_matrix(X: np.ndarray, prototypes: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Calcula a matriz de kernel Gaussiano K(x_k, g_i) para todas as amostras e protótipos.

    Parâmetros
    ----------
    X : np.ndarray, shape (n, p)
        Matriz de dados.
    prototypes : np.ndarray, shape (c, p)
        Protótipos dos clusters.
    s2 : np.ndarray, shape (p,)
        Vetor global de parâmetros de largura (variâncias).

    Retorna
    -------
    K : np.ndarray, shape (n, c)
        Matriz de kernel Gaussiano.
    """
    n, p = X.shape
    c = prototypes.shape[0]
    K = np.zeros((n, c), dtype=float)
    inv_s2 = 1.0 / np.clip(s2, 1e-12, None)

    for i in range(c):
        diff2 = (X - prototypes[i]) ** 2
        weighted_sum = np.sum(diff2 * inv_s2, axis=1)
        K[:, i] = np.exp(-0.5 * weighted_sum)

    return K


def compute_objective(K: np.ndarray, labels: np.ndarray) -> float:
    """Avalia a função objetivo J para a partição e valores de kernel atuais.

    Parâmetros
    ----------
    K : np.ndarray, shape (n, c)
        Matriz de kernel Gaussiano.
    labels : np.ndarray, shape (n,)
        Rótulos de cluster de cada amostra.

    Retorna
    -------
    float
        Valor da função objetivo J.
    """
    return float(2.0 * np.sum(1.0 - K[np.arange(len(labels)), labels]))


def relabel_clusters_to_match_binary(y_true: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Rerotula clusters binários para maximizar a concordância com os rótulos verdadeiros.

    Parâmetros
    ----------
    y_true : np.ndarray, shape (n,)
        Rótulos verdadeiros (binários).
    labels : np.ndarray, shape (n,)
        Rótulos de cluster atribuídos.

    Retorna
    -------
    np.ndarray
        Rótulos possivelmente permutados para melhor concordância.
    """
    unique_clusters = np.unique(labels)
    if len(unique_clusters) != 2:
        return labels.copy()

    cm_original = confusion_matrix(y_true, labels)
    cm_swapped = confusion_matrix(y_true, 1 - labels)
    if np.trace(cm_swapped) > np.trace(cm_original):
        return 1 - labels
    return labels.copy()


def _handle_empty_clusters(labels: np.ndarray, prototypes: np.ndarray,
                           X: np.ndarray, c: int,
                           rng: np.random.Generator) -> None:
    """Trata clusters vazios reinicializando o protótipo a partir do maior cluster.

    Se algum cluster ficou sem amostras após a etapa de atribuição, seu protótipo
    é substituído por um ponto de dados escolhido aleatoriamente do maior cluster.
    A modificação é feita in-place em `prototypes`.

    Parâmetros
    ----------
    labels : np.ndarray, shape (n,)
        Rótulos de cluster atuais.
    prototypes : np.ndarray, shape (c, p)
        Protótipos dos clusters (modificados in-place).
    X : np.ndarray, shape (n, p)
        Matriz de dados.
    c : int
        Número de clusters.
    rng : np.random.Generator
        Gerador de números aleatórios.
    """
    cluster_sizes = np.array([np.sum(labels == i) for i in range(c)])
    largest_cluster = np.argmax(cluster_sizes)
    largest_mask = np.where(labels == largest_cluster)[0]

    for i in range(c):
        if cluster_sizes[i] == 0:
            donor_idx = rng.choice(largest_mask)
            prototypes[i] = X[donor_idx].copy()


def run_kcm_k_gh(X: np.ndarray, c: int, gamma: float = 1.0,
                 max_iter: int = 200, tol: float = 1e-6,
                 random_state: int = None,
                 init_s2: np.ndarray = None) -> tuple:
    """Executa o algoritmo KCM-K-GH para um dado número de clusters c.

    Implementa o algoritmo descrito em de Carvalho et al. (2018),
    equações 14 (atualização de protótipos), 16 (atualização de s²)
    e 18 (atribuição).

    Parâmetros
    ----------
    X : np.ndarray, shape (n, p)
        Matriz de dados.
    c : int
        Número de clusters.
    gamma : float, opcional
        Hiperparâmetro de regularização (padrão=1.0).
    max_iter : int, opcional
        Número máximo de iterações (padrão=200).
    tol : float, opcional
        Tolerância para convergência (padrão=1e-6).
    random_state : int, opcional
        Semente para reprodutibilidade.
    init_s2 : np.ndarray, opcional
        Vetor inicial de s². Se None, usa np.ones(p).

    Retorna
    -------
    tuple
        (prototypes, s2, labels, obj_history)
    """
    rng = np.random.default_rng(random_state)
    n, p = X.shape

    # Passo 0: inicialização dos protótipos (Forgy) e s² global
    init_idx = rng.choice(n, size=c, replace=False)
    prototypes = X[init_idx].copy()
    s2 = init_s2.copy() if init_s2 is not None else np.ones(p, dtype=float)
    labels = None
    obj_history = []

    for iteration in range(max_iter):
        labels_prev = None if labels is None else labels.copy()

        # ---------- Passo 1: atribuição e atualização de protótipos ----------
        K = gaussian_kernel_matrix(X, prototypes, s2)
        dist = 2.0 * (1.0 - K)
        labels = np.argmin(dist, axis=1)

        # Tratamento de clusters vazios após atribuição
        _handle_empty_clusters(labels, prototypes, X, c, rng)

        new_prototypes = prototypes.copy()
        for i in range(c):
            mask = labels == i
            if np.any(mask):
                Ki = K[mask, i]
                denom = np.sum(Ki) + 1e-12
                new_prototypes[i] = np.sum(Ki[:, None] * X[mask], axis=0) / denom
        prototypes = new_prototypes

        # ---------- Passo 2: atualização dos parâmetros de largura s² ----------
        K = gaussian_kernel_matrix(X, prototypes, s2)
        dispersion = np.zeros(p, dtype=float)
        for i in range(c):
            mask = labels == i
            if np.any(mask):
                diff2 = (X[mask] - prototypes[i]) ** 2
                weights = K[mask, i][:, None]
                dispersion += np.sum(weights * diff2, axis=0)
        dispersion = np.clip(dispersion, 1e-12, None)
        geometric_mean = np.exp(np.mean(np.log(dispersion)))
        w = gamma * geometric_mean / dispersion
        s2 = 1.0 / np.clip(w, 1e-12, None)

        # ---------- Passo 3: realocação de clusters ----------
        K = gaussian_kernel_matrix(X, prototypes, s2)
        dist = 2.0 * (1.0 - K)
        labels = np.argmin(dist, axis=1)

        # Tratamento de clusters vazios após realocação
        _handle_empty_clusters(labels, prototypes, X, c, rng)

        current_obj = compute_objective(K, labels)
        obj_history.append(current_obj)

        # Teste de convergência: partição estável ou melhoria desprezível
        if iteration > 0:
            delta_obj = abs(obj_history[-2] - obj_history[-1])
            same_partition = np.array_equal(labels, labels_prev)
            if same_partition or delta_obj < tol:
                break

    return prototypes, s2, labels, obj_history


def run_question1(c_values=None, n_runs: int = 100, gamma: float = 1.0,
                  random_seed: int = 42, plot: bool = True,
                  scale: bool = True) -> dict:
    """Executa o experimento Q1: roda KCM-K-GH para vários valores de c e resume os resultados.

    Parâmetros
    ----------
    c_values : list[int], opcional
        Lista de números de clusters a testar (padrão=[2, 3, 4, 5, 6]).
    n_runs : int, opcional
        Número de execuções independentes para cada c (padrão=100).
    gamma : float, opcional
        Hiperparâmetro de regularização (padrão=1.0).
    random_seed : int, opcional
        Semente base para reprodutibilidade (padrão=42).
    plot : bool, opcional
        Se True, gera e salva os gráficos (padrão=True).
    scale : bool, opcional
        Se True, aplica StandardScaler aos dados. Quando False, usa os dados
        originais e inicializa s² com a variância de cada variável (padrão=True).

    Retorna
    -------
    dict
        Dicionário com todos os resultados do experimento.
    """
    if c_values is None:
        c_values = [2, 3, 4, 5, 6]

    df, X, y_true = load_ionosphere()

    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        init_s2 = np.ones(X.shape[1], dtype=float)
    else:
        X_scaled = X.copy()
        init_s2 = np.var(X, axis=0)

    print("Dimensão do dataset:", df.shape)
    print(df.head())
    print("Shape de X padronizado:", X_scaled.shape)
    print("Classes:", np.unique(y_true, return_counts=True))

    best_results_per_c = {}
    silhouette_scores = []
    seed_base = random_seed

    for c in c_values:
        best_obj = np.inf
        best_result = None
        for run in range(n_runs):
            prototypes, s2, labels, obj_history = run_kcm_k_gh(
                X_scaled,
                c=c,
                gamma=gamma,
                max_iter=200,
                tol=1e-6,
                random_state=seed_base + 1000 * c + run,
                init_s2=init_s2
            )
            final_obj = obj_history[-1]
            if final_obj < best_obj:
                best_obj = final_obj
                best_result = {
                    "prototypes": prototypes,
                    "s2": s2,
                    "labels": labels,
                    "obj_history": obj_history,
                    "final_obj": final_obj
                }

        sil = silhouette_score(X_scaled, best_result["labels"])
        best_result["silhouette"] = sil
        best_results_per_c[c] = best_result
        silhouette_scores.append(sil)
        print(f"c = {c} | melhor J = {best_obj:.6f} | silhueta = {sil:.6f}")

    c_star = c_values[np.argmax(silhouette_scores)]
    best_star = best_results_per_c[c_star]
    prototypes_star = best_star["prototypes"]
    s2_star = best_star["s2"]
    labels_star = best_star["labels"]
    obj_history_star = best_star["obj_history"]

    print(f"Número ótimo de clusters escolhido: c* = {c_star}")
    print("Melhor função objetivo para c*:", best_star["final_obj"])
    print("Silhueta para c*:", best_star["silhouette"])
    ari = adjusted_rand_score(y_true, labels_star)
    print(f"Índice de Rand Corrigido (ARI) para c* = {c_star}: {ari:.6f}")
    print(ari_comment(ari))

    df_prototypes = pd.DataFrame(
        prototypes_star,
        columns=[f"V{i}" for i in range(1, 35)],
        index=[f"Cluster_{i}" for i in range(1, c_star + 1)]
    )
    print("Protótipos dos grupos (na base padronizada):")
    print(df_prototypes)

    df_s = pd.DataFrame({
        "Variável": [f"V{i}" for i in range(1, 35)],
        "s²_j": s2_star,
        "s_j": np.sqrt(s2_star)
    })
    print("Vetor global de parâmetros de largura:")
    print(df_s)

    df_s_sorted = df_s.sort_values("s_j", ascending=True).reset_index(drop=True)
    print("Variáveis mais relevantes (menores valores de s_j):")
    print(df_s_sorted.head(10))

    labels_for_cm = labels_star.copy()
    if c_star == 2:
        labels_for_cm = relabel_clusters_to_match_binary(y_true, labels_star)

    cm = confusion_matrix(y_true, labels_for_cm, labels=np.arange(c_star))
    print("Matriz de confusão:")
    print(cm)

    if plot:
        # Garante que o diretório de resultados existe
        os.makedirs(RESULTS_DIR, exist_ok=True)

        plt.figure(figsize=(6, 4.5))
        yticklabels = ["b (0)", "g (1)"] + [""] * (c_star - 2)
        xticklabels = [f"Cl.{i+1}" for i in range(c_star)]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                    xticklabels=xticklabels, yticklabels=yticklabels)
        plt.title(f"Matriz de Confusão — c* = {c_star}")
        plt.ylabel("Classe a priori")
        plt.xlabel("Cluster (KCM-K-GH)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "matriz_confusao.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.plot(c_values, silhouette_scores, marker="o", color="#1f77b4", linewidth=1.5, markersize=6)
        plt.axvline(c_star, color="#d62728", linestyle="--", linewidth=1.5, label=f"c* = {c_star}")
        plt.title("Índice de Silhueta por Número de Clusters", fontweight="bold")
        plt.xlabel("Número de clusters (c)")
        plt.ylabel("Silhueta")
        plt.legend(frameon=True, shadow=False)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "silhueta_vs_clusters.png"))
        plt.show()

        plt.figure(figsize=(6, 4.5))
        plt.plot(obj_history_star, marker="s", color="#2ca02c", linewidth=1.5, markersize=5)
        plt.title(f"Convergência da Função Objetivo — KCM-K-GH (c* = {c_star})", fontweight="bold")
        plt.xlabel("Iterações")
        plt.ylabel("J (Função Objetivo)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "convergencia_funcao_objetivo.png"))
        plt.show()

    summary_q1 = pd.DataFrame({
        "c": c_values,
        "Melhor_J": [best_results_per_c[c]["final_obj"] for c in c_values],
        "Silhueta": [best_results_per_c[c]["silhouette"] for c in c_values]
    })
    print(summary_q1)

    return {
        "df": df,
        "X_scaled": X_scaled,
        "X": X,
        "y_true": y_true,
        "best_results_per_c": best_results_per_c,
        "c_star": c_star,
        "labels_star": labels_star,
        "prototypes_star": prototypes_star,
        "s2_star": s2_star,
        "obj_history_star": obj_history_star,
        "summary_q1": summary_q1,
        "silhouette_scores": silhouette_scores,
        "ari": ari,
    }


def ari_comment(ari_value: float) -> str:
    """Retorna uma mensagem curta de interpretação do ARI com base no valor obtido.

    Parâmetros
    ----------
    ari_value : float
        Valor do Índice de Rand Ajustado.

    Retorna
    -------
    str
        Mensagem interpretativa.
    """
    if ari_value < 0.20:
        return "ARI muito baixo: a partição encontrada tem pouca concordância com as classes a priori."
    elif ari_value < 0.40:
        return "ARI baixo: há alguma estrutura coincidente, mas a concordância com as classes a priori é fraca."
    elif ari_value < 0.60:
        return "ARI moderado: o agrupamento recupera parcialmente a estrutura das classes a priori."
    elif ari_value < 0.80:
        return "ARI bom: há boa concordância entre clusters e classes a priori."
    else:
        return "ARI muito alto: o agrupamento recupera muito bem as classes a priori."


if __name__ == "__main__":
    run_question1()
