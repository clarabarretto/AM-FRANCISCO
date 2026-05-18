"""Questão 1: execução do KCM-K-GH sobre o conjunto Ionosphere."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
np.set_printoptions(precision=4, suppress=True)

# Dataset constants
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
COLUMN_NAMES = [f"V{i}" for i in range(1, 35)] + ["class"]


def load_ionosphere():
    """Load the Ionosphere dataset and return features and binary labels."""
    df = pd.read_csv(URL, header=None, names=COLUMN_NAMES)
    df["class_num"] = df["class"].map({"g": 1, "b": 0})
    X = df.drop(columns=["class", "class_num"]).values.astype(float)
    y_true = df["class_num"].values
    return df, X, y_true


def gaussian_kernel_matrix(X, prototypes, s2):
    """Compute the Gaussian kernel matrix K(x_k, g_i) for all samples and prototypes."""
    n, p = X.shape
    c = prototypes.shape[0]
    K = np.zeros((n, c), dtype=float)
    inv_s2 = 1.0 / np.clip(s2, 1e-12, None)

    for i in range(c):
        diff2 = (X - prototypes[i]) ** 2
        weighted_sum = np.sum(diff2 * inv_s2, axis=1)
        K[:, i] = np.exp(-0.5 * weighted_sum)

    return K


def compute_objective(K, labels):
    """Evaluate objective J for the current partition and kernel values."""
    return float(2.0 * np.sum(1.0 - K[np.arange(len(labels)), labels]))


def relabel_clusters_to_match_binary(y_true, labels):
    """Relabels binary clusters to maximize agreement with the true binary labels."""
    unique_clusters = np.unique(labels)
    if len(unique_clusters) != 2:
        return labels.copy()

    cm_original = confusion_matrix(y_true, labels)
    cm_swapped = confusion_matrix(y_true, 1 - labels)
    if np.trace(cm_swapped) > np.trace(cm_original):
        return 1 - labels
    return labels.copy()


def run_kcm_k_gh(X, c, gamma=1.0, max_iter=200, tol=1e-6, random_state=None):
    """Run the KCM-K-GH algorithm for a given number of clusters c."""
    rng = np.random.default_rng(random_state)
    n, p = X.shape

    # Step 0: initialize prototypes randomly (Forgy) and s² globally
    init_idx = rng.choice(n, size=c, replace=False)
    prototypes = X[init_idx].copy()
    s2 = np.ones(p, dtype=float)
    labels = None
    obj_history = []

    for iteration in range(max_iter):
        labels_prev = None if labels is None else labels.copy()

        # ---------- Step 1: assignment and prototype update ----------
        K = gaussian_kernel_matrix(X, prototypes, s2)
        dist = 2.0 * (1.0 - K)
        labels = np.argmin(dist, axis=1)

        new_prototypes = prototypes.copy()
        for i in range(c):
            mask = labels == i
            if np.any(mask):
                Ki = K[mask, i]
                denom = np.sum(Ki) + 1e-12
                new_prototypes[i] = np.sum(Ki[:, None] * X[mask], axis=0) / denom
        prototypes = new_prototypes

        # ---------- Step 2: update global width parameters s² ----------
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

        # ---------- Step 3: cluster reallocation ----------
        K = gaussian_kernel_matrix(X, prototypes, s2)
        dist = 2.0 * (1.0 - K)
        labels = np.argmin(dist, axis=1)

        current_obj = compute_objective(K, labels)
        obj_history.append(current_obj)

        # Convergence test: stable partition or negligible objective improvement
        if iteration > 0:
            delta_obj = abs(obj_history[-2] - obj_history[-1])
            same_partition = np.array_equal(labels, labels_prev)
            if same_partition or delta_obj < tol:
                break

    return prototypes, s2, labels, obj_history


def run_question1(c_values=None, n_runs=100, gamma=1.0, random_seed=42, plot=True):
    """Execute the Q1 experiment: run KCM-K-GH for several c values and summarize results."""
    if c_values is None:
        c_values = [2, 3, 4, 5, 6]

    df, X, y_true = load_ionosphere()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
                random_state=seed_base + 1000 * c + run
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

    cm = confusion_matrix(y_true, labels_for_cm)
    print("Matriz de confusão:")
    print(cm)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(c_values, silhouette_scores, marker="o")
        plt.axvline(c_star, color="red", linestyle="--", label=f"c* = {c_star}")
        plt.title("Índice de Silhueta por Número de Clusters")
        plt.xlabel("Número de clusters (c)")
        plt.ylabel("Silhueta")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(obj_history_star, marker="o", linewidth=2)
        plt.title(f"Convergência da Função Objetivo — KCM-K-GH (c* = {c_star})")
        plt.xlabel("Iterações")
        plt.ylabel("J")
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


def ari_comment(ari_value):
    """Return a short ARI interpretation message based on the result."""
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
