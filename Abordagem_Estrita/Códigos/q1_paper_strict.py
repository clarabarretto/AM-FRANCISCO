import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Configurações de estilo para publicação
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_RESULTADOS_DIR = os.path.join(_PROJECT_ROOT, "Resultados")
os.makedirs(_RESULTADOS_DIR, exist_ok=True)


def load_ionosphere_strict(scale=True):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    df = pd.read_csv(URL, header=None)
    y_true = df.iloc[:, -1].map({"g": 1, "b": 0}).values
    
    # O artigo afirma explicitamente que a coluna V2 (variância zero) é removida.
    X = df.drop(columns=[1, 34]).values.astype(float)
    
    if scale:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X, y_true


def compute_sigma2(X):
    dist_sq = pdist(X, metric='sqeuclidean')
    q10 = np.quantile(dist_sq, 0.1)
    q90 = np.quantile(dist_sq, 0.9)
    return (q10 + q90) / 2.0


def gaussian_kernel_matrix(X, prototypes, s2):
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
    return float(2.0 * np.sum(1.0 - K[np.arange(len(labels)), labels]))


def run_kcm_k_gh_strict(X, c, max_iter=100, tol=1e-7, random_state=None):
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    
    # Passo 1: Forgy Initialization
    init_idx = rng.choice(n, size=c, replace=False)
    prototypes = X[init_idx].copy()
    
    sigma2 = compute_sigma2(X)
    s2 = np.full(p, sigma2)
    gamma_pow_1_p = 1.0 / sigma2
    
    K = gaussian_kernel_matrix(X, prototypes, s2)
    labels = np.argmin(2.0 * (1.0 - K), axis=1)
    obj_history = [compute_objective(K, labels)]
    
    for iteration in range(max_iter):
        labels_prev = labels.copy()
        
        # Representação
        new_prototypes = prototypes.copy()
        for i in range(c):
            mask = labels == i
            if np.any(mask):
                Ki = K[mask, i]
                denom = np.sum(Ki) + 1e-12
                new_prototypes[i] = np.sum(Ki[:, None] * X[mask], axis=0) / denom
        prototypes = new_prototypes
        
        # Larguras de banda
        K = gaussian_kernel_matrix(X, prototypes, s2)
        dispersion = np.zeros(p, dtype=float)
        for i in range(c):
            mask = labels == i
            if np.any(mask):
                diff2 = (X[mask] - prototypes[i]) ** 2
                weights = K[mask, i][:, None]
                dispersion += np.sum(weights * diff2, axis=0)
        
        dispersion = np.clip(dispersion, 1e-12, None)
        geometric_mean_dispersion = np.exp(np.mean(np.log(dispersion)))
        inv_s2 = gamma_pow_1_p * geometric_mean_dispersion / dispersion
        s2 = 1.0 / np.clip(inv_s2, 1e-12, None)
        
        # Alocação
        K = gaussian_kernel_matrix(X, prototypes, s2)
        labels = np.argmin(2.0 * (1.0 - K), axis=1)
        
        current_obj = compute_objective(K, labels)
        obj_history.append(current_obj)
        
        delta_obj = abs(obj_history[-2] - obj_history[-1])
        if np.array_equal(labels, labels_prev) or delta_obj < tol:
            break
            
    return prototypes, s2, labels, obj_history


def plot_convergence(obj_hist, c, filepath):
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(obj_hist)), obj_hist, marker='o', linestyle='-', color='#1f77b4')
    plt.title(f"Convergência do KCM-K-GH Estrito (c={c})")
    plt.xlabel("Iteração")
    plt.ylabel("Função Objetivo $J$")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_silhouette(c_values, silhouette_scores, filepath):
    plt.figure(figsize=(6, 4.5))
    c_star = max(silhouette_scores, key=silhouette_scores.get)
    scores = [silhouette_scores[c] for c in c_values]
    
    plt.plot(c_values, scores, marker="o", color="#1f77b4", linewidth=1.5, markersize=6)
    plt.axvline(c_star, color="#d62728", linestyle="--", linewidth=1.5, label=f"c* = {c_star}")
    
    plt.title("Índice de Silhueta por Número de Clusters", fontweight="bold")
    plt.xlabel("Número de clusters (c)")
    plt.ylabel("Silhueta")
    plt.legend(frameon=True, shadow=False)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_confusion_matrix_strict(y_true, y_pred, filepath):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_pred))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.title("Matriz de Confusão (Agrupamento vs Ground Truth)")
    plt.ylabel("Rótulo Verdadeiro (0=Bad, 1=Good)")
    plt.xlabel("Cluster Atribuído")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size or x.size < 2:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_pca_strict(X, labels, prototypes, filepath):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    unique_clusters = np.unique(labels)
    palette = sns.color_palette("Set1", n_colors=len(unique_clusters))
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    for i, cluster in enumerate(unique_clusters):
        idx = labels == cluster
        n_samples = np.sum(idx)
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[palette[i]], s=60, alpha=0.8, 
                   edgecolor='white', linewidth=0.5, marker=markers[i % len(markers)],
                   label=f"Cluster {cluster} (n={n_samples})")
        confidence_ellipse(X_pca[idx, 0], X_pca[idx, 1], ax, n_std=2.0, 
                           edgecolor=palette[i], linestyle='--', linewidth=2, alpha=0.7)
    
    # Calcular e plotar as posições dos protótipos em PCA apenas por aproximação
    # Como os protótipos não vivem no mesmo espaço que X (eles são uma média ponderada),
    # nós os projetamos com o modelo PCA treinado.
    if prototypes is not None and len(prototypes) > 0:
        prototypes_pca = pca.transform(prototypes)
        for i, cluster in enumerate(unique_clusters):
            # O protótipo pode não corresponder exatamente ao cluster i se a indexação mudar, 
            # mas assumiremos que len(unique_clusters) <= len(prototypes) e a ordem se manteve.
            ax.scatter(prototypes_pca[cluster, 0], prototypes_pca[cluster, 1],
                       c=[palette[i]], s=250, marker='X', edgecolor='black', linewidth=2, zorder=5)

    evr = pca.explained_variance_ratio_
    plt.title("Projeção PCA Estrita (2D) com Elipses (95% CI)")
    plt.xlabel(f"Componente Principal 1 ({evr[0]*100:.1f}%)")
    plt.ylabel(f"Componente Principal 2 ({evr[1]*100:.1f}%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def run_experiment():
    X, y_true = load_ionosphere_strict(scale=True)
    c_values = [2, 3, 4, 5, 6]
    n_runs = 100
    
    best_ari_per_c = {}
    best_sil_per_c = {}
    best_labels_per_c = {}
    best_obj_hist_per_c = {}
    best_prototypes_per_c = {}
    
    for c in c_values:
        best_obj = np.inf
        best_labels = None
        best_sil = -1
        best_hist = None
        best_prototypes = None
        
        for run in range(n_runs):
            prototypes, s2, labels, obj_hist = run_kcm_k_gh_strict(
                X, c, max_iter=100, tol=1e-7, random_state=42*c + run
            )
            
            final_obj = obj_hist[-1]
            if final_obj < best_obj:
                best_obj = final_obj
                best_labels = labels
                best_hist = obj_hist
                best_prototypes = prototypes
        
        unique_labels = np.unique(best_labels)
        if len(unique_labels) > 1:
            sil = silhouette_score(X, best_labels)
        else:
            sil = -1 
            
        ari = adjusted_rand_score(y_true, best_labels)
        
        best_sil_per_c[c] = sil
        best_ari_per_c[c] = ari
        best_labels_per_c[c] = best_labels
        best_obj_hist_per_c[c] = best_hist
        best_prototypes_per_c[c] = best_prototypes

    # Identificar o melhor C de acordo com Silhueta
    best_c = max(best_sil_per_c, key=best_sil_per_c.get)
    
    # Salvar Gráficos
    plot_silhouette(c_values, best_sil_per_c, os.path.join(_RESULTADOS_DIR, "strict_silhouette_bars.png"))
    plot_convergence(best_obj_hist_per_c[best_c], best_c, os.path.join(_RESULTADOS_DIR, "strict_convergence.png"))
    plot_confusion_matrix_strict(y_true, best_labels_per_c[best_c], os.path.join(_RESULTADOS_DIR, f"strict_cm_c{best_c}.png"))
    plot_pca_strict(X, best_labels_per_c[best_c], best_prototypes_per_c[best_c], os.path.join(_RESULTADOS_DIR, f"strict_pca_c{best_c}.png"))
    
    # Salvar rótulos para uso na questão 2
    np.save(os.path.join(_RESULTADOS_DIR, "strict_best_labels.npy"), best_labels_per_c[best_c])
    
    # Gerar um DataFrame resumo
    res_df = pd.DataFrame({
        "c": c_values,
        "Silhouette": [best_sil_per_c[c] for c in c_values],
        "ARI": [best_ari_per_c[c] for c in c_values]
    })
    res_df.to_csv(os.path.join(_RESULTADOS_DIR, "strict_q1_metrics.csv"), index=False)

if __name__ == '__main__':
    print("Iniciando Questão 1 Estrita...")
    run_experiment()
    print("Concluído. Resultados em Resultados_Strict/")
