import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
    StratifiedShuffleSplit
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

import q1_paper_strict as q1

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


# ============================================================
# Classificador Bayesiano Gaussiano Clássico (Sem Ledoit-Wolf)
# ============================================================
class ClassicGaussianBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, reg_covar: float = 1e-5):
        self.reg_covar = reg_covar

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.priors_ = {}
        self.means_ = {}
        self.inv_covariances_ = {}
        self.log_dets_ = {}

        n = len(y)
        for cls in self.classes_:
            Xc = X[y == cls]
            self.priors_[cls] = Xc.shape[0] / n
            self.means_[cls] = Xc.mean(axis=0)
            
            if Xc.shape[0] <= 1:
                cov = np.zeros((Xc.shape[1], Xc.shape[1]))
            else:
                cov = np.cov(Xc, rowvar=False, bias=True)
            
            # Adiciona apenas a pequena regularização diagonal padrão
            cov = cov + self.reg_covar * np.eye(cov.shape[0])
            
            self.inv_covariances_[cls] = np.linalg.pinv(cov)
            sign, logdet = np.linalg.slogdet(cov)
            self.log_dets_[cls] = logdet
        return self

    def _log_gaussian_density(self, X: np.ndarray, cls) -> np.ndarray:
        mean = self.means_[cls]
        inv_cov = self.inv_covariances_[cls]
        logdet = self.log_dets_[cls]
        d = X.shape[1]
        diff = X - mean
        mahal = np.sum((diff @ inv_cov) * diff, axis=1)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        log_posteriors = []
        for cls in self.classes_:
            log_prior = np.log(self.priors_[cls] + 1e-15)
            log_lik = self._log_gaussian_density(X, cls)
            log_posteriors.append(log_prior + log_lik)
        return self.classes_[np.argmax(np.vstack(log_posteriors).T, axis=1)]


# ============================================================
# Classificador Bayesiano com Janela de Parzen (Clássico)
# ============================================================
class ParzenBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.priors_ = {}
        for cls in self.classes_:
            Xc = X[y == cls]
            kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
            kde.fit(Xc)
            self.models_[cls] = kde
            self.priors_[cls] = Xc.shape[0] / len(y)
        return self

    def predict(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        log_posteriors = []
        for cls in self.classes_:
            log_density = self.models_[cls].score_samples(X)
            log_prior = np.log(self.priors_[cls] + 1e-15)
            log_posteriors.append(log_density + log_prior)
        return self.classes_[np.argmax(np.vstack(log_posteriors).T, axis=1)]


# ============================================================
# Configuração dos modelos
# ============================================================
def build_model_configs():
    return {
        "BayesGauss": {
            "estimator": Pipeline([("sc", StandardScaler()), ("clf", ClassicGaussianBayes(reg_covar=1e-5))]),
            "param_grid": None
        },
        "kNN": {
            "estimator": Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier())]),
            "param_grid": {
                "clf__n_neighbors": [3, 5, 7],
                "clf__metric": ["euclidean", "manhattan", "chebyshev"]
            }
        },
        "Parzen": {
            "estimator": Pipeline([("sc", StandardScaler()), ("clf", ParzenBayesClassifier())]),
            "param_grid": {"clf__bandwidth": [0.1, 0.3, 0.5, 1.0]}
        },
        "LogReg": {
            "estimator": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]),
            "param_grid": {"clf__C": [0.1, 1.0, 10.0]}
        }
    }


def majority_vote(pred_list):
    preds = np.vstack(pred_list).T
    final = []
    for row in preds:
        classes, counts = np.unique(row, return_counts=True)
        final.append(classes[np.argmax(counts)])
    return np.array(final)


def evaluate_scenario_strict(X, y, scenario_name="A"):
    # Validação Cruzada 30x10-fold
    outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_configs = build_model_configs()
    model_names = ["BayesGauss", "kNN", "Parzen", "LogReg", "Voting"]
    
    results = {name: {"error": [], "f1": []} for name in model_names}
    
    total_folds = outer_cv.get_n_splits(X, y)
    
    for fold_id, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        fitted = {}
        
        # BayesGauss
        bg = clone(model_configs["BayesGauss"]["estimator"]).fit(X_tr, y_tr)
        fitted["BayesGauss"] = bg
        
        # kNN
        knn_gs = GridSearchCV(clone(model_configs["kNN"]["estimator"]), model_configs["kNN"]["param_grid"], cv=inner_cv, n_jobs=-1, scoring="f1_macro").fit(X_tr, y_tr)
        fitted["kNN"] = knn_gs.best_estimator_
        
        # Parzen
        pz_gs = GridSearchCV(clone(model_configs["Parzen"]["estimator"]), model_configs["Parzen"]["param_grid"], cv=inner_cv, n_jobs=-1, scoring="f1_macro").fit(X_tr, y_tr)
        fitted["Parzen"] = pz_gs.best_estimator_
        
        # LogReg
        lr_gs = GridSearchCV(clone(model_configs["LogReg"]["estimator"]), model_configs["LogReg"]["param_grid"], cv=inner_cv, n_jobs=-1, scoring="f1_macro").fit(X_tr, y_tr)
        fitted["LogReg"] = lr_gs.best_estimator_
        
        preds = {name: fitted[name].predict(X_te) for name in ["BayesGauss", "kNN", "Parzen", "LogReg"]}
        preds["Voting"] = majority_vote(list(preds.values()))
        
        for name in model_names:
            y_pred = preds[name]
            results[name]["error"].append(1.0 - accuracy_score(y_te, y_pred))
            results[name]["f1"].append(f1_score(y_te, y_pred, average="macro", zero_division=0))
            
        if fold_id % 50 == 0:
            print(f"[{scenario_name}] Processados {fold_id}/{total_folds} folds.")
            
    return results


def friedman_nemenyi_strict(res, scenario_name):
    metrics = ["error", "f1"]
    model_names = ["BayesGauss", "kNN", "Parzen", "LogReg", "Voting"]
    k = len(model_names)
    qa = 2.728  # q_alpha for k=5, alpha=0.05
    
    with open(os.path.join(_RESULTADOS_DIR, f"strict_friedman_{scenario_name}.txt"), "w") as f:
        f.write(f"=== Friedman & Nemenyi Test - Cenário {scenario_name} ===\n\n")
        
        for metric in metrics:
            scores = np.array([res[m][metric] for m in model_names]).T
            N = scores.shape[0]
            chi2_stat, p_val = stats.friedmanchisquare(*[scores[:, j] for j in range(k)])
            
            ranks = np.apply_along_axis(lambda x: stats.rankdata(x if metric == "error" else -x), 1, scores)
            mean_ranks = ranks.mean(axis=0)
            
            f.write(f"--- Métrica: {metric.upper()} ---\n")
            f.write(f"p-valor Friedman: {p_val:.4e}\n")
            CD = qa * np.sqrt(k * (k + 1) / (6 * N))
            f.write(f"CD (Nemenyi): {CD:.4f}\n")
            
            for m_name, r in zip(model_names, mean_ranks):
                f.write(f"Rank médio {m_name}: {r:.4f}\n")
            f.write("\n")


def learning_curves_strict(X, y, scenario_name):
    proportions = np.arange(0.05, 1.00, 0.05)
    model_names = ["BayesGauss", "kNN", "Parzen", "LogReg", "Voting"]
    test_scores = {m: [] for m in model_names}
    
    models = {
        "BayesGauss": Pipeline([("sc", StandardScaler()), ("clf", ClassicGaussianBayes(reg_covar=1e-5))]),
        "kNN": Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "Parzen": Pipeline([("sc", StandardScaler()), ("clf", ParzenBayesClassifier(bandwidth=1.0))]),
        "LogReg": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])
    }
    
    for prop in proportions:
        f1_reps = {m: [] for m in model_names}
        # Apenas 5 repetições para curva de aprendizado
        for rep in range(5):
            sss = StratifiedShuffleSplit(n_splits=1, train_size=prop, random_state=42+rep)
            tr_idx, te_idx = next(sss.split(X, y))
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            
            preds_te = {}
            for name, model in models.items():
                try:
                    model.fit(X_tr, y_tr)
                    preds_te[name] = model.predict(X_te)
                except:
                    preds_te[name] = np.zeros_like(y_te)
                    
            preds_te["Voting"] = majority_vote(list(preds_te.values()))
            
            for name in model_names:
                f1_reps[name].append(f1_score(y_te, preds_te[name], average="macro", zero_division=0))
                
        for name in model_names:
            test_scores[name].append(np.mean(f1_reps[name]))
            
    plt.figure(figsize=(10, 6))
    pct_labels = [f"{int(round(p*100))}%" for p in proportions]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]
    
    for i, name in enumerate(model_names):
        plt.plot(pct_labels, test_scores[name], marker=markers[i], color=colors[i], label=name, linewidth=2)
        
    plt.title(f"Curva de Aprendizado Estrita - Cenário {scenario_name}")
    plt.xlabel("Proporção de Treinamento")
    plt.ylabel("F1-Macro (Teste)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(_RESULTADOS_DIR, f"strict_learning_curve_{scenario_name}.png"))
    plt.close()


if __name__ == '__main__':
    print("Iniciando Questão 2 Estrita...")
    
    # Carregar base
    X, y_true = q1.load_ionosphere_strict(scale=True)
    
    # Cenário A (Ground Truth)
    print("Avaliando Cenário A...")
    res_A = evaluate_scenario_strict(X, y_true, "A")
    friedman_nemenyi_strict(res_A, "A")
    learning_curves_strict(X, y_true, "A")
    
    # Cenário B (Rótulos do KCM-K-GH)
    labels_file = os.path.join(_RESULTADOS_DIR, "strict_best_labels.npy")
    if os.path.exists(labels_file):
        print("Avaliando Cenário B com rótulos do Q1 estrito...")
        y_q1 = np.load(labels_file)
        res_B = evaluate_scenario_strict(X, y_q1, "B")
        friedman_nemenyi_strict(res_B, "B")
        learning_curves_strict(X, y_q1, "B")
    else:
        print("Arquivo strict_best_labels.npy não encontrado. Rode q1_paper_strict.py primeiro.")
        
    print("Concluído. Resultados salvos em Resultados_Strict/")
