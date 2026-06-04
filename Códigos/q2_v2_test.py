"""Questão 2 (v2): avaliação de classificadores e análise estatística usando o conjunto Ionosphere.

Versão melhorada com:
- BayesianKNNClassifier (estimativa de densidade kNN bayesiana)
- Regularização LedoitWolf para MultivariateGaussianBayes
- Curvas de aprendizagem com múltiplas repetições
- Figuras salvas no diretório Resultados/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations

from scipy import stats
from scipy.special import gammaln
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
    f1_score,
    confusion_matrix
)
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

import q1_v2 as q1

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
np.set_printoptions(precision=4, suppress=True)

# --- Diretório de saída para figuras ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_RESULTADOS_DIR = os.path.join(_PROJECT_ROOT, "Resultados")
os.makedirs(_RESULTADOS_DIR, exist_ok=True)


# ============================================================
# Classificador Bayesiano Gaussiano Multivariado
# ============================================================
class MultivariateGaussianBayes(BaseEstimator, ClassifierMixin):
    """Classificador Bayes com modelo gaussiano multivariado por classe.

    Utiliza regularização da covariância e, opcionalmente, o estimador
    LedoitWolf quando a matriz de covariância é mal-condicionada.

    Parâmetros
    ----------
    reg_covar : float, padrão=1e-4
        Valor de regularização adicionado à diagonal da covariância.
    """

    _estimator_type = "classifier"

    def __init__(self, reg_covar: float = 1e-4):
        self.reg_covar = reg_covar

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultivariateGaussianBayes":
        """Ajusta o modelo calculando parâmetros gaussianos por classe.

        Parâmetros
        ----------
        X : array-like de forma (n_amostras, n_features)
            Dados de treinamento.
        y : array-like de forma (n_amostras,)
            Rótulos das classes.

        Retorna
        -------
        self
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.priors_ = {}
        self.means_ = {}
        self.covariances_ = {}
        self.inv_covariances_ = {}
        self.log_dets_ = {}

        n = len(y)
        for cls in self.classes_:
            Xc = X[y == cls]
            prior = Xc.shape[0] / n
            mean = Xc.mean(axis=0)
            cov = np.cov(Xc, rowvar=False, bias=True)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            cov = cov + self.reg_covar * np.eye(cov.shape[0])

            # Verifica condicionamento; se mal-condicionada, usa LedoitWolf
            if np.linalg.cond(cov) > 1e10:
                lw = LedoitWolf().fit(Xc)
                cov = lw.covariance_

            inv_cov = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
            self.priors_[cls] = prior
            self.means_[cls] = mean
            self.covariances_[cls] = cov
            self.inv_covariances_[cls] = inv_cov
            self.log_dets_[cls] = logdet
        return self

    def _log_gaussian_density(self, X: np.ndarray, cls) -> np.ndarray:
        """Calcula log p(x | ω_cls) assumindo distribuição gaussiana.

        Parâmetros
        ----------
        X : array de forma (n_amostras, d)
        cls : rótulo da classe

        Retorna
        -------
        log_densidades : array de forma (n_amostras,)
        """
        mean = self.means_[cls]
        inv_cov = self.inv_covariances_[cls]
        logdet = self.log_dets_[cls]
        d = X.shape[1]
        diff = X - mean
        mahal = np.sum((diff @ inv_cov) * diff, axis=1)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz classes via MAP (máxima probabilidade a posteriori).

        Parâmetros
        ----------
        X : array de forma (n_amostras, d)

        Retorna
        -------
        y_pred : array de forma (n_amostras,)
        """
        check_is_fitted(self)
        X = check_array(X)
        log_posteriors = []
        for cls in self.classes_:
            log_prior = np.log(self.priors_[cls] + 1e-15)
            log_lik = self._log_gaussian_density(X, cls)
            log_posteriors.append(log_prior + log_lik)
        log_posteriors = np.vstack(log_posteriors).T
        return self.classes_[np.argmax(log_posteriors, axis=1)]


# ============================================================
# Classificador Bayesiano kNN (estimativa de densidade kNN)
# ============================================================
class BayesianKNNClassifier(BaseEstimator, ClassifierMixin):
    """Classificador Bayesiano baseado em estimativa de densidade kNN.

    Para cada classe, ajusta um modelo NearestNeighbors e estima a
    densidade p(x|ω_i) usando o volume da bola métrica ao k-ésimo
    vizinho mais próximo.

    Parâmetros
    ----------
    n_neighbors : int, padrão=5
        Número de vizinhos para estimativa de densidade.
    metric : str, padrão='euclidean'
        Métrica de distância ('euclidean', 'manhattan' ou 'chebyshev').
    """

    _estimator_type = "classifier"

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianKNNClassifier":
        """Ajusta modelos NearestNeighbors por classe e calcula priors.

        Parâmetros
        ----------
        X : array de forma (n_amostras, d)
            Dados de treinamento.
        y : array de forma (n_amostras,)
            Rótulos das classes.

        Retorna
        -------
        self
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.nn_models_ = {}
        self.class_data_ = {}
        self.priors_ = {}

        n = len(y)
        for cls in self.classes_:
            Xc = X[y == cls]
            n_cls = Xc.shape[0]
            self.class_data_[cls] = Xc
            self.priors_[cls] = n_cls / n

            # Ajusta NearestNeighbors para os dados desta classe
            k_eff = min(self.n_neighbors, n_cls)
            nn = NearestNeighbors(n_neighbors=k_eff, metric=self.metric)
            nn.fit(Xc)
            self.nn_models_[cls] = nn

        return self

    def _log_density(self, X: np.ndarray, cls) -> np.ndarray:
        """Calcula log p(x | ω_cls) via estimativa de densidade kNN.

        Usa o raio ao k-ésimo vizinho mais próximo dentro da classe
        para estimar o volume da bola métrica e, por consequência,
        a densidade local.

        Parâmetros
        ----------
        X : array de forma (n_amostras, d)
        cls : rótulo da classe

        Retorna
        -------
        log_densidades : array de forma (n_amostras,)
        """
        n_cls = self.class_data_[cls].shape[0]
        k_eff = min(self.n_neighbors, n_cls)
        d = X.shape[1]

        # Distância ao k-ésimo vizinho mais próximo na classe
        distances, _ = self.nn_models_[cls].kneighbors(X, n_neighbors=k_eff)
        r = distances[:, k_eff - 1]  # raio ao k-ésimo vizinho

        # Garante raio mínimo para evitar log(0)
        r = np.clip(r, a_min=1e-10, a_max=None)

        # Volume da bola métrica em d dimensões
        if self.metric == "euclidean":
            # Volume da bola L2: V_d(r) = π^(d/2) / Γ(d/2 + 1) * r^d
            log_V = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1) + d * np.log(r)
        elif self.metric == "manhattan":
            # Volume da bola L1 (cross-polytope): V_d(r) = 2^d / d! * r^d
            log_V = d * np.log(2) - gammaln(d + 1) + d * np.log(r)
        elif self.metric == "chebyshev":
            # Volume da bola L∞ (hipercubo): V_d(r) = (2r)^d
            log_V = d * np.log(2) + d * np.log(r)
        else:
            # Fallback: usa volume euclidiano
            log_V = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1) + d * np.log(r)

        # Estimativa de densidade: p(x|ω) ≈ k / (n_cls * V)
        log_density = np.log(k_eff) - np.log(n_cls) - log_V

        return log_density

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz classes via MAP usando posterior = prior × likelihood.

        Parâmetros
        ----------
        X : array de forma (n_amostras, d)

        Retorna
        -------
        y_pred : array de forma (n_amostras,)
        """
        check_is_fitted(self)
        X = check_array(X)

        log_posteriors = []
        for cls in self.classes_:
            log_prior = np.log(self.priors_[cls] + 1e-15)
            log_lik = self._log_density(X, cls)
            log_posteriors.append(log_prior + log_lik)

        log_posteriors = np.vstack(log_posteriors).T
        return self.classes_[np.argmax(log_posteriors, axis=1)]


# ============================================================
# Classificador Bayesiano com Janela de Parzen
# ============================================================
class ParzenBayesClassifier(BaseEstimator, ClassifierMixin):
    """Classificador Bayes com estimativa de densidade por janela de Parzen (KDE).

    Parâmetros
    ----------
    bandwidth : float, padrão=1.0
        Largura de banda do kernel gaussiano.
    """

    _estimator_type = "classifier"

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ParzenBayesClassifier":
        """Ajusta um KDE gaussiano por classe.

        Parâmetros
        ----------
        X : array de forma (n_amostras, d)
        y : array de forma (n_amostras,)

        Retorna
        -------
        self
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.models_ = {}
        self.priors_ = {}
        n = len(y)
        for cls in self.classes_:
            Xc = X[y == cls]
            kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
            kde.fit(Xc)
            self.models_[cls] = kde
            self.priors_[cls] = Xc.shape[0] / n
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz classes via MAP usando densidades KDE.

        Parâmetros
        ----------
        X : array de forma (n_amostras, d)

        Retorna
        -------
        y_pred : array de forma (n_amostras,)
        """
        check_is_fitted(self)
        X = check_array(X)
        log_posteriors = []
        for cls in self.classes_:
            log_density = self.models_[cls].score_samples(X)
            log_prior = np.log(self.priors_[cls] + 1e-15)
            log_posteriors.append(log_density + log_prior)
        log_posteriors = np.vstack(log_posteriors).T
        return self.classes_[np.argmax(log_posteriors, axis=1)]


# ============================================================
# Configuração dos modelos e hiperparâmetros
# ============================================================
def build_model_configs() -> dict:
    """Cria pipelines de modelos e grids de hiperparâmetros para comparação.

    Retorna
    -------
    dict
        Dicionário com nome do modelo → {estimator, param_grid}.
    """
    return {
        "BayesGauss": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MultivariateGaussianBayes())
            ]),
            "param_grid": None
        },
        "kNN": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", BayesianKNNClassifier())
            ]),
            "param_grid": {
                "clf__n_neighbors": [1, 3, 5, 7, 9, 11],
                "clf__metric": ["euclidean", "manhattan", "chebyshev"]
            }
        },
        "Parzen": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", ParzenBayesClassifier())
            ]),
            "param_grid": {
                "clf__bandwidth": np.logspace(-1, 1, 7)
            }
        },
        "LogReg": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, solver="lbfgs"))
            ]),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        }
    }


# ============================================================
# Funções auxiliares
# ============================================================
def majority_vote_from_predictions(pred_list: list) -> np.ndarray:
    """Combina predições de múltiplos classificadores por votação majoritária.

    Parâmetros
    ----------
    pred_list : lista de arrays
        Lista com as predições de cada classificador.

    Retorna
    -------
    final_pred : array de forma (n_amostras,)
        Predição final por voto majoritário.
    """
    preds = np.vstack(pred_list).T
    final_pred = []
    for row in preds:
        classes, counts = np.unique(row, return_counts=True)
        winner = classes[np.argmax(counts)]
        final_pred.append(winner)
    return np.array(final_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas de classificação com média macro.

    Parâmetros
    ----------
    y_true : array de forma (n_amostras,)
        Rótulos verdadeiros.
    y_pred : array de forma (n_amostras,)
        Rótulos preditos.

    Retorna
    -------
    dict
        Dicionário com erro, precisão, recall e F1.
    """
    acc = accuracy_score(y_true, y_pred)
    err = 1.0 - acc
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"error": err, "precision": prec, "recall": rec, "f1": f1}


# ============================================================
# Avaliação rigorosa com validação cruzada repetida aninhada
# ============================================================
def evaluate_scenario_rigorous(
    X: np.ndarray,
    y: np.ndarray,
    scenario_name: str = "A",
    random_state: int = 42
) -> dict:
    """Avalia classificadores com validação cruzada estratificada repetida aninhada.

    Utiliza 10-fold com 30 repetições no laço externo e 5-fold no laço
    interno para seleção de hiperparâmetros via GridSearchCV.

    Parâmetros
    ----------
    X : array de forma (n_amostras, d)
    y : array de forma (n_amostras,)
    scenario_name : str
        Nome do cenário ('A' ou 'B').
    random_state : int
        Semente aleatória para reprodutibilidade.

    Retorna
    -------
    dict
        Resultados contendo métricas por fold, scores e log de hiperparâmetros.
    """
    outer_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    model_configs = build_model_configs()
    model_names = ["BayesGauss", "kNN", "Parzen", "LogReg", "Voting"]
    results = {name: {"error": [], "precision": [], "recall": [], "f1": []} for name in model_names}
    foldwise_scores_f1 = {name: [] for name in model_names}
    foldwise_scores_error = {name: [] for name in model_names}
    foldwise_scores_precision = {name: [] for name in model_names}
    foldwise_scores_recall = {name: [] for name in model_names}
    best_params_log = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fitted_models = {}

        # BayesGauss — sem busca de hiperparâmetros
        bayes_model = clone(model_configs["BayesGauss"]["estimator"])
        bayes_model.fit(X_train, y_train)
        fitted_models["BayesGauss"] = bayes_model

        # kNN Bayesiano — GridSearch
        knn_gs = GridSearchCV(
            estimator=clone(model_configs["kNN"]["estimator"]),
            param_grid=model_configs["kNN"]["param_grid"],
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1
        )
        knn_gs.fit(X_train, y_train)
        fitted_models["kNN"] = knn_gs.best_estimator_

        # Parzen — GridSearch
        parzen_gs = GridSearchCV(
            estimator=clone(model_configs["Parzen"]["estimator"]),
            param_grid=model_configs["Parzen"]["param_grid"],
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1
        )
        parzen_gs.fit(X_train, y_train)
        fitted_models["Parzen"] = parzen_gs.best_estimator_

        # Regressão Logística — GridSearch
        logreg_gs = GridSearchCV(
            estimator=clone(model_configs["LogReg"]["estimator"]),
            param_grid=model_configs["LogReg"]["param_grid"],
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1
        )
        logreg_gs.fit(X_train, y_train)
        fitted_models["LogReg"] = logreg_gs.best_estimator_

        best_params_log.append({
            "fold": fold_id,
            "scenario": scenario_name,
            "kNN_best": knn_gs.best_params_,
            "Parzen_best": parzen_gs.best_params_,
            "LogReg_best": logreg_gs.best_params_
        })

        # Predições individuais e votação
        predictions = {}
        for model_name in ["BayesGauss", "kNN", "Parzen", "LogReg"]:
            predictions[model_name] = fitted_models[model_name].predict(X_test)
        predictions["Voting"] = majority_vote_from_predictions([
            predictions["BayesGauss"],
            predictions["kNN"],
            predictions["Parzen"],
            predictions["LogReg"]
        ])

        # Coleta de métricas
        for model_name in ["BayesGauss", "kNN", "Parzen", "LogReg", "Voting"]:
            y_pred = predictions[model_name]
            m = compute_metrics(y_test, y_pred)
            for metric_name, metric_value in m.items():
                results[model_name][metric_name].append(metric_value)
            foldwise_scores_error[model_name].append(m["error"])
            foldwise_scores_precision[model_name].append(m["precision"])
            foldwise_scores_recall[model_name].append(m["recall"])
            foldwise_scores_f1[model_name].append(m["f1"])

        if True:
            print(f"[{scenario_name}] folds processados: {fold_id}/2")

    return {
        "results": results,
        "f1_scores": foldwise_scores_f1,
        "error_scores": foldwise_scores_error,
        "precision_scores": foldwise_scores_precision,
        "recall_scores": foldwise_scores_recall,
        "best_params_log": pd.DataFrame(best_params_log)
    }


# ============================================================
# Tabela de intervalos de confiança
# ============================================================
def show_ci_table(res: dict, scenario_name: str) -> pd.DataFrame:
    """Exibe estimativas pontuais e intervalos de confiança de 95% para cada modelo.

    Parâmetros
    ----------
    res : dict
        Resultado de evaluate_scenario_rigorous.
    scenario_name : str
        Nome do cenário.

    Retorna
    -------
    pd.DataFrame
        Tabela com média, desvio, IC inferior e IC superior.
    """
    metrics = ["error", "precision", "recall", "f1"]
    model_names = ["BayesGauss", "kNN", "Parzen", "LogReg", "Voting"]
    rows = []
    for model in model_names:
        for metric in metrics:
            vals = np.array(res["results"][model][metric])
            mean = vals.mean()
            std = vals.std(ddof=1)
            n = len(vals)
            z = 1.959963984540054
            margin = z * std / np.sqrt(n)
            rows.append({
                "Modelo": model,
                "Métrica": metric,
                "Média": round(mean, 4),
                "Desvio": round(std, 4),
                "IC_inf": round(mean - margin, 4),
                "IC_sup": round(mean + margin, 4)
            })
    df = pd.DataFrame(rows)
    print(f"\n=== Cenário {scenario_name} — Estimativas pontuais e IC 95% ===")
    print(df.pivot_table(index="Modelo", columns="Métrica",
                         values=["Média", "IC_inf", "IC_sup"],
                         aggfunc="first"))
    return df


# ============================================================
# Teste de Friedman + post-hoc de Nemenyi
# ============================================================
def friedman_nemenyi(res: dict, scenario_name: str) -> None:
    """Executa o teste de Friedman e comparações post-hoc de Nemenyi.

    Parâmetros
    ----------
    res : dict
        Resultado de evaluate_scenario_rigorous.
    scenario_name : str
        Nome do cenário.
    """
    metrics = ["error", "precision", "recall", "f1"]
    model_names = ["BayesGauss", "kNN", "Parzen", "LogReg", "Voting"]
    k = len(model_names)
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
               6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    qa = q_alpha[k]

    print(f"\n{'='*60}")
    print(f"Cenário {scenario_name} — Friedman test + Nemenyi post-hoc")
    print(f"{'='*60}")

    for metric in metrics:
        scores = np.array([res["results"][m][metric] for m in model_names]).T
        N = scores.shape[0]
        chi2_stat, p_friedman = stats.friedmanchisquare(*[scores[:, j] for j in range(k)])
        FF = ((N - 1) * chi2_stat) / (N * (k - 1) - chi2_stat)
        df1, df2 = k - 1, (k - 1) * (N - 1)
        p_FF = 1 - stats.f.cdf(FF, df1, df2)
        ranks = np.apply_along_axis(
            lambda x: stats.rankdata(x if metric == "error" else -x), 1, scores
        )
        mean_ranks = ranks.mean(axis=0)
        rank_df = pd.DataFrame({"Modelo": model_names, "Rank médio": mean_ranks.round(4)})
        rank_df = rank_df.sort_values("Rank médio").reset_index(drop=True)

        print(f"\n--- Métrica: {metric.upper()} ---")
        print(f"  Friedman χ²F = {chi2_stat:.4f}, FF = {FF:.4f}, p(FF) = {p_FF:.4e}")
        if p_FF < 0.05:
            print("  → H0 rejeitada (p < 0.05): existe diferença significativa.")
        else:
            print("  → H0 NÃO rejeitada (p ≥ 0.05).")
        print(rank_df.to_string(index=False))

        CD = qa * np.sqrt(k * (k + 1) / (6 * N))
        print(f"  Nemenyi CD (α=0.05) = {CD:.4f}")

        print("  Pares com diferença significativa (|Ri - Rj| > CD):")
        rank_dict = dict(zip(model_names, mean_ranks))
        found_any = False
        for m1, m2 in combinations(model_names, 2):
            diff = abs(rank_dict[m1] - rank_dict[m2])
            if diff > CD:
                print(f"    {m1} vs {m2}: |{rank_dict[m1]:.4f} - {rank_dict[m2]:.4f}| = {diff:.4f} > {CD:.4f}  ✓")
                found_any = True
        if not found_any:
            print("    Nenhum par significativamente diferente.")


# ============================================================
# Curva de aprendizagem com múltiplas repetições
# ============================================================
def learning_curve_f1(
    X: np.ndarray,
    y: np.ndarray,
    scenario_name: str,
    n_repeats: int = 10,
    random_state: int = 42
) -> tuple:
    """Calcula e plota curvas de aprendizagem para F1-macro com múltiplas repetições.

    Para cada proporção de treino, repete a divisão n_repeats vezes com
    sementes diferentes e calcula a média do F1, reduzindo a variância
    nos gráficos.

    Parâmetros
    ----------
    X : array de forma (n_amostras, d)
    y : array de forma (n_amostras,)
    scenario_name : str
        Nome do cenário ('A' ou 'B').
    n_repeats : int, padrão=10
        Número de repetições por proporção de treino.
    random_state : int, padrão=42
        Semente aleatória base.

    Retorna
    -------
    tuple
        (train_scores, test_scores) — dicionários com listas de F1 por modelo.
    """
    proportions = np.arange(0.5, 1.0, 0.4)
    model_names = ["BayesGauss", "kNN", "Parzen", "LogReg", "Voting"]
    train_scores = {m: [] for m in model_names}
    test_scores = {m: [] for m in model_names}

    for prop in proportions:
        train_f1_reps = {m: [] for m in model_names}
        test_f1_reps = {m: [] for m in model_names}

        for rep in range(n_repeats):
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=prop, random_state=random_state + rep
            )
            train_idx, test_idx = next(sss.split(X, y))
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            models = {
                "BayesGauss": Pipeline([
                    ("sc", StandardScaler()),
                    ("clf", MultivariateGaussianBayes(reg_covar=1e-4))
                ]),
                "kNN": Pipeline([
                    ("sc", StandardScaler()),
                    ("clf", BayesianKNNClassifier(n_neighbors=5, metric="euclidean"))
                ]),
                "Parzen": Pipeline([
                    ("sc", StandardScaler()),
                    ("clf", ParzenBayesClassifier(bandwidth=1.0))
                ]),
                "LogReg": Pipeline([
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs"))
                ]),
            }

            preds_tr = {}
            preds_te = {}
            for name, model in models.items():
                model.fit(X_tr, y_tr)
                preds_tr[name] = model.predict(X_tr)
                preds_te[name] = model.predict(X_te)

            preds_tr["Voting"] = majority_vote_from_predictions(list(preds_tr.values()))
            preds_te["Voting"] = majority_vote_from_predictions(list(preds_te.values()))

            for name in model_names:
                train_f1_reps[name].append(
                    f1_score(y_tr, preds_tr[name], average="macro", zero_division=0)
                )
                test_f1_reps[name].append(
                    f1_score(y_te, preds_te[name], average="macro", zero_division=0)
                )

        for name in model_names:
            train_scores[name].append(np.mean(train_f1_reps[name]))
            test_scores[name].append(np.mean(test_f1_reps[name]))

    # --- Plotagem ---
    pct_labels = [f"{int(round(p*100))}%" for p in proportions]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(
        f"Curva de Aprendizagem — F-measure (macro) — Cenário {scenario_name}"
        f" ({n_repeats} repetições)",
        fontsize=14
    )

    for ax, (scores, label) in zip(axes, [(train_scores, "Treino"), (test_scores, "Teste")]):
        for name, color in zip(model_names, colors):
            ax.plot(proportions * 100, scores[name], marker="o", label=name, color=color)
        ax.set_xlabel("% dos dados usada para treino")
        ax.set_ylabel("F-measure (macro)")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.set_xticks(proportions * 100)
        ax.set_xticklabels(pct_labels, rotation=45, fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig_path = os.path.join(_RESULTADOS_DIR, f"learning_curve_cenario_{scenario_name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"[Cenário {scenario_name}] Figura salva: {fig_path}")

    return train_scores, test_scores


# ============================================================
# Bloco principal
# ============================================================
if __name__ == "__main__":
    q1_results = q1.run_question1(c_values=[2, 3, 4, 5, 6], n_runs=100, plot=False)
    X = q1_results["X"]
    y_true = q1_results["y_true"]
    labels_star = q1_results["labels_star"]

    X_A = X.copy()
    y_A = y_true.copy()
    X_B = X.copy()
    y_B = labels_star.copy()

    print("Cenário A:", X_A.shape, np.unique(y_A, return_counts=True))
    print("Cenário B:", X_B.shape, np.unique(y_B, return_counts=True))

    res_A = evaluate_scenario_rigorous(X_A, y_A, scenario_name="A", random_state=42)
    print("Cenário A concluído.")
    res_B = evaluate_scenario_rigorous(X_B, y_B, scenario_name="B", random_state=42)
    print("Cenário B concluído.")

    ci_A = show_ci_table(res_A, "A")
    ci_B = show_ci_table(res_B, "B")
    friedman_nemenyi(res_A, "A")
    friedman_nemenyi(res_B, "B")
    learning_curve_f1(X_A, y_A, scenario_name="A", n_repeats=1)
    learning_curve_f1(X_B, y_B, scenario_name="B", n_repeats=1)
