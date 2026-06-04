# Plano de Implementação — Revisão do Projeto AM 2026-1

> **Autor da revisão**: Assistente AI  
> **Data**: 03/06/2026  
> **Arquivos originais**: `Códigos/q1.py`, `Códigos/q2.py`  
> **Arquivos revisados**: `Códigos/q1_v2.py`, `Códigos/q2_v2.py`  
> **Os arquivos originais NÃO foram modificados.**

---

## Sumário

- [Visão Geral](#visão-geral)
- [Questão 1 — q1_v2.py](#questão-1--q1_v2py)
- [Questão 2 — q2_v2.py](#questão-2--q2_v2py)
- [Resumo das Mudanças](#resumo-das-mudanças)
- [Como Executar](#como-executar)

---

## Visão Geral

O projeto implementa:
- **Q1**: Algoritmo KCM-K-GH (Gaussian Kernel C-Means Hard Clustering com hiper-parâmetros de largura automáticos) sobre o dataset Ionosphere.
- **Q2**: Avaliação e comparação de 5 classificadores com validação cruzada 30×10-folds, testes estatísticos (Friedman + Nemenyi) e curvas de aprendizagem.

A revisão identificou que o código original está **algoritmicamente correto** nas equações principais (Eq. 14, 16, 18 do paper). As melhorias focam em:
1. Correção conceitual do classificador kNN (deve ser bayesiano, não discriminativo)
2. Robustez numérica
3. Qualidade dos resultados experimentais

---

## Questão 1 — q1_v2.py

### Mudanças Implementadas

#### 1. Padronização opcional (`scale` parameter)
- **O quê**: Adicionado parâmetro `scale=True` à função `run_question1()`.
- **Por quê**: O algoritmo KCM-K-GH já aprende a importância relativa das variáveis através dos hiper-parâmetros s²ⱼ. A padronização prévia com `StandardScaler` pode interferir nessa interpretação. Com `scale=False`, os dados originais são usados diretamente.
- **Como**: Quando `scale=False`, o `StandardScaler` é ignorado e `X_scaled = X`.

#### 2. Inicialização inteligente de s²
- **O quê**: Quando `scale=False`, s² é inicializado com a variância de cada variável (`np.var(X, axis=0)`) em vez de `np.ones(p)`.
- **Por quê**: Para dados em escala original, `s²=1` pode ser uma escolha ruim se as variáveis têm magnitudes muito diferentes. Usar a variância como ponto de partida dá ao algoritmo uma inicialização mais razoável.

#### 3. Tratamento de clusters vazios
- **O quê**: Após cada etapa de alocação, se um cluster ficar vazio, seu protótipo é reinicializado com um ponto aleatório do maior cluster.
- **Por quê**: Clusters vazios fazem o algoritmo desperdiçar um grupo e podem levar a resultados sub-ótimos. A reinicialização permite que o cluster "renasça" em uma região densa.

#### 4. Salvamento das figuras em `Resultados/`
- **O quê**: Figuras agora são salvas automaticamente na pasta `Resultados/` (criada se não existir).
- **Arquivos gerados**:
  - `Resultados/silhueta_vs_clusters.png`
  - `Resultados/convergencia_funcao_objetivo.png`

### O que NÃO mudou
- Lógica do algoritmo KCM-K-GH (equações 14, 16, 18)
- Função objetivo e critério de convergência
- Estrutura de retorno da função `run_question1()`
- Seleção de c* pela silhueta
- Cálculo do ARI e matriz de confusão

---

## Questão 2 — q2_v2.py

### Mudanças Implementadas

#### 1. 🔴 Classificador Bayesiano baseado em kNN (`BayesianKNNClassifier`)
- **O quê**: Nova classe `BayesianKNNClassifier` substituindo o `KNeighborsClassifier` do sklearn.
- **Por quê**: O professor pede explicitamente um **"classificador bayesiano baseado em k-vizinhos"**, que deve:
  1. Estimar a densidade p(x|ωᵢ) **por classe** usando kNN
  2. Aplicar a regra de Bayes: P(ωᵢ|x) ∝ p(x|ωᵢ) · P(ωᵢ)
  
  O `KNeighborsClassifier` do sklearn é um classificador discriminativo direto (voto majoritário entre vizinhos), **não** um classificador bayesiano.

- **Implementação**:
  - Para cada classe ωᵢ, ajusta um `NearestNeighbors` nos dados da classe
  - Para estimar p(x|ωᵢ), calcula a distância ao k-ésimo vizinho mais próximo **dentro da classe** e usa:
    
    ```
    p̂(x|ωᵢ) = k / (nᵢ · V_d(r_k))
    ```
    
    onde `V_d(r_k)` é o volume da bola d-dimensional com raio `r_k` na métrica usada:
    - **Euclidiana (L2)**: V = π^(d/2) · r^d / Γ(d/2 + 1)
    - **Manhattan (L1)**: V = 2^d · r^d / d!
    - **Chebyshev (L∞)**: V = (2r)^d
  
  - Grid search sobre: `n_neighbors ∈ {1, 3, 5, 7, 9, 11}` e `metric ∈ {euclidean, manhattan, chebyshev}`

#### 2. Regularização melhorada do Bayes Gaussiano
- **O quê**: `reg_covar` padrão alterado de `1e-6` para `1e-4`. Adicionada verificação do número de condição da covariância — se `cond > 1e10`, usa estimador Ledoit-Wolf (shrinkage).
- **Por quê**: Com 34 features e folds pequenos (~315 amostras de treino), a matriz de covariância pode ficar quase singular. O shrinkage de Ledoit-Wolf é mais robusto.

#### 3. Curva de aprendizagem com múltiplas repetições
- **O quê**: Parâmetro `n_repeats=10` adicionado a `learning_curve_f1()`. Para cada proporção de treino, são feitas 10 repetições com seeds diferentes e a F1 é a **média** sobre as repetições.
- **Por quê**: Com um único split por proporção, os resultados são muito ruidosos (visível nos gráficos originais, especialmente para proporções pequenas). A média reduz a variância e produz curvas mais suaves e interpretáveis.

#### 4. Salvamento das figuras em `Resultados/`
- **O quê**: Curvas de aprendizagem salvas em `Resultados/learning_curve_cenario_{A,B}.png`.

#### 5. Importação de q1_v2
- **O quê**: `import q1_v2 as q1` em vez de `import q1`.

### O que NÃO mudou
- `MultivariateGaussianBayes` (estrutura geral — apenas reg_covar e fallback para shrinkage)
- `ParzenBayesClassifier` (sem alteração)
- `evaluate_scenario_rigorous` (mesma lógica de nested CV 30×10)
- `show_ci_table` (mesma lógica de IC com z = 1.96)
- `friedman_nemenyi` (mesma lógica com FF corrigido e q_alpha)
- `majority_vote_from_predictions` (sem alteração)
- `compute_metrics` (sem alteração)
- Bloco `__main__` (mesma lógica: Cenário A = classes originais, Cenário B = clusters da Q1)

---

## Resumo das Mudanças

| Prioridade | Arquivo | Mudança | Status |
|:---:|---------|---------|:------:|
| 🔴 Alta | q2_v2.py | Classificador Bayesiano kNN (BayesianKNNClassifier) | ✅ |
| 🟡 Média | q1_v2.py | Padronização opcional (parâmetro `scale`) | ✅ |
| 🟡 Média | q1_v2.py | Inicialização inteligente de s² | ✅ |
| 🟡 Média | q1_v2.py | Tratamento de clusters vazios | ✅ |
| 🟡 Média | q2_v2.py | Regularização melhorada (reg_covar + Ledoit-Wolf) | ✅ |
| 🟡 Média | q2_v2.py | Curva de aprendizagem com múltiplas repetições | ✅ |
| 🟢 Baixa | ambos | Salvamento de figuras em `Resultados/` | ✅ |

---

## Notas para o Relatório

### Pontos a mencionar:

1. **Sobre a padronização na Q1**: O KCM-K-GH já lida com a escala das variáveis via s²ⱼ. Se a padronização for mantida, justifique que é para estabilidade numérica.

2. **Sobre o IC na Q2**: Os 300 folds da validação cruzada **não são independentes** (há overlap dos dados de treino). O IC calculado com z=1.96 pode ser otimista (estreito demais). Isso é uma limitação conhecida.

3. **Sobre o Friedman test**: O paper de Demšar (2006) aplica o Friedman test sobre N **datasets**, não sobre folds de CV. Usar N=300 (folds) infla o poder estatístico. Documente essa ressalva.

4. **Sobre o kernel produto do Parzen**: O `KernelDensity` do sklearn com `kernel="gaussian"` e um único bandwidth é **equivalente** ao kernel multivariado produto com bandwidths iguais, pois N(0, h²I) = ∏ⱼ N(0, h²).

---

## Como Executar

```bash
cd Códigos/

# Questão 1 (versão revisada)
python q1_v2.py

# Questão 2 (versão revisada — roda Q1 internamente)
python q2_v2.py
```

**Requisitos**: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```
