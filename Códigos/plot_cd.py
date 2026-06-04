import matplotlib.pyplot as plt
import numpy as np
import os

# Configs LaTeX-like
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def draw_cd_diagram(ranks, cd, title, filename):
    # Sort models by rank (lower is better, left to right)
    sorted_models = sorted(ranks.keys(), key=lambda x: ranks[x])
    sorted_ranks = [ranks[m] for m in sorted_models]
    
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Draw main horizontal axis
    min_rank = math.floor(min(sorted_ranks))
    max_rank = math.ceil(max(sorted_ranks))
    if min_rank == max_rank:
        min_rank -= 1
        max_rank += 1
    
    ax.hlines(0, min_rank, max_rank, color='k', lw=2)
    
    # Ticks for ranks
    for i in range(min_rank, max_rank + 1):
        ax.vlines(i, 0, 0.1, color='k')
        ax.text(i, 0.15, str(i), ha='center', va='bottom')
    
    # Draw CD bar at the top left
    ax.hlines(0.6, min_rank, min_rank + cd, color='r', lw=3)
    ax.text(min_rank + cd/2, 0.65, f'CD = {cd:.4f}', ha='center', va='bottom', color='r', fontweight='bold')
    
    # Draw models
    y_offsets = np.linspace(-0.2, -0.8, len(sorted_models))
    # Alternate left/right if crowded, but here we just drop lines
    
    for i, (model, rank) in enumerate(zip(sorted_models, sorted_ranks)):
        # vertical line from axis down to label
        y_pos = y_offsets[i]
        ax.vlines(rank, 0, y_pos, color='k', linestyles='dotted')
        # line to text
        ax.hlines(y_pos, rank, rank + 0.1, color='k')
        ax.text(rank + 0.15, y_pos, f"{model} ({rank:.2f})", va='center')
    
    # Draw thick lines for non-significant differences
    # Find cliques
    cliques = []
    for i in range(len(sorted_ranks)):
        for j in range(i+1, len(sorted_ranks)):
            if sorted_ranks[j] - sorted_ranks[i] <= cd:
                cliques.append((sorted_ranks[i], sorted_ranks[j]))
    
    # Filter maximal cliques (simplified, just merge overlapping if they are true cliques)
    # Actually, a line between min and max if max - min <= cd
    maximal_lines = []
    i = 0
    while i < len(sorted_ranks):
        j = len(sorted_ranks) - 1
        while j > i:
            if sorted_ranks[j] - sorted_ranks[i] <= cd:
                maximal_lines.append((sorted_ranks[i], sorted_ranks[j]))
                break
            j -= 1
        i += 1
    
    # Filter redundant
    final_lines = []
    for l in maximal_lines:
        covered = False
        for fl in final_lines:
            if l[0] >= fl[0] and l[1] <= fl[1]:
                covered = True
                break
        if not covered:
            final_lines.append(l)
    
    # Draw clique lines
    line_y = -0.05
    for l in final_lines:
        ax.hlines(line_y, l[0], l[1], color='blue', lw=4)
        line_y -= 0.05

    ax.set_title(title, pad=20, fontweight='bold')
    ax.axis('off')
    
    os.makedirs('../Resultados', exist_ok=True)
    plt.savefig(f'../Resultados/{filename}')
    plt.close()

import math
ranks_A = {'Voting': 1.57, 'kNN': 2.83, 'BayesGauss': 2.95, 'LogReg': 3.68, 'Parzen': 3.97}
ranks_B = {'LogReg': 1.88, 'Voting': 2.07, 'Parzen': 2.94, 'kNN': 3.12, 'BayesGauss': 5.00}

draw_cd_diagram(ranks_A, 0.3522, 'Diagrama de Diferença Crítica (CD) - Cenário A (F1)', 'cd_diagram_A.png')
draw_cd_diagram(ranks_B, 0.3522, 'Diagrama de Diferença Crítica (CD) - Cenário B (F1)', 'cd_diagram_B.png')

print("Diagramas CD gerados.")
