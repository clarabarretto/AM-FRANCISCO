import math

data_A = {
    "Bayes Gaussiano": {"err": [0.1057, 0.1118], "f1": [0.8751, 0.8814], "prec": [0.8753, 0.8815], "rec": [0.8856, 0.8915]},
    "Regressão Log.": {"err": [0.1155, 0.1214], "f1": [0.8526, 0.8597], "prec": [0.8859, 0.8920], "rec": [0.8403, 0.8477]},
    "Parzen": {"err": [0.1214, 0.1273], "f1": [0.8403, 0.8478], "prec": [0.8969, 0.9022], "rec": [0.8229, 0.8306]},
    "Voto Majoritário": {"err": [0.0597, 0.0644], "f1": [0.9222, 0.9276], "prec": [0.9378, 0.9425], "rec": [0.9141, 0.9201]},
    "$k$-NN Bayesiano": {"err": [0.0869, 0.0920], "f1": [0.8868, 0.8930], "prec": [0.9234, 0.9280], "rec": [0.8714, 0.8782]}
}

data_B = {
    "Bayes Gaussiano": {"err": [0.2362, 0.2441], "f1": [0.6703, 0.6831], "prec": [0.7813, 0.7961], "rec": [0.6421, 0.6541]},
    "Regressão Log.": {"err": [0.0366, 0.0402], "f1": [0.9539, 0.9586], "prec": [0.9595, 0.9640], "rec": [0.9557, 0.9604]},
    "Parzen": {"err": [0.0632, 0.0679], "f1": [0.9192, 0.9256], "prec": [0.9474, 0.9525], "rec": [0.9093, 0.9165]},
    "Voto Majoritário": {"err": [0.0410, 0.0451], "f1": [0.9378, 0.9439], "prec": [0.9616, 0.9663], "rec": [0.9276, 0.9346]},
    "$k$-NN Bayesiano": {"err": [0.0668, 0.0714], "f1": [0.9168, 0.9227], "prec": [0.9375, 0.9427], "rec": [0.9131, 0.9196]}
}

def format_cell(vals):
    inf, mean = vals
    sup = mean + (mean - inf)
    return f"{mean:.3f} [{inf:.3f}, {sup:.3f}]"

def gen_table(scenario_name, data):
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("    \\centering")
    lines.append(f"    \\caption{{Estimativas Pontuais e IC (95\\%) de todas as métricas — Cenário {scenario_name}.}}")
    lines.append(f"    \\label{{tab:resultados_{scenario_name}}}")
    lines.append("    \\resizebox{\\textwidth}{!}{")
    lines.append("    \\begin{tabular}{@{}lcccc@{}}")
    lines.append("        \\toprule")
    lines.append("        \\textbf{Classificador} & \\textbf{Erro} & \\textbf{Precisão (macro)} & \\textbf{Recall (macro)} & \\textbf{F-measure (macro)} \\\\")
    lines.append("        \\midrule")
    
    # Order
    order = ["Bayes Gaussiano", "$k$-NN Bayesiano", "Parzen", "Regressão Log.", "Voto Majoritário"]
    for mod in order:
        c_err = format_cell(data[mod]['err'])
        c_prec = format_cell(data[mod]['prec'])
        c_rec = format_cell(data[mod]['rec'])
        c_f1 = format_cell(data[mod]['f1'])
        lines.append(f"        {mod:<17} & {c_err} & {c_prec} & {c_rec} & {c_f1} \\\\")
        
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("    }")
    lines.append("\\end{table}")
    return "\n".join(lines)

full_code = gen_table("A (Rótulos a Priori)", data_A) + "\n\n" + gen_table("B (Clusters)", data_B)
with open("tabelas_full.tex", "w") as f:
    f.write(full_code)
print("GERADO")
