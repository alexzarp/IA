import pandas as pd

data = pd.read_csv("resultado.csv")

linhas_selecionadas = data.iloc[4:9, :]
contagem_de_1s = linhas_selecionadas.sum()
ordenado = contagem_de_1s.sort_values(ascending=False)

print(ordenado)
