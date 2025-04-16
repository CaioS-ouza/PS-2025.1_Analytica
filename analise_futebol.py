import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados
try:
    df = pd.read_csv("Data/campeonatos_futebol_atualizacao.csv")  # Ajuste o nome/caminho se necessário
    print("Dados carregados com sucesso!")
except FileNotFoundError:
    print("Arquivo não encontrado. Verifique o caminho.")

# 2. Primeira inspeção
print("\n--- Primeiras linhas do DataFrame ---")
print(df.head())

print("\n--- Informações das colunas ---")
print(df.info())

print("\n--- Estatísticas descritivas ---")
print(df.describe())

# 3. Análise inicial dos gols (variável alvo)
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="Gols 1", color="blue", label="Time da Casa", kde=True)
sns.histplot(data=df, x="Gols 2", color="red", label="Time Visitante", kde=True)
plt.title("Distribuição de Gols por Time")
plt.legend()
plt.show()