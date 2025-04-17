import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados
try:
    df = pd.read_csv("Data/campeonatos_futebol_atualizacao.csv")
    print("Dados carregados com sucesso!")
    print("\nColunas disponíveis:", df.columns.tolist())  # Mostrar todas as colunas disponíveis
except FileNotFoundError:
    print("Arquivo não encontrado. Verifique o caminho.")
    exit()
    
# Tratamento de NaN (exemplo: preencher com mediana para numéricas e moda para categóricas)
for col in df.select_dtypes(include=['float64', 'int64']):
    df[col].fillna(df[col].median(), inplace=True)  # Preenche numéricas com mediana

for col in df.select_dtypes(include=['object']):
    df[col].fillna(df[col].mode()[0], inplace=True)  # Preenche categóricas com moda

print("\n--- Valores ausentes após tratamento ---")
print(df.isnull().sum())  # Deve mostrar 0 NaN

# Detectar colunas numéricas automaticamente
cols_numericas = df.select_dtypes(include=['int64', 'float64']).columns

# Função para encontrar índices com outliers usando IQR
def get_outlier_indices(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] < lower) | (df[column] > upper)].index

# Coletar todos os índices de outliers
outlier_indices = set()
for col in cols_numericas:
    indices = get_outlier_indices(df, col)
    outlier_indices.update(indices)

# Remover apenas uma vez
print(f"Número de linhas removidas por outliers: {len(outlier_indices)}")
df = df.drop(index=outlier_indices).reset_index(drop=True)

print("Outliers removidos de todas as colunas numéricas usando IQR!")



# 2. Primeira inspeção
print("\n--- Primeiras linhas do DataFrame ---")
print(df.head())

print("\n--- Informações das colunas ---")
print(df.info())

# 3. Verificar colunas numéricas para a matriz de correlação
colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("\nColunas numéricas disponíveis:", colunas_numericas)

# # 4. Matriz de correlação apenas com colunas numéricas existentes
# plt.figure(figsize=(16, 12))
# sns.heatmap(df[colunas_numericas].corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlação entre Variáveis Numéricas")
# plt.tight_layout()
# plt.show()

cols = ['Chutes a gol 1', 'Gols 1', 'Posse 1(%)', 'Escanteios 1', 
        'Faltas 1', 'Cartões amarelos 1', 'Cartões vermelhos 1',
        'Chutes a gol 2', 'Gols 2', 'Posse 2(%)', 'Escanteios 2', 
        'Faltas 2', 'Cartões amarelos 2', 'Cartões vermelhos 2']

plt.figure(figsize=(10,8))
sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlação entre Variáveis Selecionadas")
plt.show()


# 5. Análise de valores ausentes
print("\n--- Valores ausentes por coluna ---")
print(df.isnull().sum())

# 6. Análise de distribuição dos gols
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# Distribuição de Gols - Time da Casa
plt.subplot(1, 2, 1)
ax1 = sns.histplot(df['Gols 1'], bins=range(0, 10), color='blue')
plt.title('Distribuição de Gols - Time da Casa')
for p in ax1.patches:
    height = p.get_height()
    ax1.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                 ha='center', va='bottom', fontsize=9)


plt.subplot(1, 2, 2)
# Distribuição de Gols - Time Visitante
ax2 = sns.histplot(df['Gols 2'], bins=range(0, 10), color='red')
plt.title('Distribuição de Gols - Time Visitante')
for p in ax2.patches:
    height = p.get_height()
    ax2.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                 ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 7. Criar variável de resultado
df['Resultado'] = df.apply(lambda x: 'Casa' if x['Gols 1'] > x['Gols 2'] else 
                          ('Empate' if x['Gols 1'] == x['Gols 2'] else 'Fora'), axis=1)

# 8. Análise de resultados
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Resultado', order=['Casa', 'Empate', 'Fora'], palette={'Casa':'blue', 'Empate':'gray', 'Fora':'red'})
plt.title('Distribuição de Resultados')

# Adicionar os rótulos nas barras
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10)

plt.show()


# # 9. Análise de relações entre variáveis
# variaveis_analise = ['Chutes a gol 1', 'Chutes a gol 2', 'Escanteios 1', 'Escanteios 2',
#                      'Cartões amarelos 1', 'Cartões amarelos 2', 'Gols 1', 'Gols 2']

# sns.pairplot(df[variaveis_analise + ['Resultado']], hue='Resultado', palette={'Casa':'blue', 'Empate':'gray', 'Fora':'red'})
# plt.suptitle('Relação entre Variáveis por Resultado', y=1.02)
# plt.show()

# 9. Análises mais diretas substituindo pairplot

# Boxplot: Chutes a gol do time da casa por resultado
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Resultado', y='Chutes a gol 1', palette={'Casa': 'blue', 'Empate': 'gray', 'Fora': 'red'})
plt.title('Chutes a Gol do Time da Casa por Resultado')
plt.xlabel('Resultado')
plt.ylabel('Chutes a Gol (Time da Casa)')
plt.show()

# Boxplot: Cartões amarelos do time visitante por resultado
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Resultado', y='Cartões amarelos 2', palette={'Casa': 'blue', 'Empate': 'gray', 'Fora': 'red'})
plt.title('Cartões Amarelos do Time Visitante por Resultado')
plt.xlabel('Resultado')
plt.ylabel('Cartões Amarelos (Time Visitante)')
plt.show()

# Scatterplot: Chutes a gol vs. Gols marcados (time da casa)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Chutes a gol 1', y='Gols 1', hue='Resultado',
                palette={'Casa': 'blue', 'Empate': 'gray', 'Fora': 'red'}, alpha=0.7)
plt.title('Relação entre Chutes a Gol e Gols Marcados (Time da Casa)')
plt.xlabel('Chutes a Gol (Time da Casa)')
plt.ylabel('Gols Marcados (Time da Casa)')
plt.legend(title='Resultado')
plt.show()
