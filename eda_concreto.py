import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

print("Cargando datos...")
df = pd.read_csv('concrete_data.csv')

# Calcular la Relación Agua/Cemento
df['Relacion_AC'] = df['water'] / df['cement']

print("Generando Gráfico 1...")
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Relacion_AC', y='concrete_compressive_strength', data=df, hue='age', palette='viridis', alpha=0.7)
plt.title('Relacion Agua/Cemento vs Resistencia a Compresion')
plt.xlabel('Relacion Agua / Cemento (w/c)')
plt.ylabel('Resistencia (MPa)')
plt.grid(True)
plt.tight_layout()
plt.savefig('grafico_relacion_ac.png')
plt.show()

print("Generando Gráfico 2...")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlacion de Componentes')
plt.tight_layout()
plt.savefig('matriz_correlacion.png')
plt.show()

print("¡Proceso terminado con éxito!")
