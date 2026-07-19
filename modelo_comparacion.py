"""
modelo_comparacion.py

Proyecto: Predicción de la Resistencia a Compresión del Concreto
Grupo 06 - UPC

Objetivo:
Comparar el desempeño de dos algoritmos de aprendizaje automático
(Regresión Ridge y Random Forest) para predecir la resistencia a
compresión del concreto (MPa) a partir de sus componentes y edad
de curado, usando validación cruzada de 5 particiones (5-fold CV).

Entradas:
    - concrete_data.csv  (dataset original, en la misma carpeta)

Salidas:
    - Impresión en consola de las métricas R², RMSE y MAE por modelo
    - Impresión de la importancia de variables del Random Forest
    - comparacion_modelos.png  (gráfico de barras comparando R²)
    - importancia_variables.png (gráfico de barras de feature importance)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


# ----------------------------------------------------------------
# 1. CARGA Y LIMPIEZA DE DATOS
# ----------------------------------------------------------------
def cargar_datos(ruta_csv):
    """Carga el CSV, limpia nombres de columnas y elimina duplicados."""
    df = pd.read_csv(ruta_csv)

    # Limpiar espacios en blanco en los nombres de columnas
    # (ej: "fine_aggregate " -> "fine_aggregate")
    df.columns = df.columns.str.strip()

    n_antes = len(df)
    df = df.drop_duplicates()
    n_despues = len(df)
    print(f"Registros originales: {n_antes} | Duplicados eliminados: {n_antes - n_despues} "
          f"| Registros finales: {n_despues}")

    return df


# ----------------------------------------------------------------
# 2. ENTRENAMIENTO Y VALIDACIÓN CRUZADA
# ----------------------------------------------------------------
def evaluar_modelos(X, y, n_splits=5, random_state=42):
    """
    Entrena y evalúa Ridge y Random Forest con validación cruzada
    de n_splits particiones. Devuelve un diccionario con las métricas
    promedio de cada modelo (R², RMSE, MAE).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Métricas a calcular durante la validación cruzada
    scoring = {
        'r2': 'r2',
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
    }

    modelos = {
        'Ridge Regression': Ridge(alpha=1.0, random_state=random_state),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        cv = cross_validate(modelo, X, y, cv=kf, scoring=scoring)
        resultados[nombre] = {
            'R2': cv['test_r2'].mean(),
            'R2_std': cv['test_r2'].std(),
            'RMSE': -cv['test_rmse'].mean(),
            'MAE': -cv['test_mae'].mean(),
        }

    return resultados


# ----------------------------------------------------------------
# 3. IMPORTANCIA DE VARIABLES (Random Forest)
# ----------------------------------------------------------------
def calcular_importancia_variables(X, y, random_state=42):
    """Entrena un Random Forest sobre todo el dataset y devuelve
    la importancia de cada variable, ordenada de mayor a menor."""
    modelo = RandomForestRegressor(n_estimators=100, random_state=random_state)
    modelo.fit(X, y)
    importancias = pd.Series(modelo.feature_importances_, index=X.columns)
    return importancias.sort_values(ascending=False)


# ----------------------------------------------------------------
# 4. VISUALIZACIÓN DE RESULTADOS
# ----------------------------------------------------------------
def graficar_comparacion(resultados, ruta_salida='comparacion_modelos.png'):
    """Genera un gráfico de barras comparando el R² de cada modelo."""
    nombres = list(resultados.keys())
    r2_valores = [resultados[n]['R2'] for n in nombres]

    plt.figure(figsize=(6, 5))
    barras = plt.bar(nombres, r2_valores, color=['#4C72B0', '#55A868'])
    plt.ylabel('R² (validación cruzada, 5-fold)')
    plt.title('Comparación de Modelos: Ridge vs Random Forest')
    plt.ylim(0, 1)
    for barra, valor in zip(barras, r2_valores):
        plt.text(barra.get_x() + barra.get_width() / 2, valor + 0.02,
                  f'{valor:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    print(f"Gráfico guardado en: {ruta_salida}")


def graficar_importancia(importancias, ruta_salida='importancia_variables.png'):
    """Genera un gráfico de barras horizontales con la importancia
    de cada variable según el Random Forest."""
    plt.figure(figsize=(8, 6))
    importancias.sort_values().plot(kind='barh', color='#C44E52')
    plt.xlabel('Importancia relativa')
    plt.title('Importancia de Variables (Random Forest)')
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    print(f"Gráfico guardado en: {ruta_salida}")


# ----------------------------------------------------------------
# 5. PROGRAMA PRINCIPAL
# ----------------------------------------------------------------
def main():
    RUTA_CSV = 'concrete_data.csv'
    COLUMNA_OBJETIVO = 'concrete_compressive_strength'

    print("=" * 60)
    print("CARGANDO Y LIMPIANDO DATOS")
    print("=" * 60)
    df = cargar_datos(RUTA_CSV)

    X = df.drop(columns=[COLUMNA_OBJETIVO])
    y = df[COLUMNA_OBJETIVO]

    print("\n" + "=" * 60)
    print("EVALUANDO MODELOS (5-FOLD CROSS-VALIDATION)")
    print("=" * 60)
    resultados = evaluar_modelos(X, y)

    print("\n{:<20} {:>10} {:>12} {:>12}".format("Modelo", "R2", "RMSE (MPa)", "MAE (MPa)"))
    print("-" * 56)
    for nombre, r in resultados.items():
        print("{:<20} {:>10.3f} {:>12.2f} {:>12.2f}".format(
            nombre, r['R2'], r['RMSE'], r['MAE']))

    print("\n" + "=" * 60)
    print("IMPORTANCIA DE VARIABLES (Random Forest)")
    print("=" * 60)
    importancias = calcular_importancia_variables(X, y)
    for variable, valor in importancias.items():
        print(f"{variable:<25} {valor:.4f}")

    print("\n" + "=" * 60)
    print("GENERANDO GRÁFICOS")
    print("=" * 60)
    graficar_comparacion(resultados)
    graficar_importancia(importancias)

    print("\n¡Proceso completado con éxito!")


if __name__ == '__main__':
    main()
