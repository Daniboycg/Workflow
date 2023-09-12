**Métodos y Funciones:**
1. `df.head()`: Muestra las primeras filas del DataFrame.
2. `df.describe()`: Proporciona estadísticas resumidas de las columnas numéricas.
3. `df.info()`: Muestra información sobre el DataFrame, incluyendo tipos de datos y valores no nulos.
4. `df.shape`: Devuelve el número de filas y columnas del DataFrame.
5. `df.isnull().sum()`: Calcula la cantidad de valores nulos en cada columna.
6. `df.value_counts()`: Cuenta los valores únicos en una columna.
7. `df.corr()`: Calcula la matriz de correlación entre las columnas numéricas.
  
**Gráficas:**  

8. `import matplotlib.pyplot as plt`: Importa la biblioteca de visualización Matplotlib.
9. `import seaborn as sns`: Importa la biblioteca Seaborn para visualización estadística.
10. `sns.histplot(df['columna'])`: Crea un histograma de una columna.
11. `sns.boxplot(x='columna_x', y='columna_y', data=df)`: Crea un diagrama de caja para comparar dos variables.
12. `sns.pairplot(df)`: Genera una matriz de gráficos de dispersión para explorar relaciones entre variables numéricas.
13. `plt.figure(figsize=(10, 6))`: Define el tamaño de la figura para gráficos personalizados.
14. `plt.bar(x, height)`: Crea un gráfico de barras.
15. `plt.scatter(x, y)`: Crea un gráfico de dispersión.
16. `plt.plot(x, y)`: Crea un gráfico de líneas.
17. `sns.heatmap(data.corr(), annot=True, cmap='coolwarm')`: Crea un mapa de calor de la matriz de correlación.

Utiliza estos métodos, funciones y gráficas en tu análisis exploratorio de datos para obtener información valiosa sobre tus conjuntos de datos.
