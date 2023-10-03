> Te explico el contexto: Soy un cientifico de datos especializado en XGBoost para aprendizaje supervisado en python. Sklearn es quizá la librería más importante en la creación de modelos. Un diagrama de pareto dice que el 80% de los resultados provienen del 20% de las causas. dame ese 20% de las causas al crear modelos de apredizaje supervisado. El output me lo darás en orden secuencial como seguirías el workflow en la creación de un modelo de regression y clasificacion, ya sea la importacion de metodos, splitting, training, validacion, evaluacion, etc. Quiero que la creación de cada modelo me la separes en 3 diferentes. En la primera utilizaras regresion lineal y regresion logistica, en la segunda random forest regression and classification y la tercera xgboost regresor and classification, entendido??

### Importación de bibliotecas y  division de datos:
```py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

data = pd.read_csv('data.csv')
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

1. **Linear Regression and Logistic Regression**
```py
'''Linear Regression'''
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Crear un pipeline con estandarización y características polinómicas
linear_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarizar características
    ('poly', PolynomialFeatures(degree=2)),  # Generar características polinómicas
    ('linear_model', LinearRegression())  # Modelo de Regresión Lineal
])

# Ajustar el modelo con el pipeline
linear_pipeline.fit(X_train, y_train)

# Predecir y evaluar
y_pred_regression = linear_pipeline.predict(X_test)
mse_regression = mean_squared_error(y_test, y_pred_regression)
print("MSE (Regresión Lineal con Preprocesamiento):", mse_regression)


'''Logistic Regression'''
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Crear un pipeline con estandarización y características polinómicas
logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarizar características
    ('poly', PolynomialFeatures(degree=2)),  # Generar características polinómicas
    ('logistic_model', LogisticRegression())  # Modelo de Regresión Logística
])

# Ajustar el modelo con el pipeline
logistic_pipeline.fit(X_train, y_train)

# Predecir y evaluar
y_pred_classification = logistic_pipeline.predict(X_test)
accuracy_classification = accuracy_score(y_test, y_pred_classification)
print("Accuracy (Regresión Logística con Preprocesamiento):", accuracy_classification)
```
2. **Random Forest con hiperparametros comunes**
Para Random Forest, algunos hiperparámetros comunes son:

- `n_estimators`: El número de árboles en el bosque.
- `max_depth`: La profundidad máxima de cada árbol.
- `min_samples_split`: El número mínimo de muestras requeridas para dividir un nodo interno.
- `min_samples_leaf`: El número mínimo de muestras requeridas para ser una hoja.
- `max_features`: El número máximo de características a considerar para dividir en cada árbol.

```py
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Modelo de Regresión con Random Forest
random_forest_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='auto', random_state=42)
random_forest_regressor.fit(X_train, y_train)
y_pred_rf_regression = random_forest_regressor.predict(X_test)
mse_rf_regression = mean_squared_error(y_test, y_pred_rf_regression)
print("MSE (Random Forest Regresión):", mse_rf_regression)

# Modelo de Clasificación con Random Forest (clasificación binaria)
random_forest_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='auto', random_state=42)
random_forest_classifier.fit(X_train, y_train)
y_pred_rf_classification = random_forest_classifier.predict(X_test)
accuracy_rf_classification = accuracy_score(y_test, y_pred_rf_classification)
print("Accuracy (Random Forest Clasificación):", accuracy_rf_classification)
```

3. **XGBoost con Hiperparámetros Comunes**
- `n_estimators`: El número de árboles (o rondas de boosting).
- `max_depth`: La profundidad máxima de cada árbol.
- `learning_rate`: La tasa de aprendizaje que controla la contribución de cada árbol.
- `subsample`: La fracción de muestras utilizadas para entrenar cada árbol.
- `colsample_bytree`: La fracción de características utilizadas para entrenar cada árbol.
- `gamma`: Un parámetro de regularización que controla la complejidad del modelo.
```py
import xgboost as xgb

# Modelo de Regresión con XGBoost
xgboost_regressor = xgb.XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.1, subsample=1.0, 
    colsample_bytree=1.0, gamma=0.0, random_state=42)
xgboost_regressor.fit(X_train, y_train)
y_pred_xgboost_regression = xgboost_regressor.predict(X_test)
mse_xgboost_regression = mean_squared_error(y_test, y_pred_xgboost_regression)
print("MSE (XGBoost Regresión):", mse_xgboost_regression)

# Modelo de Clasificación con XGBoost (clasificación binaria)
xgboost_classifier = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1, subsample=1.0, 
    colsample_bytree=1.0, gamma=0.0, random_state=42)
xgboost_classifier.fit(X_train, y_train)
y_pred_xgboost_classification = xgboost_classifier.predict(X_test)
accuracy_xgboost_classification = accuracy_score(y_test, y_pred_xgboost_classification)
print("Accuracy (XGBoost Clasificación):", accuracy_xgboost_classification)
```