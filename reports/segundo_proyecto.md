# Resumen de 03_FeatureEngineering.ipynb

En el notebook `03_FeatureEngineering.ipynb`, se llevaron a cabo las siguientes actividades específicas:

1. **Limpieza de Datos**: 
    - Se eliminaron los valores 'Current' y 'Late' del dataset, ya que eran un porcentaje pequeño de los datos, y nuestro target es identificar préstamos en dos outcomes posibles: Paid y Abandoned.

2. **Selección y Transformación de Características**: 
    - Se transformó emp_length a una variable booleana employment.
    - Se identificó que personas con situación de vivienda precaria tenían más posibilidad de abandonar el préstamo, por lo tanto se creó la variable booleana 'housing_instability'.

3. **Interest to Payment Ratio**: 
    - En uno de los gráficos se visualizó que entre más de los pagos se iba al interés, más probable era que un usuario abandone el préstamo.
    - Con el fin de preservar y utilizar esta relación, se creó la variable i2p_ratio, se realizó una transformación logarítmica para comprimir sus valores, y se eliminaron los outliers para que no afecten el entrenamiento del modelo.

Estas acciones fueron diseñadas para optimizar el conjunto de datos y mejorar el rendimiento de los modelos de aprendizaje automático en etapas posteriores.

---

# Resumen de 04_Modeling.ipynb

En el notebook `04_Modeling.ipynb`, se realizaron las siguientes actividades relacionadas con la construcción y evaluación de modelos de aprendizaje automático para predecir el abandono de préstamos:

1. **Carga y Exploración del Dataset**
    - Se cargó el dataset procesado `processed_loan_data.csv`.
    - Se verificó la estructura del dataset, incluyendo información básica y la existencia de valores nulos.

2. **Modelos Iniciales**
    **Regresión Logística**
    - Se entrenó un modelo de regresión logística básico para predecir el abandono.
    - Se comparó el desempeño del modelo con y sin el uso del parámetro `class_weight='balanced'`:
        - **Sin `class_weight`**: Recall para la clase de abandono fue de **0.78**.
        - **Con `class_weight='balanced'`**: Recall mejoró significativamente a **0.91**.
    - Se concluyó que el ajuste del peso de las clases es crucial debido al desbalance en las clases del dataset.

3. **Modelos Avanzados**
    **Random Forest y Gradient Boosting**
    - Se entrenaron dos modelos avanzados:
        - **Random Forest Classifier** con `class_weight='balanced'`.
        - **Gradient Boosting Classifier`.
    - Resultados:
        - **Random Forest** mostró un mejor desempeño general, con un F1-Score de **0.96** para la clase de abandono.
        - **Gradient Boosting** tuvo un desempeño inferior en comparación con Random Forest.

4. **Optimización de Hiperparámetros**
    - Se realizó un **GridSearchCV** para optimizar los hiperparámetros del modelo Random Forest.
    - Parámetros evaluados:
        - `n_estimators`: [100, 200, 300].
        - `max_depth`: [None, 10, 20, 30].
        - `min_samples_split`: [2, 5, 10].
        - `min_samples_leaf`: [1, 2, 4].
        - `class_weight`: ['balanced', 'balanced_subsample'].
    - Resultado:
        - El mejor modelo encontrado por GridSearch tuvo un **ROC-AUC Score** ligeramente inferior al modelo inicial de Random Forest.
        - Se decidió mantener el modelo inicial debido a su desempeño superior.

5. **Importancia de las Características**
    - Se analizó la importancia de las características utilizando el atributo `feature_importances_` del modelo Random Forest.
    - Se validó que la columna `log_i2p_ratio` (proporción de pagos al interés vs. principal) es una de las características más relevantes para predecir el abandono.

---

## Métricas Obtenidas

- **Random Forest Classifier**:
    - **F1-Score (Clase de Abandono)**: **0.96**.
    - **Recall (Clase de Abandono)**: **0.94**.
    - **ROC-AUC Score**: **0.996**.

- **Regresión Logística**:
    - **F1-Score (Clase de Abandono)**: **0.94**.
    - **Recall (Clase de Abandono)**: **0.91** (con `class_weight='balanced'`).
    - **ROC-AUC Score**: **0.983**.

- **Gradient Boosting Classifier**:
    - **Recall (Clase de Abandono)**: **0.91**.
    - **F1-Score**: **0.95**.

---

## Conclusiones

- El modelo **Random Forest** con `class_weight='balanced'` fue seleccionado como el mejor modelo debido a su alto desempeño en métricas clave como el F1-Score y el ROC-AUC.
- La creación de la columna `log_i2p_ratio` fue una decisión acertada, ya que esta característica demostró ser altamente predictiva en el modelo.
- Se estableció una base sólida para la implementación del modelo en producción y para realizar recomendaciones basadas en los resultados.

---

## Lecciones Aprendidas

1. **Importancia de la Ingeniería de Características**:
    - La creación de `log_i2p_ratio` tuvo un impacto significativo en el desempeño del modelo, destacando la importancia de entender el dominio del problema.

2. **Ajuste de Peso de Clases**:
    - En datasets desbalanceados, ajustar el peso de las clases (`class_weight='balanced'`) puede mejorar significativamente métricas como el Recall, especialmente para la clase minoritaria.

3. **Limitaciones de la Optimización de Hiperparámetros**:
    - Aunque el **GridSearchCV** puede encontrar configuraciones óptimas, no siempre garantiza un mejor desempeño en métricas clave. En este caso, el modelo inicial de Random Forest superó al modelo optimizado en Gridsearch.

4. **Visualización y Análisis Exploratorio**:
    - Los gráficos y análisis exploratorios fueron esenciales para identificar patrones clave, como la relación entre pagos al interés y abandono, lo que permitió tomar decisiones informadas en la ingeniería de características.

5. **Simplicidad vs. Complejidad**:
    - Modelos más complejos como Gradient Boosting no siempre superan a modelos más simples como Random Forest, especialmente cuando el dataset está bien preparado y las características son relevantes.