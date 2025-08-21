# Guía Maestra del Proyecto: Sistema Integral Multi-Capa para la Detección de Desinformación

*Versión 2.0 - Edición Detallada*

## 1. Filosofía y Objetivos del Proyecto

Este documento es la guía maestra para la construcción de un sistema avanzado de detección de desinformación. El enfoque va más allá de una simple clasificación binaria. Buscamos crear un sistema holístico que responda a tres preguntas clave:

1.  **¿QUÉ?** -> ¿Es esta noticia falsa o verdadera? (Clasificación de Contenido)
2.  **¿QUIÉN?** -> ¿La fuente que publica la noticia es creíble? ¿Es un bot? (Análisis de la Fuente)
3.  **¿POR QUÉ?** -> ¿Qué factores (palabras clave, características del autor) llevaron al sistema a su conclusión? (Explicabilidad)

El objetivo final es desarrollar un prototipo de sistema robusto, interpretable y multi-capa, utilizando el `dataset1` como núcleo y el `dataset2` para un análisis contextual final.

## 2. Configuración del Entorno de Desarrollo

(Sin cambios, la sección anterior era correcta)

## 3. Fases de Ejecución: Guía Paso a Paso

---

### **Fase 0: El Fundamento - Limpieza y Preparación de Datos**

*   **🎯 Objetivo:** Transformar los datos crudos en conjuntos de datos limpios, validados y listos para el modelado. El 90% del éxito de un proyecto de ML reside en esta fase.

#### **Actividad 0.1: Preparación de `dataset1`**

*   **📝 Plan de Acción Detallado:**

    1.  **Análisis de Nulos y Ceros (Validación Semántica):**
        *   **Justificación:** No podemos confiar ciegamente en los datos. Un valor de 0 o `NaN` puede significar cosas distintas. Debemos entenderlo antes de actuar.
        *   **Cómo:** Usar `df.info()`, `df.isnull().sum()`, y `df.describe().transpose()` para obtener un panorama general. Luego, generar histogramas (`df.hist()`) para visualizar la distribución de cada variable y detectar picos anómalos en cero.
        *   **Conclusión Esperada:** Para este dataset, la mayoría de los ceros son **significativos** (ej. 0 adjetivos en un texto, 0% de entidades de un tipo). No los trataremos como nulos, pero este paso de validación es una práctica indispensable.

    2.  **Análisis de Desbalance de Clases (Paso Crítico):**
        *   **Justificación:** Si tenemos un 95% de noticias verdaderas y un 5% de falsas, un modelo ingenuo que siempre prediga "Verdadero" tendrá un 95% de accuracy, pero será inútil. Debemos saber si nuestras clases están desbalanceadas para elegir las métricas y técnicas de modelado correctas.
        *   **Cómo:** `print(df1['BinaryNumTarget'].value_counts(normalize=True))`. 
        *   **Acción:** Si hay desbalance, usaremos `stratify` en `train_test_split` y métricas como el **F1-Score** en lugar de la exactitud (accuracy).

    3.  **Limpieza y Selección de Columnas:**
        *   **Justificación:** Evitar el *data leakage* y eliminar datos no relevantes para la tarea de modelado clásica.
        *   **Cómo:**
            ```python
            # Data Leakage: 'majority_target' es una versión del target. Eliminarla es obligatorio.
            df1.drop(columns=['majority_target'], inplace=True)

            # Columnas de texto: No se usan en el modelo clásico. Se guardan para la fase de NLP.
            text_data = df1[['statement', 'tweet']]
            df1.drop(columns=['statement', 'tweet'], inplace=True)
            ```

    4.  **Imputación de Valores Faltantes (NaNs):**
        *   **Justificación:** Los algoritmos de ML no pueden manejar valores `NaN`. Necesitamos una estrategia para rellenarlos.
        *   **Cómo:** Usar la **mediana** es preferible a la media, ya que no se ve afectada por valores extremos (outliers).
            ```python
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            # Aplicar solo a las columnas que lo necesiten
            df1['followers_count'] = imputer.fit_transform(df1[['followers_count']]) 
            ```

    5.  **Escalado de Características Numéricas:**
        *   **Justificación:** Algoritmos como SVM o Regresión Logística son sensibles a la escala de las variables. Estandarizar los datos ayuda a que el entrenamiento sea más rápido y eficiente.
        *   **Cómo:** `StandardScaler` es la opción estándar. Transforma los datos para que tengan una media de 0 y desviación estándar de 1.
            ```python
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            numeric_cols = df1.select_dtypes(include=np.number).columns.drop('BinaryNumTarget')
            df1[numeric_cols] = scaler.fit_transform(df1[numeric_cols])
            ```

*   **✅ Entregable:** Un DataFrame `df1_processed` listo para el modelado.

---

### **Fase 1: El Motor de Clasificación de Contenido**

*   **🎯 Objetivo:** Construir y evaluar nuestros dos modelos predictivos principales.

#### **Actividad 1.1: Modelo con Features Tradicionales (XGBoost)**

*   **📝 Pasos:**
    1.  **División Estratificada:** Dividir `df1_processed` en entrenamiento y prueba. Usar `stratify=y` es crucial si hay desbalance de clases para mantener la misma proporción en ambos conjuntos.
        ```python
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        ```
    2.  **Entrenamiento del Modelo:** Entrenar `XGBClassifier`. Si hay desbalance, se puede usar el parámetro `scale_pos_weight` para dar más importancia a la clase minoritaria.
    3.  **Evaluación Rigurosa:** Usar `classification_report` (que incluye F1-score) y la matriz de confusión para entender el rendimiento en cada clase.

#### **Actividad 1.2: Modelo con NLP (Transformer - BERT)**

*   **Justificación:** Mientras que XGBoost ve los datos como una bolsa de números, BERT lee y entiende el texto, capturando sarcasmo, contexto y relaciones semánticas. Es un enfoque mucho más profundo.
*   **Consejo Profesional:** Fine-tuning de BERT es computacionalmente intensivo. Es recomendable empezar con una muestra pequeña del dataset (ej. 1000 filas) para asegurar que todo el pipeline funciona, antes de lanzarlo sobre el conjunto de datos completo, preferiblemente en un entorno con GPU (como Google Colab).
*   **📝 Pasos:**
    1.  **Carga y Preparación:** Usar la librería `datasets` de Hugging Face. Cargar un modelo pre-entrenado **en español** es clave (`dccuchile/bert-base-spanish-wwm-uncased` es una excelente opción).
    2.  **Fine-tuning:** Seguir el proceso de tokenización y entrenamiento. La librería `Trainer` de Hugging Face simplifica mucho este proceso y maneja la optimización por nosotros.

---

### **Fase 2: El Módulo de Análisis de la Fuente**

*   **🎯 Objetivo:** Ir más allá del contenido y evaluar al mensajero.

*   **📝 Pasos:**
    1.  **Puntuación de Credibilidad (Target Encoding con Precaución):**
        *   **Justificación:** La idea de reemplazar a un autor por su ratio histórico de veracidad es potente. Esto se llama *Target Encoding*.
        *   **Advertencia:** Hacer esto ingenuamente puede causar *overfitting*. Un método más seguro es calcular estos scores usando solo el conjunto de entrenamiento y luego aplicarlos al de prueba, o usar una estrategia de validación cruzada.
    2.  **Detector de Bots:** Entrenar un clasificador para predecir `BotScoreBinary`.

---

### **Fase 3, 4 y 5 (Sinopsis Mejorada)**

*   **Fase 3 (Modelo Híbrido):**
    *   **Justificación:** Unimos los mundos: el análisis numérico de XGBoost, la comprensión semántica de BERT y la credibilidad de la fuente. El meta-modelo aprende a ponderar la opinión de cada experto para tomar la mejor decisión.
*   **Fase 4 (Explicabilidad - XAI):**
    *   **Justificación:** Un modelo que dice "Falso" sin más, es una caja negra. Un modelo que dice "Falso PORQUE el autor tiene muy baja credibilidad Y el texto usa un lenguaje emocionalmente cargado" es una herramienta de análisis.
*   **Fase 5 (Estudio Contextual):**
    *   **Justificación:** Este es el paso final de la ciencia de datos: la síntesis. Conectamos nuestros hallazgos con un problema más amplio (comportamiento en redes), generando hipótesis que podrían guiar futuras investigaciones.

## 4. Estructura de Carpetas Sugerida

(Sin cambios, la estructura propuesta es robusta)

Este documento es ahora un plan de batalla mucho más completo y profesional. Refleja el proceso de pensamiento iterativo y cuidadoso que requiere un proyecto de esta magnitud.