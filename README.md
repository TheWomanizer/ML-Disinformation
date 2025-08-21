# Sistema Integral Multi-Capa para la Detección de Desinformación

Este repositorio contiene el código fuente y los recursos para un proyecto de Machine Learning enfocado en la detección de noticias falsas. El sistema va más allá de una simple clasificación binaria, implementando un enfoque holístico que busca responder a tres preguntas clave:

1.  **¿QUÉ?** -> ¿Es esta noticia falsa o verdadera? (Clasificación de Contenido)
2.  **¿QUIÉN?** -> ¿La fuente que publica la noticia es creíble? ¿Es un bot? (Análisis de la Fuente)
3.  **¿POR QUÉ?** -> ¿Qué factores llevaron al sistema a su conclusión? (Explicabilidad)

El objetivo es desarrollar un prototipo de sistema robusto, interpretable y multi-capa, utilizando una combinación de modelos de Machine Learning tradicionales y técnicas avanzadas de Procesamiento de Lenguaje Natural (NLP).

## Metodología

El proyecto se desarrolla a través de varias fases, desde la preparación de datos hasta la evaluación de modelos complejos:

1.  **Análisis Exploratorio y Limpieza de Datos (Fase 0):** Se realiza una limpieza exhaustiva de los datos, incluyendo la validación semántica de ceros y nulos, análisis de desbalance de clases, y la separación de datos para modelado tradicional y NLP. Un desafío clave identificado en esta fase es el manejo de **outliers extremos** en variables numéricas, para lo cual se implementan técnicas como el _Winsorizing_.

2.  **Modelado de Clasificación de Contenido (Fase 1):** Se construyen y evalúan dos tipos de modelos principales:

    - **Modelo con Features Tradicionales:** Utiliza un clasificador `XGBoost` sobre 58 características numéricas pre-calculadas que describen el texto y el comportamiento del autor.
    - **Modelo con NLP:** Emplea un modelo Transformer (BERT), específicamente `dccuchile/bert-base-spanish-wwm-uncased`, para realizar una clasificación basada en la semántica y el contexto del texto de las noticias y tweets.

3.  **Análisis de la Fuente y Explicabilidad (Fases Futuras):** Las fases posteriores del proyecto se centran en evaluar la credibilidad de la fuente y en implementar técnicas de XAI (Explainable AI) como SHAP o LIME para interpretar las predicciones de los modelos.

## Estructura del Repositorio

El proyecto está organizado de la siguiente manera:

```
.
├── config/
│   └── config.yaml
├── dataset1/
│   ├── Features_For_Traditional_ML_Techniques.csv
│   ├── Truth_Seeker_Model_Dataset.csv
│   └── readme.txt
├── dataset2/
│   └── social media usage dataset.csv
├── models/
│   ├── ... (modelos guardados y resultados)
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb
│   ├── 02_limpieza_datos.ipynb
│   ├── ... (notebooks para cada fase del modelado)
├── processed_data/
│   ├── dataset_features_processed.csv
│   └── ... (datos limpios y listos para modelar)
├── ANTEPROYECTO.pdf
├── PROYECTO_GUIA.md
└── README.md
```

- **config/**: Archivos de configuración.
- **dataset1/**: Dataset principal para la detección de desinformación.
- **dataset2/**: Dataset secundario para análisis de uso de redes sociales.
- **models/**: Almacena los modelos entrenados, comparaciones y resultados.
- **notebooks/**: Contiene los Jupyter Notebooks con el código de cada fase del proyecto, desde la exploración hasta el modelado.
- **processed_data/**: Guarda los datasets limpios y transformados, listos para ser consumidos por los modelos.

## Uso

Para replicar el proyecto, siga los pasos delineados en los Jupyter Notebooks en orden numérico, comenzando por `01_exploracion_inicial.ipynb`.

1.  **Configuración del Entorno:** Se recomienda crear un entorno virtual de Python y instalar las dependencias listadas en `notebooks/requirements.txt`.
2.  **Ejecución de Notebooks:** Ejecute los notebooks en secuencia para realizar la limpieza de datos, el entrenamiento de modelos y la evaluación.

## Autores y Contribuciones

- **Autor Principal:** Jose Alejandro Jimenez Vasquez : jajimenez4@eafit.edu.co

---

Este `README.md` fue generado como parte del proceso de desarrollo.
