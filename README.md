# Sistema Integral para la Detección de Desinformación

### La historia de cómo un mega-ensamble de 16 modelos compitió contra RoBERTa y vivió para contarlo.

Este repositorio contiene el proyecto final del Diplomado de Machine Learning de la Universidad EAFIT. El trabajo documenta el proceso de construcción de un sistema de detección de desinformación, desde el preprocesamiento de datos hasta la evaluación comparativa de arquitecturas de Machine Learning de distinta complejidad.

---

## Resumen y Hallazgo Principal

Se desarrolló un sistema de meta-aprendizaje, **"The Real Slim Ensemble"**, que integra 16 modelos predictivos (incluyendo árboles de decisión, redes neuronales y transformers) con el objetivo de maximizar la precisión en la clasificación de noticias.

El resultado más significativo del proyecto no fue el rendimiento del ensamble (F1-Score: 0.8870), sino la comparación con su componente individual más fuerte. Un único modelo **RoBERTa**, sin el andamiaje del ensamble, obtuvo un rendimiento superior (F1-Score: 0.8889).

Esta conclusión subraya una lección fundamental: en problemas donde un modelo experto (como un Transformer pre-entrenado) es excepcionalmente bueno, la complejidad adicional de un ensamble puede no traducirse en una mejora de rendimiento.

## Conclusiones Técnicas Clave

-   **Dominio de los Transformers:** RoBERTa se consolidó como la arquitectura de vanguardia para esta tarea de clasificación de texto, estableciendo un baseline de rendimiento muy difícil de superar.
-   **Límites del Ensamblaje:** Este proyecto sirve como caso de estudio sobre los rendimientos decrecientes. Apilar modelos de menor rendimiento sobre un modelo SOTA (*State-Of-The-Art*) puede introducir ruido y degradar la predicción final.
-   **Impacto Crítico del Preprocesamiento:** La corrección del *data leakage* fue el paso más importante para una evaluación honesta de los modelos NLP. Al haber múltiples tweets para una misma noticia, fue necesario reducir el dataset de 134,000 a 1,058 registros únicos para evitar que el modelo "memorizara" las respuestas.
-   **Estabilización con Winsorizing:** La técnica de *Winsorizing* fue crucial para acotar los valores atípicos en las características numéricas, lo que mejoró la estabilidad y el rendimiento de los modelos tradicionales.

## Estructura del Repositorio

-   **[`PROYECTO.md`](PROYECTO.md)**: El informe final y la bitácora completa del proyecto. Contiene un análisis detallado de la metodología y los resultados. **(Recomendado para una inmersión profunda)**.
-   **`/notebooks`**: Contiene los 9 Jupyter Notebooks que documentan cada paso del proyecto, desde la exploración (`01_...`) hasta el ensamble final (`09_...`).
-   **`/models`**: Almacena los modelos serializados (`.pkl`, `.h5`), así como los resultados de los experimentos y las comparaciones de rendimiento.
-   **`/processed_data`**: Contiene los datasets intermedios generados tras la limpieza y el preprocesamiento.

## Guía de Instalación y Reproducción

Para replicar este proyecto, se recomienda seguir los siguientes pasos en un entorno limpio.

**1. Clonar el Repositorio**
```bash
git clone https://github.com/TheWomanizer/ML-Disinformation.git
cd ML-Disinformation
```

**2. Crear un Entorno Virtual**

Es una buena práctica para aislar las dependencias del proyecto. Se recomienda usar Python 3.9 o superior.
```bash
python -m venv .venv
source .venv/bin/activate  # En Linux/macOS
# .venv\Scripts\activate   # En Windows
```

**3. Instalar Dependencias**

Las librerías requeridas se encuentran en la carpeta `notebooks`.
```bash
pip install -r notebooks/requirements.txt
```

**4. Ejecutar los Notebooks**

Para entender el flujo de trabajo completo, ejecute los notebooks en orden secuencial, comenzando por `01_exploracion_inicial.ipynb`.

## Contacto

-   **Autor:** José Alejandro Jiménez Vásquez
-   **GitHub:** [TheWomanizer](https://github.com/TheWomanizer) (¡un `follow` siempre es un gran motivador!)

---