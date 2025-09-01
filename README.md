# Sistema Integral para la Detección de Desinformación
### :O ._. La historia de cómo un mega-ensamble de 16 modelos compitió contra RoBERTa y vivió para contarlo.

Este repositorio es el recuento del viaje que fue mi proyecto final del Diplomado de Machine Learning. Lo que empezó como una misión para detectar noticias falsas se convirtió en una odisea a través de docenas de modelos, una batalla campal contra el *data leakage* y, finalmente, una valiosa lección sobre la eficiencia de los Transformers.

## La Trama Principal: 'The Real Slim Ensemble' vs. RoBERTa

Se construyó "The Real Slim Ensemble", una ambiciosa arquitectura de 16 modelos apilados con la meta de ser la solución definitiva a la desinformación. ¿El resultado? Un sólido F1-Score de **0.8870**.

Nada mal, ¿no? El giro de la trama es que un único modelo, **RoBERTa**, sin todo el andamiaje, alcanzó por su cuenta un **0.8889**.

La conclusión fue fascinante: después de un esfuerzo considerable, el complejo ensamble fue como una orquesta sinfónica compitiendo contra un virtuoso solista. Una gran lección y una historia que vale la pena compartir.

## Lecciones Aprendidas en la Trinchera (Hallazgos Clave)

- **RoBERTa es un Titán:** Para esta tarea, los modelos Transformers demostraron estar en una categoría propia. El rendimiento que ofrecen de base es formidable.
- **Más No Siempre es Mejor:** Este proyecto es un caso de estudio. Apilar modelos más débiles sobre uno muy fuerte puede no generar una mejora neta si introducen más ruido que señal.
- **El Preprocesamiento es Ley:** Sin una corrección rigurosa del *data leakage* (pasando de 134k a 1k de filas para NLP), las métricas no son confiables. Un paso doloroso pero indispensable para obtener resultados realistas.
- **Winsorizing al Rescate:** Ante datos con valores extremos, la técnica de *Winsorizing* fue una herramienta clave para estabilizar los modelos y reducir la influencia de outliers.

## El Mapa de la Aventura (Estructura del Repositorio)

- **[`PROYECTO.md`](PROYECTO.md)**: La bitácora completa del viaje. Contiene cada detalle del proceso y los análisis. **(Haz clic aquí para leer la historia completa)**
- **`notebooks/`**: Las etapas de la expedición. Nueve notebooks, numerados del `01` al `09`. Se recomienda seguirlos en orden para entender la historia.
- **`models/`**, **`processed_data/`**, etc.: El equipo y los artefactos recuperados. Modelos, datos limpios y otros recursos.

## ¿Quieres Replicar la Expedición?

1.  Clona este repositorio.
2.  Crea un entorno virtual (es una buena práctica).
3.  Instala las dependencias desde la carpeta `notebooks`:
    ```bash
    pip install -r notebooks/requirements.txt
    ```
4.  Ejecuta los notebooks en orden numérico, empezando por el `01`.

## Una Pequeña Invitación

Si te gustan los proyectos que muestran el proceso real, con sus éxitos y sus reveladores callejones sin salida, considera dar un `follow`.

**Sígueme en GitHub: [TheWomanizer](https://github.com/TheWomanizer)**

Cada `follow` y cada estrella son un gran motivador para seguir construyendo y compartiendo proyectos con este enfoque honesto.

---
*Proyecto desarrollado por José Alejandro Jiménez Vásquez.*
