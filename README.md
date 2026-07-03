# Fragmentación urbana de Barcelona mediante análisis de redes y Graph Neural Networks

Este repositorio recoge el código en Python desarrollado para la elaboración de un Trabajo Final de Máster centrado en el estudio de la fragmentación urbana de Barcelona.

La metodología propuesta modela la ciudad como una red (grafo), donde los nodos representan secciones censales y las aristas describen relaciones entre ellas. Sobre esta representación se aplican dos enfoques complementarios:

## Metodología

### 1. Métodos clásicos de análisis y clustering

- **K-means** sobre los datos tabulares derivados de la matriz de pesos.
- **Detección de comunidades** mediante el algoritmo **Leiden** aplicado sobre el grafo ponderado.

### 2. Aprendizaje profundo sobre grafos

Se emplean **Graph Neural Networks (GNN)** para obtener representaciones (*embeddings*) de las secciones censales y realizar posteriormente tareas de clustering.

Se analizan tres escenarios:

1. Utilizando toda la información disponible del grafo (atributos de nodos y aristas).
2. Utilizando únicamente los atributos de los nodos y la estructura de adyacencia.
3. Utilizando únicamente los atributos de las aristas, manteniendo la estructura de adyacencia.

## Contenido del repositorio

- **`creacion_del_grafo.ipynb`**: cuaderno que contiene todo el proceso de construcción del grafo.
- **`creacion_grafo(directo).py`**: versión en script del proceso de construcción del grafo.
- **`funciones2.py`**: script con funciones auxiliares para la construcción de los atributos de las aristas.
- **`secciones_con_datos.graphml`**: grafo final con atributos sociodemográficos asociados tanto a nodos como a aristas.
- **`clustering_GNN.ipynb`**: cuaderno donde se entrenan los modelos GNN y se realiza el clustering a partir de los *embeddings* obtenidos en los tres escenarios analizados.
- **`clustering_clasico.ipynb`**: cuaderno donde se aplica K-means sobre los datos tabulares y detección de comunidades mediante el algoritmo Leiden.

## Datos utilizados

Por motivos de licencia, tamaño y reproducibilidad, los datos originales no se incluyen en este repositorio. No obstante, las variables empleadas, sus fuentes y su justificación analítica se resumen a continuación.

| Variable | Fuente | Dimensión analítica | Interpretación |
|-----------|----------|---------------------|----------------|
| Número de locales por uso (vivienda, educación-cultura, oficinas y comercio) | Catastro | Usos del suelo | Caracteriza la estructura funcional de la sección censal y la mezcla de usos |
| Proporción de superficie construida dedicada a vivienda | Catastro | Usos del suelo | Indica el grado de especialización residencial del tejido urbano |
| Edad media y superficie de la vivienda | Catastro | Parque residencial y mercado de la vivienda | Describe la calidad, antigüedad y tipología de la vivienda |
| Precio medio del alquiler | Índice del precio de la vivienda | Parque residencial y mercado de la vivienda | Refleja el nivel de accesibilidad económica al área |
| Rango intercuartílico del alquiler | Índice del precio de la vivienda | Parque residencial y mercado de la vivienda | Captura la heterogeneidad interna del mercado residencial |
| Índice de Gini | Atlas de distribución de la renta | Desigualdad | Mide la distribución interna de la renta |
| Renta media por habitante | Atlas de distribución de la renta | Estructura socioeconómica | Aproxima el nivel socioeconómico medio |
| Proporción de población con renta inferior al 40 % y superior al 160 % de la renta media | Atlas de distribución de la renta | Estructura socioeconómica | Permite identificar situaciones de vulnerabilidad y de renta elevada |
| Tasa de paro | Censo | Estructura socioeconómica | Indicador de exclusión del mercado laboral |
| Nivel educativo (básico y superior) | Censo | Composición sociodemográfica | Aproxima la estructura de oportunidades |
| Proporción de población nacida en el extranjero | Censo | Composición sociodemográfica | Puede reflejar procesos de segregación o atracción migratoria |
| Proporción de población nacida en Barcelona | Censo | Composición sociodemográfica | Aproxima la estabilidad poblacional |

## Objetivo

El objetivo de este trabajo es evaluar hasta qué punto los métodos clásicos de análisis de redes y clustering, así como los enfoques basados en Graph Neural Networks, permiten identificar patrones de fragmentación urbana y estructuras socioespaciales en la ciudad de Barcelona.
