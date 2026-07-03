En esta repositorio se recoje el codigo de python utilizado para la elavoración de un Trabajo Final de Máster que estudia la fragmentación urbana de Barcelona. Para hacerlo se propone modelar la ciudad como red para hacer aplicar metodologiac classicas: k-means con los datos tabulares d ela matiz de peosos y deteccion de cominidades con el algoritmo Leiden del grafo ponderado. Y propone utilizar aprendizaje profundo aplicando GNN (Graph Neural Networks) en tres escenarios: con toda la informacion del grafo; solo con los atibutos de los nodos y la adyacencia; y solo con los atibutos de las aristas (conservando la adyacencia).
Y con este repositorio se comparte:
- "creacion_del_grafo.ipynb": La libreta con todo el proceso de constucción del grafo, "creacion_grafo(directo).py" es el equivalente pero en script.
- "funciones2.py": scrip con funcione fundamentales para constuir los atibutos de las aristas
- "clustering GNN.ipynb": librata donde se entrena el modelo GNN y con los embeddings se hace el clustering de los tres escenarios mencionados
- "clustering clasico.ipynb": libreta donde se hace el clustering k-means son los datos de los nodos y deteccion de comunidades con el algoritmo de Leiden.
- "analisis numerico
