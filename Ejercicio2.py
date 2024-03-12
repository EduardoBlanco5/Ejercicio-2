import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Función para cargar los datos y realizar la clasificación
def realizar_clasificacion(archivo_csv, num_partitions, train_percentage):
    # Cargar los datos del archivo CSV
    data = pd.read_csv(archivo_csv)
    
    # Dividir los datos en características (X) y etiquetas (y)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Iterar sobre las particiones
    for i in range(num_partitions):
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_percentage, random_state=i)
        
        # Inicializar y entrenar el perceptrón
        perceptron = Perceptron()
        perceptron.fit(X_train, y_train)
        
        # Predecir las etiquetas de prueba
        y_pred = perceptron.predict(X_test)
        
        # Calcular la precisión de la clasificación
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Partition {i+1}: Accuracy = {accuracy}')

# Para el primer conjunto de datos (spheres1d10.csv)
print("Clasificación para spheres1d10.csv:")
realizar_clasificacion('spheres1d10.csv', num_partitions=5, train_percentage=0.8)

# Para el segundo conjunto de datos (spheres2d10.csv)
print("\nClasificación para spheres2d10.csv:")
realizar_clasificacion('spheres2d10.csv', num_partitions=10, train_percentage=0.8)

# Para el tercer conjunto de datos (spheres2d50.csv)
print("\nClasificación para spheres2d50.csv:")
realizar_clasificacion('spheres2d50.csv', num_partitions=10, train_percentage=0.8)

# Para el cuarto conjunto de datos (spheres2d70.csv)
print("\nClasificación para spheres2d70.csv:")
realizar_clasificacion('spheres2d70.csv', num_partitions=10, train_percentage=0.8)