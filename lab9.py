import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics.pairwise import euclidean_distances

# Parte I: Implementación del SMOTE y los clasificadores

# Cargar el conjunto de datos Glass
file_path = r"C:\Users\crist\Downloads\glass+identification\glass.csv"
data = pd.read_csv(file_path)

# Verificación inicial del dataset
print(f"Dataset Glass cargado. Tamaño: {data.shape}")
print(data.head())

# Preparar los datos (se asume que la última columna es la etiqueta)
if 'Class' not in data.columns:
    raise ValueError("La columna 'Class' no se encuentra en el dataset.")

X = data.drop('Class', axis=1).values
y = data['Class'].values

# Función para implementar SMOTE
def smote(X, y, N=100, k=5):
    minority_class = np.min(np.unique(y))
    minority_indices = np.where(y == minority_class)[0]
    
    # Extraemos las muestras de la clase minoritaria
    X_minority = X[minority_indices]
    if len(X_minority) == 0:
        raise ValueError("No hay muestras de la clase minoritaria en el dataset.")
    
    # Generamos instancias sintéticas
    synthetic_samples = []
    for i in range(len(X_minority)):
        distances = euclidean_distances(X_minority[i].reshape(1, -1), X_minority)
        neighbors = np.argsort(distances[0])[1:k+1]
        
        for _ in range(N // len(X_minority)):
            neighbor = X_minority[neighbors[np.random.randint(0, k)]]
            diff = neighbor - X_minority[i]
            synthetic_sample = X_minority[i] + np.random.rand() * diff
            synthetic_samples.append(synthetic_sample)
    
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.array([minority_class] * len(X_synthetic))
    
    X_new = np.vstack([X, X_synthetic])
    y_new = np.hstack([y, y_synthetic])
    
    return X_new, y_new

# Función de clasificador Euclidiano
def euclidean_classifier(X_train, y_train, X_test):
    predictions = []
    for test_point in X_test:
        distances = euclidean_distances(X_train, test_point.reshape(1, -1))
        nearest_neighbor_index = np.argmin(distances)
        predictions.append(y_train[nearest_neighbor_index])
    return np.array(predictions)

# Función de clasificador 1-NN
def knn_classifier(X_train, y_train, X_test, k=1):
    predictions = []
    for test_point in X_test:
        distances = euclidean_distances(X_train, test_point.reshape(1, -1))
        nearest_neighbors = np.argsort(distances.flatten())[:k]
        predicted_class = np.argmax(np.bincount(y_train[nearest_neighbors].astype(int)))
        predictions.append(predicted_class)
    return np.array(predictions)

# Hold-Out (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Dataset dividido. Tamaño de entrenamiento: {X_train.shape}, Tamaño de prueba: {X_test.shape}")

# Clasificador Euclidiano antes de SMOTE
y_pred = euclidean_classifier(X_train, y_train, X_test)
accuracy_euclidean_before = accuracy_score(y_test, y_pred)
print(f"Precisión Euclidiana antes de SMOTE: {accuracy_euclidean_before}")

# Clasificador 1-NN antes de SMOTE
y_pred_knn = knn_classifier(X_train, y_train, X_test)
accuracy_knn_before = accuracy_score(y_test, y_pred_knn)
print(f"Precisión 1-NN antes de SMOTE: {accuracy_knn_before}")

# Aplicamos SMOTE
X_train_smote, y_train_smote = smote(X_train, y_train, N=100, k=5)

# Clasificador Euclidiano después de SMOTE
y_pred_smote_euclidean = euclidean_classifier(X_train_smote, y_train_smote, X_test)
accuracy_euclidean_after = accuracy_score(y_test, y_pred_smote_euclidean)
print(f"Precisión Euclidiana después de SMOTE: {accuracy_euclidean_after}")

# Clasificador 1-NN después de SMOTE
y_pred_smote_knn = knn_classifier(X_train_smote, y_train_smote, X_test)
accuracy_knn_after = accuracy_score(y_test, y_pred_smote_knn)
print(f"Precisión 1-NN después de SMOTE: {accuracy_knn_after}")

# 10-Fold Cross-Validation
kf = KFold(n_splits=10, random_state=42, shuffle=True)
cross_val_accuracy_euclidean = []
cross_val_accuracy_knn = []

for train_idx, test_idx in kf.split(X):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    # Clasificador Euclidiano
    y_pred_fold = euclidean_classifier(X_train_fold, y_train_fold, X_test_fold)
    cross_val_accuracy_euclidean.append(accuracy_score(y_test_fold, y_pred_fold))

    # Clasificador 1-NN
    y_pred_knn_fold = knn_classifier(X_train_fold, y_train_fold, X_test_fold)
    cross_val_accuracy_knn.append(accuracy_score(y_test_fold, y_pred_knn_fold))

print(f"Precisión promedio Euclidiana (10-Fold CV) antes de SMOTE: {np.mean(cross_val_accuracy_euclidean)}")
print(f"Precisión promedio 1-NN (10-Fold CV) antes de SMOTE: {np.mean(cross_val_accuracy_knn)}")

# Parte II: Implementación del Perceptrón Simple

# Parte II: Implementación del Perceptrón Simple

# Cargar el dataset Iris (Setosa y Virginica)
iris_path = r"C:\Users\crist\Downloads\iris (1)\bezdekIris.csv"
iris_data = pd.read_csv(iris_path)

# Verificación inicial del dataset
print(f"Dataset Iris cargado. Tamaño: {iris_data.shape}")
print(iris_data.head())

# Filtramos para tomar solo Setosa y Virginica
# Cambiar el filtro para las clases correctas, que son 'Iris-setosa' y 'Iris-virginica'
iris_data = iris_data[iris_data['Class'].isin(['Iris-setosa', 'Iris-virginica'])]
print(f"Dataset Iris después del filtrado. Tamaño: {iris_data.shape}")

# Verificar si hay datos después del filtrado
if iris_data.empty:
    raise ValueError("El dataset Iris filtrado está vacío. Verifica los valores en la columna 'Class'.")

# Preparar los datos
X_iris = iris_data.drop('Class', axis=1).values
y_iris = iris_data['Class'].values

# Convertir a 0 y 1 (Setosa = 0, Virginica = 1)
y_iris = np.where(y_iris == 'Iris-setosa', 0, 1)

# Dividir el dataset en Hold-Out (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

# Perceptrón simple
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)  # Incluye el sesgo

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        X_with_bias = np.c_[np.ones(X.shape[0]), X]  # Agregar columna de 1 para sesgo
        return np.array([self.activation(np.dot(x, self.weights)) for x in X_with_bias])

    def fit(self, X, y):
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        for _ in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i].reshape(1, -1))[0]
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X_with_bias[i]

# Crear y entrenar el perceptrón
perceptron = Perceptron(input_size=X_train.shape[1])
perceptron.fit(X_train, y_train)

# Predicción y validación
y_pred_perceptron = perceptron.predict(X_test)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
print(f"Precisión del perceptrón: {accuracy_perceptron}")
