#importar a bibliotecas necessárias
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#carregar os dados de transações financeiras
data = pd.read_csv("creditcard.csv")

#separar os dados em variáveis independentes (x) e dependentes (y)
x = data.drop("class", axis=1)
y= data["Class"]

#Normalizar os dados usando a técnica min-max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

#dividir os dados em treino e teste
from sklearn.model_selection import train_test_split
x_train, x_text, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Definir o modelo de detecção de fraudes usando uma rede neural autoencoder
input_dim = x_train.shape[1]
encoding_dim = 14

input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
decoder = tf.keras.layers.Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = tf.keras.models.Model(imputs=input_layer, outputs=decoder)

#compilar e treinar o modelo
autoencoder.compile(optimizer="adam", loss="mean_squared_error")
autoencoder.fit(x_train, x_train, epochs = 10, batch_size = 32)

#obter as reconstruções do modelo para os dados de teste
reconstructions = autoencoder.predict(x_text)

# Calcular o erro da reconstrução para cada transação
mse = np.mean(np.power(x_test - reconstructions, 2), axis=1)

#criar um dataframe com o erro e a classe real de cada transação
error_df = pd.DataFrame({"reconstruction_error":mse, "true_class": y_test})

#Definir um limiar para classificar uma transação como fraude ou normal
threshold = 0.01

#Classificar as transações com base no limiar
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

# Calcular as métricas de avaliação do modelo 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(error_df.true_class, y_pred)
precision = precision_score(error_df, true_class, y_pred)
recall = recall_score(error_df.true_class, y_pred)
f1 = f1_score(error_df.true_class, y_pred)

#imprimir as métricas
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:{recall:.4f}")
print(f"F1-score:{f1:.4f}")
