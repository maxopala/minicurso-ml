# Bibliografia
# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
# https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

import matplotlib.pyplot as plt

from sklearn import metrics, svm, neural_network, naive_bayes, tree
from sklearn.model_selection import train_test_split
import numpy as np

# Carrega os dados em uma matriz
dados = np.loadtxt('datasets/digits.csv', delimiter=',', dtype=str)

# No dataset tem 65 colunas.
# As primeiras 64 se referem aos 64 segmentos das imagens 8x8
pixels = dados[:, :-1].astype(int)
# A última coluna se refere ao número ap qual a imagem se refere
numeros = dados[:, 64]

qtde_tuplas = len(pixels)

# Criar um objeto classificador
# classificador = svm.SVC(gamma=0.001)
classificador = neural_network.MLPClassifier(hidden_layer_sizes=(100, 200, 50), max_iter=500)
# classificador = naive_bayes.ComplementNB()
# classificador = tree.DecisionTreeClassifier()

# Dividir os dados em dados para treino e dados para teste
X_treino, X_teste, y_treino, y_teste = train_test_split(pixels, numeros, test_size=0.3, shuffle=True, random_state=1)

# Realiza o treinamento com o conjunto de dados de treino
classificador.fit(X_treino, y_treino)

# Utilizamos os registros
predicao = classificador.predict(X_teste)

f1u = metrics.f1_score(y_teste, predicao, average='micro')
f1m = metrics.f1_score(y_teste, predicao, average='macro')
f1 = metrics.f1_score(y_teste, predicao, average=None)

print(f'f1u = {f1u}')
print(f'f1m = {f1m}')
print(f'f1 = {f1}')

# Mostrando a matriz de confusão
disp = metrics.plot_confusion_matrix(classificador, pixels, numeros)
disp.figure_.suptitle("Matriz de Confusão")
# print(f"Matriz de confusão:\n{disp.confusion_matrix}")
plt.show()

teste = classificador.predict([
    [0, 7, 14, 12, 12, 1, 0, 0, 0, 4, 8, 0, 0, 0, 0, 0, 0, 4, 10, 4, 2, 0, 0, 0, 0, 2, 11, 8, 12, 9, 0, 0, 0, 0, 0, 0,
     0, 12, 4, 0, 0, 0, 0, 0, 0, 5, 10, 0, 0, 2, 7, 1, 0, 10, 6, 0, 0, 0, 6, 13, 12, 11, 0, 0]
])

print(f"5 do Max = {teste[0]}")