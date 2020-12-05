# Bibliografia
# http://archive.ics.uci.edu/ml/datasets/Wine+Quality

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics, svm, neural_network, naive_bayes, tree, linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Carrega os dados em uma matriz
dados = pd.read_csv('datasets/winequality-white.csv', sep=';')
print(dados.columns)

df_x = dados[dados.columns[:-1]]
df_y = dados['quality']

X = np.array(df_x)
Y = np.array(df_y)

qtde_tuplas = X.shape[0]

# Criar um objeto classificador
# classificador = svm.SVC(gamma=0.001)
# classificador = neural_network.MLPClassifier(hidden_layer_sizes=(100, 200, 50), max_iter=500)
# classificador = naive_bayes.ComplementNB()
# classificador = tree.DecisionTreeClassifier()
classificador = linear_model.LinearRegression()
# classificador = linear_model.LogisticRegression()
# classificador = linear_model.Ridge(alpha=.5)
regressao = True

# Dividir os dados em dados para treino e dados para teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=1)

# Realiza o treinamento com o conjunto de dados de treino
classificador.fit(X_treino, y_treino)

# Utilizamos os registros
predicao = classificador.predict(X_teste)

if regressao:
    s = 0
    for t, p in zip(y_teste, predicao):
        print(f't={t} | p={p} | e = {(t - p) / t}')
        s += abs((t - p) / t)
    s /= len(y_teste)
    print(f'Erro relativo médio:{s}')
else:
    f1u = metrics.f1_score(y_teste, predicao, average='micro')
    f1m = metrics.f1_score(y_teste, predicao, average='macro')
    f1 = metrics.f1_score(y_teste, predicao, average=None)
    print(f'f1u = {f1u}')
    print(f'f1m = {f1m}')
    print(f'f1m = {f1}')
    # Mostrando a matriz de confusão
    disp = metrics.plot_confusion_matrix(classificador, X, Y)
    disp.figure_.suptitle("Matriz de Confusão")
    # print(f"Matriz de confusão:\n{disp.confusion_matrix}")
    plt.show()
