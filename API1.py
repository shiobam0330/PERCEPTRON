import streamlit as st
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

# Título de la aplicación
st.title("API Perceptron multiclase")


if 'grafico1' not in st.session_state:
    st.session_state.grafico1 = None  
 

with st.container():
    # Definir las dos columnas
    col1, col2 = st.columns(2)

    # ============================================
    # COLUMNA IZQUIERDA: Conjunto de Datos Simulado
    # ============================================
    with col1:
        st.header("Conjunto de datos simulado con 2 variables")
        Simulacion = st.number_input("Ingrese el numero de simulaciones que desea realizar",  min_value=1, value=1000)
        Clase = st.number_input("Ingrese el numero de clases con las que desea trabajar",  min_value=1, value=3)

        if st.button("Generar"):
            X, y = make_blobs(n_samples=Simulacion, centers=Clase, n_features=2, random_state=42)
            st.session_state.simulated_data = (X, y)

            fig, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', marker='o')
            st.session_state.grafico1 = fig  
        
        if st.session_state.grafico1 is not None:
            st.pyplot(st.session_state.grafico1)

    # ============================================
    # COLUMNA DERECHA: Perceptron
    # ============================================

    with col2:
        st.header("Limites de decision Perceptron")
        Iteraciones = st.number_input("Ingrese el numero de iteraciones que desea usar",  min_value=1, value=100)
        NumClas = st.number_input("Ingrese el numero de clases que desea usar",  min_value=1, value=3)
        Tasa = st.number_input("Ingrese la tasa de aprendizaje con la que desea trabajar", min_value=0.0, value=0.01, step=0.01)

        if st.button("Estimar Red Neuronal"):
            
            X, y = st.session_state.simulated_data
            
            class PerceptronMulticlass:
                if st.session_state.simulated_data is not None:
                    X, y = st.session_state.simulated_data
                def __init__(self,lr, itera, n_cat):
                    self.pesos = None
                    self.sesgo = None
                    self.tasa = lr
                    self.n_iter = itera
                    self.n_cat = n_cat
                    self.fun_act = self.valor_y
        
                def valor_y(self, x):
                    return np.where(x >= 0, 1, 0)
    
                def indicadora(self, x):
                    return 1 if x == True else 0
    
                def fit(self, X, y):
                    n_col = X.shape[1]
                    self.pesos = np.zeros((self.n_cat, n_col))
                    self.sesgo = np.zeros(self.n_cat)
                    self.categ = np.unique(y)
                    y_1 = pd.get_dummies(pd.array(y), prefix = "Clase").map(self.indicadora)
        
                    for _ in range(self.n_iter):
                        for i, x_i in enumerate(X):
                            for k in range(self.n_cat):
                                resultado = np.dot(x_i, self.pesos[k]) + self.sesgo[k]
                                y_est = self.fun_act(resultado)
                    
                                update = self.tasa * (y_1.iloc[:,k][i] - y_est)
                                self.pesos[k] += update * x_i
                                self.sesgo[k] += update
    
                
        
                def predict_ind(self,X):
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    y_pred = np.zeros(X.shape[0])
                    for i, x_i in enumerate(X):
                        resultados = np.zeros(self.n_cat)
                        for k in range(self.n_cat):
                            resultados[k] = np.dot(x_i, self.pesos[k]) + self.sesgo[k]
                        y_pred[i] = self.categ[np.argmax(resultados)]
                    return y_pred
        
                def plot_decision_boundary(self, X, y):
                    fig, ax = plt.subplots()
                    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', marker='o')
                    x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
                    y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
                    Z = np.array([self.predict_ind(np.array([xi, yi])) for xi, yi in zip(xx.ravel(), yy.ravel())])
                    Z = Z.reshape(xx.shape)
                    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
                    return fig
            
            perceptron = PerceptronMulticlass(lr = Tasa, itera = Iteraciones, n_cat = NumClas)
            perceptron.fit(X,y)
            
            fig = perceptron.plot_decision_boundary(X, y)
            st.session_state.grafico2 = fig
            st.pyplot(fig)

