# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo de clasificación
"""

# pip install pandas
# pip install scikit-learn
# pip install nltk
# pip install imbalanced-learn

import random
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore

# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, CategoricalNB


# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

""" ########################################
    FUNCIONES AUXILIARES
######################################## """

def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest,naive_bayes)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    args = parse.parse_args() # Parseamos los argumentos
    with open('clasificador.json') as json_file:  # Leemos los parametros del JSON
        config = json.load(json_file)
    for key, value in config.items(): # Juntamos todo en una variable
        setattr(args, key, value)
    # Parseamos los argumentos
    return args

def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        # Convertir el archivo a UTF-8
        data = pd.read_csv(file, encoding='utf-8', sep=",")
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        #data = concatenarColumnasCSV(data,["col1","col2"], "nombreColNueva")
        if args.preprocessing.get("muestra_pequeña"):
            sample_size = 1000
            data= data.sample(n=sample_size, random_state=42).reset_index(drop=True) # Muestra aleatoria de 200 filas
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

def concatenarColumnasCSV(data, concatenarColumnas, NombreColNueva):
    # Verificar que las columnas existen en el DataFrame
    print("COLUMNAS" , data.columns)
    missing_cols = [col for col in concatenarColumnas if col not in data.columns]
    print("MISSIN", missing_cols)
    if missing_cols:
        print(Fore.RED + f"Alguna columna introducida para concatenar no existe en el CSV: {missing_cols}" + Fore.RESET)
        sys.exit(1)
    # Reemplazar los NaN por cadenas vacías en las columnas seleccionadas
    data[concatenarColumnas] = data[concatenarColumnas].fillna('')
    # Concatenar las columnas especificadas
    data[NombreColNueva] = data[concatenarColumnas].astype(str).agg('##'.join, axis=1)
    print(Fore.GREEN + f"Columnas {concatenarColumnas} concatenadas en '{NombreColNueva}'" + Fore.RESET)
     # Eliminar cualquier espacio extra que quede (si hay columnas con solo espacios)
    data[NombreColNueva] = data[NombreColNueva].apply(lambda x: x.strip() if isinstance(x, str) else x)
    # Eliminar las columnas originales
    data.drop(columns=concatenarColumnas, inplace=True)
    print(Fore.GREEN + f"Columnas originales {concatenarColumnas} concatenadas han sido eliminadas" + Fore.RESET)
    
    return data 


""" ########################################
    FUNCIONES PARA PREPROCESAR LOS DATOS
######################################## """

def preprocesar_datos():
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Pasar los datos de categoriales a numéricos 
        3. Tratamos missing values (Eliminar y imputar)
        4. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Tratamos el texto (TF-IDF, BOW)
        7. Realizamos Oversampling o Undersampling
        8. Borrar columnas no necesarias
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """
    # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()
    # Simplificamos el texto
    simplify_text(text_feature)
    # Pasar los datos a categoriales a numéricos
    cat2num(categorical_feature)
    # Tratamos missing values
    data = process_missing_values(numerical_feature, categorical_feature, text_feature)
    # Volvemos a separar las características después de procesar los valores faltantes
    numerical_feature, text_feature, categorical_feature = select_features()
    # Reescalamos los datos numéricos
    reescaler(numerical_feature)
    dividirCateg()
    # Tratamos el texto
    process_text(text_feature)
    # Realizamos Oversampling o Undersampling
    over_under_sampling()
    drop_features()
    return data

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64']) # Columnas numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]
        
        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN+"Datos separados con éxito"+Fore.RESET)
        
        if args.debug:
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

# def dividirCateg():
#     global data 
#     if args.mode == "train" and args.preprocessing["dividir_categorias"] is True:
#         if args.prediction in data.columns:
#             data[args.prediction]=data[args.prediction].apply(lambda x: "negative" if x <= 2 else ("neutral" if x == 3 else "positive")) # Cambiamos la columna a predecir a 0 y 1
#         else:
#            sys.exit(1)

import sys

def dividirCateg():
    global data
    if args.mode == "train" and args.preprocessing.get("dividir_categorias") is True:
        if args.prediction in data.columns:
            unique_values = sorted(data[args.prediction].unique())
            n = len(unique_values)
            
            if n == 0:
                sys.exit("No hay valores únicos en la columna.")
            
            mid = n // 2
            
            if n % 2 == 0:
                # Si el número de categorías es par, tomar dos valores como neutros
                neutral_values = set([unique_values[mid - 1], unique_values[mid]])
            else:
                # Si es impar, tomar un solo valor como neutro
                neutral_values = set([unique_values[mid]])
            
            def categorizar(x):
                if x in neutral_values:
                    return "neutral"
                elif x < min(neutral_values):
                    return "negative"
                else:
                    return "positive"

            data[args.prediction] = data[args.prediction].apply(categorizar)
        
        else:
            sys.exit(f"La columna '{args.prediction}' no existe en el dataset.")




def simplify_text(text_feature): # ???
    """
    Función que simplifica el texto de una columna dada en un DataFrame. lower,stemmer, tokenizer, stopwords del NLTK....
    Parámetros: text_feature: DataFrame - El DataFrame que contiene la columna de texto a simplificar.
    Retorna: None
    """
    global data
    try:
        if text_feature.columns.size > 0:
            stop_words = set(stopwords.words('english')) # Utiliza el conjunto de stopwords en inglés de nltk para filtrar palabras irrelevantes. 
            stemmer = PorterStemmer() # Usa el PorterStemmer de nltk para reducir las palabras a su raíz o forma base
            def process_text_column(text): # Funcion que procesa una sola entrada de texto
                if pd.isna(text):  # Si es NaN, lo dejamos tal cual
                    return text
                text = text.lower() # Convierte todo el texto a minusculas
                words = word_tokenize(text) # Tokeniza el texto en palabras individuales.
                words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalnum()] # Elimina las stopwords y aplica el stemming a cada palabra restante
                return ' '.join(words) # Devuelve las palabras procesadas unidas en una cadena de texto
            for col in text_feature.columns: # Por cada columna del dataframe
                data[col] = data[col].apply(lambda x: process_text_column(x) if pd.notna(x) else x) # No modificamos los NaN
                # ??? data[col] = data[col].astype(str).apply(process_text_column)
            print(Fore.GREEN + "Texto simplificado con éxito" + Fore.RESET)
            
        else:
            print(Fore.YELLOW+"No se han encontrado columnas de texto para simplificar"+Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al simplificar texto" + Fore.RESET)
        print(e)
        sys.exit(1)

def cat2num(categorical_feature): # ???
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.
    Parámetros: categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.
    """
    global data
    try:
        if categorical_feature.columns.size > 0:
            labelencoder = LabelEncoder()
            for col in categorical_feature.columns:
                 # Obtener valores únicos sin NaN y crear un diccionario de reemplazo
                unique_values = data[col].dropna().unique()
                mapping = {val: idx + 1 for idx, val in enumerate(sorted(unique_values))}
                # Usar replace en lugar de map
                data[col] = data[col].replace(mapping)
                # Convertir NaN a 0 --> data[col] = data[col].fillna(0).astype(int)
                
                #data[col] = labelencoder.fit_transform(data[col])
            print(Fore.GREEN+"Datos categóricos pasados a numéricos con éxito"+Fore.RESET)
            
        else:
            print(Fore.YELLOW+"No se han encontrado columnas categóricas que pasar a numericas"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al pasar los datos categóricos a numéricos"+Fore.RESET)
        print(e)
        sys.exit(1)
  
def process_missing_values(numerical_feature, categorical_feature, text_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.
    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas.
    Returns: None
    """
    global data
    try:
        if args.preprocessing["impute_strategy"] == "eliminar":
            data = data.dropna(subset=numerical_feature.columns)
            data = data.dropna(subset=categorical_feature.columns)
            data = data.dropna(subset=text_feature.columns)
            print(Fore.GREEN+"Missing values eliminados con éxito"+Fore.RESET)       
        elif args.preprocessing["impute_strategy"] == "media":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(data[numerical_feature.columns].mean())
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(data[categorical_feature.columns].mean())
                print(Fore.GREEN+"Missing values imputados con éxito usando la media"+Fore.RESET)
        elif args.preprocessing["impute_strategy"] == "mediana":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(data[numerical_feature.columns].median())
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(data[categorical_feature.columns].median())
                print(Fore.GREEN+"Missing values imputados con éxito usando la mediana"+Fore.RESET)
        elif args.preprocessing["impute_strategy"] == "moda":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(data[numerical_feature.columns].mode().iloc[0])
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(data[categorical_feature.columns].mode().iloc[0])
                print(Fore.GREEN+"Missing values imputados con éxito usando la moda"+Fore.RESET)
        else:
                print(Fore.GREEN+"No se ha seleccionado ninguna estrategia de imputación"+Fore.RESET)
        
        return data 
    except Exception as e:
        print(Fore.RED+"Error al tratar los missing values"+Fore.RESET)
        print(e)
        sys.exit(1)

def reescaler(numerical_feature): # ???
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.
    Args: numerical_feature (DataFrame): El dataframe que contiene las características numéricas.
    Returns: None
    Raises: Exception: Si hay un error al reescalar los datos.
    """
    global data
    try:
        if numerical_feature.columns.size > 0:
            if args.preprocessing["scaling"] == "minmax":
                scaler = MinMaxScaler()
            elif args.preprocessing["scaling"] == "standard":
                scaler = StandardScaler()
            elif args.preprocessing["scaling"] == "maxabs":
                scaler = MaxAbsScaler()
            elif args.preprocessing["scaling"] == "normalize":
                scaler = Normalizer()
            else:
                print(Fore.YELLOW + "No se está aplicando ningún reescalado" + Fore.RESET)
                return
            data[numerical_feature.columns] = scaler.fit_transform(numerical_feature)
            print(Fore.GREEN + "Reescalado exitoso aplicando con " + args.preprocessing["scaling"] + Fore.RESET)
            
        else:
            print(Fore.YELLOW+"No se han encontrado columnas numericas para escalar"+Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al reescalar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.
    Parámetros: text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.
    """
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] == "tf-idf":              
               tfidf_vectorizer = TfidfVectorizer()
               text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
               tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
               text_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
               text_features_df.to_csv("output/resultado_tf-idf.csv", index=False)
               data = pd.concat([data.reset_index(drop=True), text_features_df.reset_index(drop=True)], axis=1)
               data.drop(text_feature.columns, axis=1, inplace=True)
               print(Fore.GREEN+"Texto tratado con éxito usando TF-IDF"+Fore.RESET)
               
            elif args.preprocessing["text_process"] == "bow":
                bow_vecotirizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vecotirizer.get_feature_names_out())
                text_features_df.to_csv("output/resultado_bow.csv", index=False)
                data = pd.concat([data.reset_index(drop=True), text_features_df.reset_index(drop=True)], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.GREEN+"Texto tratado con éxito usando BOW"+Fore.RESET)
                
            else:
                print(Fore.YELLOW+"No se están tratando los textos"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas de texto a procesar"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)

def over_under_sampling(): # ???
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.preprocessing["sampling"].
    Args: None
    Returns: None
    Raises: Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """
    global data
    try:
        if args.mode != "train":
            print(Fore.YELLOW + "El muestreo solo se realiza en modo 'train'. Saltando esta etapa..." + Fore.RESET)
            return
        x = data.drop(columns=[args.prediction]) # resto de atributos
        y = data[args.prediction] # atributo a predecir
        if args.preprocessing["sampling"] == "oversampling":
            sampler = RandomOverSampler()
        elif args.preprocessing["sampling"] == "undersampling":
            sampler = RandomUnderSampler()
        else:
            print(Fore.YELLOW + "No se está aplicando oversampling ni undersampling" + Fore.RESET)
            return
        x_resampled, y_resampled = sampler.fit_resample(x, y) # Aplicar muestreo
        data = pd.concat([pd.DataFrame(x_resampled, columns=x.columns), pd.Series(y_resampled, name=args.prediction)], axis=1)
        print(Fore.GREEN + "Muestreo aplicado con éxito" + Fore.RESET)
        
    except Exception as e:
        print(Fore.RED + "Error en el muestreo" + Fore.RESET)
        print(e)
        sys.exit(1)
  
def drop_features():
    """
    Elimina las columnas especificadas del conjunto de datos.
    Parámetros: features (list): Lista de nombres de columnas a eliminar.
    """
    global data
    try:
        data = data.drop(columns=args.preprocessing["drop_features"])
        print(Fore.GREEN+"Columnas eliminadas con éxito"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        print(e)
        sys.exit(1)


""" ########################################
    FUNCIONES PARA ENTRENAR UN MODELO
######################################## """

def divide_data(): # ???
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.
    Parámetros:
        - data: DataFrame que contiene los datos.
        - args: Objeto que contiene los argumentos necesarios para la división de datos.
    Retorna:
        - x_train: DataFrame con las características de entrenamiento.
        - x_dev: DataFrame con las características de desarrollo.
        - y_train: Serie con las etiquetas de entrenamiento.
        - y_dev: Serie con las etiquetas de desarrollo.
    """
    # ??? Sacamos la columna a predecir global data
    try:
        x = data.drop(columns=[args.prediction]) # Obtener todas las columnas menpos la que queremos predecir
        y = data[args.prediction] # Obtener la columna que queremos predecir
         # Primera división: 70% entrenamiento, 30% restante (dev + test)
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42) # punto de partida del muestreo (semilla)
        # Segunda división: 50% del conjunto restante (15% dev, 15% test)
        x_dev, x_test, y_dev, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
        print(Fore.GREEN + "Datos divididos con éxito" + Fore.RESET)
        # Imprimir cada porción de datos
        print(f"  >> CONJUNTO DE ENTRENAMIENTO (70%) :")
        print("Datos de entrada :")
        print(x_train.head(), "\n") # Contiene las caracteristicas usadas para entrenar el modelo
        print("Datos de salida :")
        print(y_train.head(), "\n") # Contiene los valores reales que el modelo debe aprender a predecir
        print(f"  >> CONJUNTO DE DEVELOPMENT (15%) :")
        print("Datos de entrada :")
        print(x_dev.head(), "\n") # Se usa para ajustar hiperparametros y evitar sobreajuste
        print("Datos de salida :")
        print(y_dev.head(), "\n") # Se compara con las predicciones del modelo en la fase de ajuste
        print(f"  >> CONJUNTO DE PRUEBA (15%) :")
        print("Datos de entrada :")
        print(x_test.head(), "\n") # Se usa para evaluar el modelo final
        print("Datos de salida :")
        print(y_test.head(), "\n") # Se usa para calcular la precision del modelo antes de usarlo en datos reales
        # Guardar x_test e y_test en un archivo CSV
        test_data = pd.concat([x_test, y_test], axis=1)  # Combina características y etiquetas
        test_file_name = os.path.join("output", args.file.replace(".csv", "_test.csv")) # Cambia el nombre del archivo y crearlo en la carpeta
        test_data.to_csv(test_file_name, index=False)  # Guarda el archivo
        print(Fore.GREEN + f"Conjunto de prueba guardado en: {test_file_name}" + Fore.RESET)
        return x_train, x_dev, y_train, y_dev ###estos son para entrenar
    except Exception as e:
        print(Fore.RED + "Error al dividir los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

def kNN():
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos
    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """ 
    x_train, x_dev, y_train, y_dev = divide_data() # Dividimos los datos en entrenamiento y development
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar: # Mostrar "Precesando Knn" mientras se hacen 100 iteraciones
        gs = GridSearchCV( # Realizar una busqueda de hiperparametros
            KNeighborsClassifier(), 
            args.kNN, # obtener conjuntos de hiperparametros a predecir
            scoring= { # Definir metricas de evaluacion del modelo
                'accuracy': 'accuracy',
                'precision_weighted': 'precision_weighted',
                'recall_weighted': 'recall_weighted',
                'f1': 'f1'
            }, 
            refit='f1',
            cv=5, # Validacion cruzada con 5 particiones (divide los datos en 5 grupos para evaluar el modelo)
            n_jobs=args.cpu # Usar varios nucleos de CPU para acelerar la busqueda
        )
        start_time = time.time() # Capturar tiempo inicial antes de entrenar el modelo
        gs.fit(x_train, y_train) # Entrena el modelo GridSearchCV con los datos de entrenamiento para encontrar la mejor combinacion de hiperparametros
        end_time = time.time() # Capturar tiempo al terminar de entrenar el modelo
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time # Calcular el tiempo que ha tardado en entrenarse el GridSearchCV
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    mostrar_resultados(gs, x_dev, y_dev) # Mostramos los resultados
    save_model(gs) # Guardamos el modelo utilizando pickle

def decision_tree():
    """
    Función para implementar el algoritmo de árbol de decisión.
    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """ 
    x_train, x_dev, y_train, y_dev = divide_data() # Dividimos los datos en train y development
    # Hacemos un barrido de hiperparámetros
    with tqdm(total=100, desc='Procesando Decision Tree', unit='iter', leave=True) as pbar:  # Mostrar "Procesando Precision Tree" mientras se hacen 100 iteraciones
        gs = GridSearchCV( # Realizar una busqueda de los mejores hiperparametros
            DecisionTreeClassifier(), 
            args.decision_tree, # obtener conjuntos de hiperparametros a predecir
            cv=5, # Validacion cruzada con 5 particiones (divide los datos en 5 grupos para evaluar el modelo)
            n_jobs=args.cpu, # Paraleliza el proceso en múltiples núcleos de la CPU.
            scoring= { # Definir metricas de evaluacion del modelo
                'accuracy': 'accuracy',
                'precision_weighted': 'precision_weighted',
                'recall_weighted': 'recall_weighted',
                'f1': 'f1'
            }, 
            refit='f1'
        )
        start_time = time.time() # Capturar tiempo inicial antes de entrenar el modelo
        gs.fit(x_train, y_train) # Entrena el modelo GridSearchCV con los datos de entrenamiento para encontrar la mejor combinacion de hiperparametros
        end_time = time.time() # Capturar tiempo al terminar de entrenar el modelo
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random() * 2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time # Calcular el tiempo que ha tardado en entrenarse el GridSearchCV
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
    mostrar_resultados(gs, x_dev, y_dev) #  Mostramos los resultados
    save_model(gs) # Guardamos el modelo utilizando pickle
    
def random_forest():
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.
    Parámetros: Ninguno
    Retorna: Ninguno
    """
    x_train, x_dev, y_train, y_dev = divide_data()  # Dividimos los datos en entrenamiento y development
    # Hacemos un barrido de hiperparámetros
    with tqdm(total=100, desc='Procesando Random Forest', unit='iter', leave=True) as pbar: # Mostrar "Procesando Random Forest" mientras se hacen 100 iteraciones
        gs = GridSearchCV(
            RandomForestClassifier(), 
            args.random_forest, # obtener conjuntos de hiperparametros a predecir
            cv=5, # Validacion cruzada con 5 particiones (divide los datos en 5 grupos para evaluar el modelo)
            n_jobs=args.cpu, # Paraleliza el proceso en múltiples núcleos de la CPU.
            scoring= { # Definir metricas de evaluacion del modelo
                'accuracy': 'accuracy',
                'precision_weighted': 'precision_weighted',
                'recall_weighted': 'recall_weighted',
                'f1': 'f1'
            }, 
            refit='f1'
        )
        start_time = time.time() # Capturar tiempo inicial antes de entrenar el modelo
        gs.fit(x_train, y_train) # Entrena el modelo GridSearchCV con los datos de entrenamiento para encontrar la mejor combinacion de hiperparametros
        end_time = time.time() # Capturar tiempo al terminar de entrenar el modelo
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random() * 2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time  # Calcular el tiempo que ha tardado en entrenarse el GridSearchCV
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
    mostrar_resultados(gs, x_dev, y_dev) # Mostramos los resultados
    save_model(gs) # Guardamos el modelo utilizando pickle

def naive_bayes():
    x_train, x_dev, y_train, y_dev = divide_data()
    """
    try:
        x = data.drop(columns=[args.prediction])  # Características
        y = data[args.prediction]  # Etiqueta a predecir
        # Dividir en 80% entrenamiento y 20% prueba
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        print(Fore.GREEN + "Datos divididos en entrenamiento y prueba con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al dividir los datos" + Fore.RESET)
        print(e)
        sys.exit(1)
    """
    
    if x_train.select_dtypes(include=['object']).shape[1] > 0:  # Si hay columnas categóricas
        model = GridSearchCV(
            CategoricalNB(),
            args.naive_bayes_cat,
            scoring= { # Definir metricas de evaluacion del modelo
                'accuracy': 'accuracy',
                'precision_weighted': 'precision_weighted',
                'recall_weighted': 'recall_weighted',
                'f1_macro': 'f1_macro'
            }, 
            refit='f1_macro' ,  # Métrica de evaluación
            cv=5,  # Validación cruzada
            n_jobs=args.cpu  # Usar múltiples núcleos
        )
    else:
        model = GridSearchCV(
            GaussianNB(),
            args.naive_bayes_num,
            scoring= { # Definir metricas de evaluacion del modelo
                'accuracy': 'accuracy',
                'precision_weighted': 'precision_weighted',
                'recall_weighted': 'recall_weighted',
                'f1_macro': 'f1_macro'
            }, 
            refit='f1_macro' ,  # Métrica de evaluación
            cv=5,  # Validación cruzada
            n_jobs=args.cpu  # Usar múltiples núcleos
        )
    print(Fore.GREEN + "Entrenando modelo Naive Bayes..." + Fore.RESET)
    model.fit(x_train, y_train)

    mostrar_resultados(model, x_dev, y_dev)  # Mostrar resultado
    # Guardar el modelo
    save_model(model)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.
    Parámetros:
        - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
        - x_dev: array-like, las características del conjunto de desarrollo.
        - y_dev: array-like, las etiquetas del conjunto de desarrollo.
    Imprime en la consola los siguientes resultados:
        - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
        - Mejor puntuación obtenida por el clasificador.
        - F1-score micro del clasificador en el conjunto de desarrollo.
        - F1-score macro del clasificador en el conjunto de desarrollo.
        - Informe de clasificación del clasificador en el conjunto de desarrollo.
        - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose: # Comprobar si se quiere mostrar las metricas
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_) # Muestra la mayor metrica
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, gs.predict(x_dev)))
         # Calcular métricas
        y_pred = gs.predict(x_dev) # Generar predicciones usando el modelo entrenado con los mejores hiperparametros
        fscore_micro, fscore_macro = calculate_fscore(y_dev, y_pred) # Calcular los valores F-score micro y macro
        classification_rep = classification_report(y_dev, y_pred, output_dict=True, zero_division=0) # Obtener el informe de clasificacion en formato de diccionario
        confusion_mat = calculate_confusion_matrix(y_dev, y_pred) # Obtener la matriz de confusion
        # Obtener el accuracy, precision, recall y F-score del informe de clasificacion obtenido
        accuracy = classification_rep["accuracy"]
        precision = classification_rep["weighted avg"]["precision"]
        recall = classification_rep["weighted avg"]["recall"]
        f1_score_weighted = classification_rep["weighted avg"]["f1-score"]
        # Guardar los resultados en un archivo csv
        results_file = "output/metricasMejorModeloEntrenado.csv"
        with open(results_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Mejores parametros", gs.best_params_])
            writer.writerow(["Mejor puntuacion", gs.best_score_])
            writer.writerow(["Accuracy", accuracy])
            writer.writerow(["Precision (weighted)", precision])
            writer.writerow(["Recall (weighted)", recall])
            writer.writerow(["F1-score (weighted)", f1_score_weighted])
            writer.writerow(["F1-score (micro)", fscore_micro])
            writer.writerow(["F1-score (macro)", fscore_macro])
        print(Fore.GREEN + f"Resultados guardados en: {results_file}" + Fore.RESET)

def save_model(gs):
    """
    Guarda el mejor modelo entrsave_model(gs)enado y los resultados de la búsqueda de hiperparámetros en archivos.
    """
    try:
        # Guardar el mejor modelo entrenado
        best_model = gs.best_estimator_  # Obtener el modelo entrenado con los mejores hiperparámetros
        with open('output/mejorModelo.pkl', 'wb') as file:
            pickle.dump(best_model, file) # para guardar el modelo entrenado en un archivo, de modo que pueda ser cargado y reutilizado más adelante.
            print(Fore.CYAN + "Mejor modelo guardado con éxito en output/mejorModelo.pkl" + Fore.RESET)
        # Guardar los resultados de todos los intentos en todosModelo.csv
        with open('output/todosModelo.csv', 'w', newline='') as file:
            writer = csv.writer(file) # Crear objeto para escribir en formato CSV
            writer.writerow(['Params', 'Score', 'Accuracy', 'Precision (weighted)', 'Recall (weighted)', 'F1_macro']) # Escribir encabezados de las metricas que se van a almacenar
            for params, score, accuracy, precision, recall, f1 in zip(  # Escribir los resultados de todos los intentos
                gs.cv_results_['params'],
                gs.cv_results_['mean_test_f1_macro'],  # Cambiar 'f1' por 'mean_test_f1'
                gs.cv_results_['mean_test_accuracy'],
                gs.cv_results_['mean_test_precision_weighted'],
                gs.cv_results_['mean_test_recall_weighted'],
                gs.cv_results_['mean_test_f1_macro']  # Aquí también usar 'mean_test_f1'
            ):
                writer.writerow([params, score, accuracy, precision, recall, f1])
        
        print(Fore.GREEN + "Resultados de todos los intentos guardados en output/todosModelo.csv" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al guardar el modelo o los resultados" + Fore.RESET)
        print(e)


""" ########################################
    FUNCIONES PARA CALCULAR METRICAS
######################################## """

def calculate_fscore(y_test, y_pred):
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    fscore_micro = f1_score(y_test, y_pred, average='micro') ###da mas peso a las clases con mas datos
    fscore_macro = f1_score(y_test, y_pred, average='macro') ### calcula la media de fscore de cada clase sin tener en cuenta la cantidad de datos
    return fscore_micro, fscore_macro

def calculate_classification_report(y_test, y_pred):
    """
    Función para calcular el informe de clasificación (tabla que muestra metricas de evaluacion para cada clase)
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Informe de clasificación
    """
    report = classification_report(y_test, y_pred, zero_division=0)
    return report

def calculate_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    cm = confusion_matrix(y_test, y_pred)
    return cm

""" ########################################
    FUNCIONES PARA PREDECIR CON UN MODELO
######################################## """

def load_model():
    """
    Carga el modelo desde el archivo 'output/mejorModelo.pkl' y lo devuelve.
    Returns: model: El modelo cargado desde el archivo 'output/mejorModelo.pkl'.
    Raises: Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open('output/mejorModelo.pkl', 'rb') as file: 
            model = pickle.load(file)
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)

def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.
    Parámetros: Ninguno
    Retorna: Ninguno
    """
    global data
    # Eliminamos la columna a predecir
    data = data.drop(columns=[args.prediction])
    # Predecimos
    prediction = model.predict(data)
    # Añadimos la prediccion al dataframe data
    data = pd.concat([data, pd.DataFrame(prediction, columns=[args.prediction])], axis=1)
    
def predict_score(review, model_bin, model_ord, alpha=0.4, beta=0.6):
    """
    Recibe una review y devuelve una puntuación final entre 20 y 100.
    """
    # 1. Probabilidad de clase 1 para modelo binario
    prob_bin = model_bin.predict_proba([review])[0][1]  # valor entre 0 y 1
    
    # 2. Predicción del modelo ordinal (valor entre 1 y 5)
    pred_ord = model_ord.predict([review])[0]
    prob_ord = (pred_ord - 1) / 4  # normalizado a 0–1

    # 3. Combinar
    score_combinado = alpha * prob_bin + beta * prob_ord
    final_score = 20 + score_combinado * 80

    return final_score

    
""" ########################################
    FUNCION PRINCIPAL
######################################## """

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        print(e)
        sys.exit(1)
    # Cargamos los datos
    print("\n- Cargando datos...")
    data = load_data(args.file)
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preprocesar_datos()
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-preprocesado.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
    if args.mode == "train":
        # Ejecutamos el algoritmo seleccionado
        print("\n- Ejecutando algoritmo...")
        if args.algorithm == "kNN":
            try:
                kNN()
                print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "decision_tree":
            try:
                decision_tree()
                print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "random_forest":
            try:
                random_forest()
                print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "naive_bayes":
            try:
                naive_bayes()
                print(Fore.GREEN+"Algoritmo naive_bayes ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        else:
            print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
            sys.exit(1)
    elif args.mode == "test":
        print("\n- Cargando modelos combinados...")
        try:
            with open('output/modelo_airbnb.pkl', 'rb') as f:
                modelo_binario = pickle.load(f)
            with open('output/modelo_tripadvisor.pkl', 'rb') as f:
                modelo_ordinal = pickle.load(f)
            print(Fore.GREEN + "Modelos cargados con éxito" + Fore.RESET)
        except Exception as e:
            print(Fore.RED + "Error al cargar los modelos combinados" + Fore.RESET)
            print(e)
            sys.exit(1)

        # Cargamos y preprocesamos los datos
        print("\n- Preprocesando datos para predicción...")
        preprocesar_datos()

        # Asumimos que hay una columna de texto identificada
        columna_texto = args.columna_texto if hasattr(args, 'columna_texto') else 'review'

        if columna_texto not in data.columns:
            print(Fore.RED + f"La columna de texto '{columna_texto}' no existe." + Fore.RESET)
            sys.exit(1)

        # Calculamos las puntuaciones finales
        print("\n- Calculando puntuaciones finales...")
        puntuaciones = []
        for review in data[columna_texto]:
            puntuacion = predict_score(review, modelo_binario, modelo_ordinal)
            puntuaciones.append(puntuacion)

        data["puntuacion_final"] = puntuaciones

        # Guardamos los resultados
        salida = 'output/data-puntuacion-final.csv'
        data.to_csv(salida, index=False)
        print(Fore.GREEN + f"Puntuaciones guardadas en {salida}" + Fore.RESET)
        sys.exit(0)
            
    else:
        print(Fore.RED+"Modo no soportado"+Fore.RESET)
        sys.exit(1)
