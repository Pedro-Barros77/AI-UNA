import pandas as pd
import numpy as np
from sklearn import preprocessing

from domain.models.results import KNNResult, KMeansResult
from domain.services import k_algorithms_service
from domain.utils import math


def knn(dataset: pd.DataFrame, num_neighbors:int = 3, df_test_size: float = 0.3) -> KNNResult:
    """Executes K-Nearest Neighbors on given DataFrame

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to train on.
        num_neighbors (int, optional): Number of neighbors to calculate distance to. Defaults to 3.
        df_test_size (float, optional): Percentage of the dataframe (0-1) to use for testing. Defaults to 0.3 (30%).

    Returns:
        KNNResult: A class object containig results data.
    """
    
    encoder = preprocessing.LabelEncoder()
    
    # converts categories into index (labels)
    categories = encoder.fit_transform(dataset["category"])
    category_column = pd.DataFrame(categories, columns=['category'])
    
    # drops text columns
    df = dataset.drop(["fancyname","category","company"],axis=1)
    
    # appending the encoded category column
    df = pd.concat([df, category_column],axis=1)
    
    # creating classification column
    df['top_rating'] = np.where(df['rating'] > 4.5, 1, 0)
    
    # dropping score columns that are not the target    
    df = df.drop(["Downloads","numberreviews","five", "four", "three", "two", "one", "rating"],axis=1)
    
    result = k_algorithms_service.knn(df, 'top_rating', num_neighbors, df_test_size)
    
    return result
    
    
def kmeans(dataset: pd.DataFrame, num_clusters:int | None = None, dimensions: tuple[str,str] | int | None = None) -> KMeansResult:
    """Executa o algoritmo KMeans no dataframe especificado.

    Args:
        dataset (pd.DataFrame): O dataset contendo os dados.
        num_clusters (int | None, optional): O número de clusters (centróides) a serem criados. Caso seja none, serão feitos testes com os valores [2, 3, 4, 5, 10, 15, 20] e escolhido aquele que apresentar o maior score. Defaults to None.
        dimensions (tuple[str,str] | int | None): O nome das colunas que representarão as dimensões X e Y. Caso seja none, será aplicado o PCA para definir as dimensões com maior peso. Caso seja um número, serão utilizadas as dimensões de um dos 5 exemplos.
    """
    
    encoder = preprocessing.LabelEncoder()
    
    # converts categories into index (labels)
    categories = encoder.fit_transform(dataset["category"])
    category_column = pd.DataFrame(categories, columns=['category'])
    
    # drops text columns
    df = dataset.drop(["fancyname","category","company"],axis=1)
    
    # appending the encoded category column
    df = pd.concat([df, category_column],axis=1)
    
    # creating classification column
    df['top_rating'] = np.where(df['rating'] > 4.5, 1, 0)
    
    # if score columns (that are not game features) should be included
    include_scores = False
    
    # exemplos para testar
    dimension_examples = [
        None, # Será calculado PCA. Testar com 'include_scores' True e False.
        ('price', 'rating'), # Relação de avaliação por preço
        ('usersinteract','category'), # Quais categorias têm uma boa aceitação nos jogos multiplayer
        ('numberreviews','price'), # Quantidade de avaliações por preço. 'include_scores' deve ser True
        ('numberreviews','five'), # Relação de 5 estrelas para quantidade de avaliações. 'include_scores' deve ser True
    ]
    
    if type(dimensions) == int:
        index = math.clamp(dimensions, 0, len(dimension_examples)-1)
        dimensions = dimension_examples[index]
        if index in [3,4]:
            include_scores = True
            
    # dropping score columns that are not the target 
    if not include_scores:
        df = df.drop(["Downloads","numberreviews","five", "four", "three", "two", "one"],axis=1)
    
    return k_algorithms_service.kmeans(df, 'top_rating', dimensions= dimensions, num_clusters= num_clusters)