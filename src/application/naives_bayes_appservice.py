import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from domain.services import naives_bayes_service


from domain.services import repository_service
from domain.utils import constants
from domain.models.results import NaivesBayesResult

def naives_bayes(dataset: pd.DataFrame, df_test_size: float = 0.3) -> NaivesBayesResult:
    """Executes Naives Bayes on given DataFrame.

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to train on.
        df_test_size (float, optional): Percentage of the dataframe (0-1) to use for testing. Defaults to 0.3 (30%).

    Returns:
        NaivesBayesResult: A class object containig results data.
    """
    encoder = preprocessing.LabelEncoder()
    
    # drops text columns
    df = dataset.drop(["fancyname","company"],axis=1)
    
    # converts categories into index (labels)
    categoria_encoded = encoder.fit_transform(df["category"])
    df['category'] = categoria_encoded
    
    
    target = 'top_rating'
    # creating classification column
    df[target] = np.where(df['rating'] > 4.5, 1, 0)
    
    # dropping score columns that are not the target    
    df = df.drop(["Downloads","numberreviews","five", "four", "three", "two", "one", "rating"],axis=1)
    
    result = naives_bayes_service.naives_bayes(df, 'top_rating', df_test_size)
    return result