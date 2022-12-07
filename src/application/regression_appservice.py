import pandas as pd
import numpy as np

from domain.models.results import LinearRegressionResult, LogisticRegressionResult
from domain.services import regression_service


def handle_text(dataset: pd.DataFrame):
    """Prepares dataset for regression algorithms, handling or removing text columns.

    Args:
        dataset (pd.DataFrame): The dataset to be normalized.

    Returns:
        pd.DataFrame: The normalized dataframe.
    """
    # converts categories into columns (dummies)
    categories = pd.get_dummies(dataset["category"],drop_first=True)
    categories = pd.DataFrame.add_prefix(categories, 'category_')

    # drops text columns
    df = dataset.drop(["fancyname","category","company"],axis=1)
    
    # drops score columns that are not the target
    df.drop(["Downloads","numberreviews","five", "four", "three", "two", "one"],axis=1,inplace=True)
    
    # adds categories as dummies
    return pd.concat([df,categories],axis=1)



def linear(dataset: pd.DataFrame, df_test_size: float = 0.3) -> LinearRegressionResult:
    """Executes Linear Regression on given DataFrame

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to train on.
        df_test_size (float, optional): Percentage of the dataframe (0-1) to use for testing. Defaults to 0.3 (30%).

    Returns:
        LinearRegressionResult: A class object containig results data.
    """
    
    df = handle_text(dataset)
    
    result = regression_service.linear(df, "rating", df_test_size)
    
    return result

def logistic(dataset: pd.DataFrame, df_test_size: float = 0.3) -> LogisticRegressionResult:
    """Executes Logistic Regression on given DataFrame

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to train on.
        df_test_size (float, optional): Percentage of the dataframe (0-1) to use for testing. Defaults to 0.3 (30%).

    Returns:
        LogisticRegressionResult: A class object containig results data.
    """
    df = handle_text(dataset)
    
    # creating the classification column
    df['top_rating'] = np.where(df['rating'] > 4.5, 1, 0)
    
    result = regression_service.logistic(df, "top_rating", df_test_size)
    
    return result








