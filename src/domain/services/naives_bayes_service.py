import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

from domain.models.results import NaivesBayesResult

def naives_bayes(dataset: pd.DataFrame, target: str, df_test_size: float) -> NaivesBayesResult:
    """Executes Naives Bayes on given DataFrame.

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to train on.
        target (str): The name of the target column.
        df_test_size (float): Percentage of the dataframe (0-1) to use for testing.

    Returns:
        NaivesBayesResult: A class object containig results data.
    """    
    
    # instantiating class to store results
    result = NaivesBayesResult()
    
    # x = training resources (predictor), y = target (predicted)
    X = dataset.drop([target], axis = 1)
    Y = dataset[target]
    
    # getting train and test data
    x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size= df_test_size, random_state=0)
    
    # training and learning
    nb_model = GaussianNB()
    nb_model.fit(x_train,y_train)
    
    # predicting with the test row (x_test)
    predictions = nb_model.predict(x_test)
    
    result.normalized_data = dataset
    result.score = nb_model.score(x_test, y_test)
    result.classification_report = metrics.classification_report(y_test,predictions)
    result.confusion_matrix = metrics.confusion_matrix(y_test, predictions, labels=[True, False])
    
    return result