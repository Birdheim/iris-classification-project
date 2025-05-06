import pandas as pd
from sklearn.datasets import load_iris, load_wine

def load_iris_dataset(as_frame: bool = True):
    iris = load_iris()

    if not as_frame:
        return iris
    
    df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

def load_wine_dataset(as_frame: bool = True):
    wine = load_wine()

    if not as_frame:
        return wine
    
    df = pd.DataFrame(data= wine.data, columns= wine.feature_names)
    df['cultivars'] = pd.Categorical.from_codes(wine.target, wine.target_names)
    return df

