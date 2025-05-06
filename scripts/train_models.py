from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def train_logisitc_regression(X_train, y_train, max_iter=200, **kwargs):
    """
    Parameters:
        X_train: pd.DataFrame or np.ndarray
        y_train: pd.Series or np.ndarray
        max_iter: int, default 200
        **kwargs: any other LogisticRegression keyword arguments
    
    Returns:
        Trained LogisticRegression model
    """

    model = LogisticRegression(max_iter=max_iter, **kwargs)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, n_neighbors=5, **kwargs):
    """
    Parameters:
        X_train: pd.DataFrame or np.ndarray
        y_train: pd.Series or np.ndarray
        n_neighbors: int, default 5
        **kwargs: any other KNeighborsClassifier keyword arguments

    Returns:
        Trained KNeighborsClassifiers model
    """

    model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, max_depth=None, **kwargs):
    """
    Train a Decision Tree classifier
    
    Parameters:
    - X_train: pd.DataFrame or np.ndarray
    - y_train: pd.Series or np.ndarray
    - max_depth: int or None, default None

    Returns:
    - Trained DescisionTreeClassifier model
    """

    model = DecisionTreeClassifier(max_depth=max_depth, **kwargs)
    model.fit(X_train, y_train)
    return model
