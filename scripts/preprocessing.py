from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test, **Kwargs):
    """
    Scale features using StandardScaler.

    Parameters:
    - X_train: Training feature data.
    - X_test: Test feature data.
    - **Kwargs: sd.

    Returns:
    - X_train_scaled: Scaled training data
    - X_test_scaled: Scaled test data
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled