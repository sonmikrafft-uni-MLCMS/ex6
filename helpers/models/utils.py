# module for prediction related utiliy functions
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

RANDOM_STATE = 42

"""
Full example

>>> X_learn, X_test, y_learn, y_test = get_learn_test()
>>> # find the best hyperparameters
>>> mse_train = []
>>> mse_val = []
>>> for i in range(50):
>>>     X_train, X_val, y_train, y_val = bootstrap_and_split(X, y, n_samples=2000, test_size=0.5)
>>>     model.fit(X_train, y_train)
>>>     mse_train.append(model.score(X_train, y_train))
>>>     mse_val.append(model.score(X_val, y_val))
>>> # final performance only once at the end after we have found the best hyperparameters
>>> model.fit(X_learn, y_learn)
>>> final_mse = model.score(X_test, y_test)
"""


def get_learn_test(test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get the data for the learning and testing.

    This function will load the data from the data directory and split it into 80% for learning and 20% for testing.

    learning: The data for fitting the model and finding the best set of hyperparameters.
    testing: Serves as a proxy for the model's performance on unseen data and should only be used once at the very end
        to report the final performance of the model.

    Args:
        test_size (float, optional): Fraction of the data to reserve for testing. Defaults to 0.2.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple where:
            first element is X_learn,
            second element is y_learn,
            third element is X_test,
            fourth element is y_test.
    """
    df = pd.read_csv(os.path.join("data", "modified", "bottleneck_070.csv"))

    # only ones where we have 10 neighbours
    df = df[~df["DX_10"].isna()]

    # define features and target
    features = ["AVG_DISTANCE_TO_K_CLOSEST"]
    for i in range(10):
        str_dx = f"DX_{i+1}"
        str_dy = f"DY_{i+1}"
        features.append(str_dx)
        features.append(str_dy)
    target = "VEL"
    X = df[features]
    y = df[target]

    # split the data
    X_learn, X_test, y_learn, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    return X_learn, X_test, y_learn, y_test


def bootstrap_and_split(
    X: pd.DataFrame, y: pd.DataFrame, n_samples: int = 1000, test_size: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bootstrap the data and split it into training and testing.

    Creates a bootstrap sample of the data and splits it into training and testing.

    Args:
        X (pd.DataFrame): Input data. Of shape (n_samples, n_features).
        y (pd.DataFrame): Target data. Of shape (n_samples,).
        n_samples (int, optional): Number of samples to draw with replacement. Defaults to 1000.
        test_size (float, optional): Fraction of the data to reserve for testing for each draw of the bootstrap.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: [description]
    """
    X_bootstrapped, y_bootstrapped = resample(X, y, n_samples=n_samples, replace=True)
    X_learn, X_test, y_learn, y_test = train_test_split(
        X_bootstrapped, y_bootstrapped, test_size=test_size, random_state=RANDOM_STATE
    )
    return X_learn, X_test, y_learn, y_test
