import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_model(data_path):

    df = pd.read_csv(data_path, sep=";")

    df = df.drop_duplicates()

    df = df.dropna(subset=["G3"])

    features = [
        "studytime",
        "G1",
        "G2",
        "absences"
    ]

    X = df[features]
    y = df["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "regressor",
            LinearRegression()
        )
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    return model


if __name__ == "__main__":

    model = train_model(
        "../data/student_performance_dataset.csv"
    )

    print("Model training completed.")