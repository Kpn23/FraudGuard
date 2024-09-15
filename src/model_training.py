from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(fraud_FT, not_fraud_FT):
    train_fraud_FT, temp_fraud_FT = train_test_split(
        fraud_FT, test_size=0.3, random_state=12
    )
    test_fraud_FT, valid_fraud_FT = train_test_split(
        temp_fraud_FT, test_size=0.5, random_state=12
    )

    train_not_fraud_FT, temp_not_fraud_FT = train_test_split(
        not_fraud_FT, test_size=0.3, random_state=12
    )
    test_not_fraud_FT, valid_not_fraud_FT = train_test_split(
        temp_not_fraud_FT, test_size=0.5, random_state=12
    )
    return (
        train_fraud_FT,
        test_fraud_FT,
        valid_fraud_FT,
        train_not_fraud_FT,
        test_not_fraud_FT,
        valid_not_fraud_FT,
    )


def concat_data(fraud_FT, not_fraud_FT):
    concat = pd.concat([fraud_FT, not_fraud_FT], ignore_index=True)
    return concat


def prepare_data(concat):
    x = concat.drop("is_fraud", axis=1)
    y = concat["is_fraud"]
    return x, y


def train_model(x_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)
    return model
