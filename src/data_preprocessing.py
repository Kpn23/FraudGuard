import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
)
import matplotlib.pyplot as plt
from datetime import datetime
import category_encoders as ce


def find_missing_value(df):
    """
    Data Cleaning: This involves identifying and correcting errors in the dataset,
    such as removing duplicates, correcting inconsistencies, and addressing erroneous data points.
        Option
        1. Drop rows with missing value [.dropna() or fillna()]
        2. Filter empty row only when the rows in select columns has missing value
    """
    missing_values = df.isna().sum()

    # duplicates = df_train.duplicated()
    # print(f"list out boolean of rows:\n{duplicates}")
    # duplicates_row = df_train[duplicates]
    # print(f"list out duplicated rows:\n{duplicates_row}")

    # count = duplicates_row.shape[0]
    # print(f"number of duplicated rows:{count}")

    # df_train_drop_duplicated = df_train.drop_duplicates()
    # print(f"list out rows without duplicates: {df_train_drop_duplicated}")

    # Remove duplicated rows
    # df = df.drop_duplicates()
    return missing_values


def clean_data(df):
    """
    Handling Missing Values: Techniques such as imputation (replacing missing values with estimated values),
    interpolation, or deletion are employed to manage missing data effectively.
    """
    # Fill missing values or drop rows
    df = df.fillna(method="ffill")
    df = df.dropna()
    print("Missing values after cleaning:\n", df.isnull().sum())
    df.info()
    return df


def preprocess_data(df):
    """convert data type
    involve numerical / catergorical conversion
    Label Encoding: Suitable for ordinal categorical variables: May not be applicable
    one-Hot Encoding: Suitable for nominal categorical variables.
    Scaling is crucial for algorithms like logistic regression
    """
    df["dob"] = pd.to_datetime(df["dob"])
    df["dob"] = datetime.now().year - df["dob"].dt.year
    df = df.rename(columns={"dob": "age"})

    object_columns = [
        "trans_date_trans_time",
        "merchant",
        "category",
        "first",
        "last",
        "gender",
        "street",
        "city",
        "state",
        "job",
        "trans_num",
    ]
    updated_object_columns = object_columns

    categorical_columns = [
        "trans_date_trans_time",
        "merchant",
        "category",
        "first",
        "last",
        "gender",
        "street",
        "city",
        "state",
        "job",
        "trans_num",
    ]
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])
    df_encoded = df

    colums_without_isfraud = df_encoded.columns.tolist()
    colums_without_isfraud.remove("is_fraud")

    for col in categorical_columns:
        if col in updated_object_columns:
            updated_object_columns.remove(col)

    for col in updated_object_columns:
        if col in colums_without_isfraud:
            colums_without_isfraud.remove(col)

    df_temp = df_encoded.drop(["is_fraud"], axis=1)
    df_temp = df_temp.drop(columns=updated_object_columns, axis=1)
    encoder = StandardScaler()
    df_temp[colums_without_isfraud] = encoder.fit_transform(
        df_temp[colums_without_isfraud]
    )
    df_encoded[colums_without_isfraud] = df_temp[colums_without_isfraud]
    return df_encoded


def balance_class(df):
    """For logistic regression
    Handle class imbalance: If one class is much more prevalent than the other,
    use techniques like oversampling, undersampling, or SMOTE to balance the classes
    """
    # Check distribution of label data
    fraud_row = df[df.is_fraud == 1].shape[0]
    not_fraud_row = df[df.is_fraud == 0].shape[0]
    fraud_target_percentage = fraud_row / (fraud_row + not_fraud_row) * 100
    # undersampling
    not_fraud = df[df.is_fraud == 0].sample(fraud_row, random_state=42)
    fraud = df[df.is_fraud == 1]
    # Check the population distribution
    # plt.figure(figsize=(20, 7))
    # df_balance[df_balance["is_fraud"] == 1].groupby("age").count()["is_fraud"].plot(
    #     kind="bar"
    # )
    # df_balance[df_balance["is_fraud"] == 0].groupby("age").count()["is_fraud"].plot(
    #     kind="bar"
    # )
    # plt.legend()
    # plt.show()
    # plt.clf()

    # Assuming df_balance is your DataFrame
    # fraud_counts = (
    #     df_balance[df_balance["is_fraud"] == 1].groupby("age").count()["is_fraud"]
    # )
    # non_fraud_counts = (
    #     df_balance[df_balance["is_fraud"] == 0].groupby("age").count()["is_fraud"]
    # )

    # ages = fraud_counts.index  # Get the unique ages

    # fig, ax = plt.subplots(figsize=(20, 7))

    # bar_width = 0.4
    # x = range(len(ages))

    # ax.bar(x, fraud_counts, width=bar_width, color="b", label="Fraud")
    # ax.bar(
    #     [i + bar_width for i in x],
    #     non_fraud_counts,
    #     width=bar_width,
    #     color="g",
    #     label="Non-Fraud",
    # )

    # ax.set_xticks([i + bar_width / 2 for i in x])
    # ax.set_xticklabels(ages)

    # ax.set_xlabel("Age")
    # ax.set_ylabel("Count")
    # ax.set_title("Fraud vs Non-Fraud by Age")
    # ax.legend()

    # plt.show()
    return fraud, not_fraud, fraud_target_percentage
