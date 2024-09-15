import pandas as pd


def feature_engineering(fraud, not_fraud, features_list, target):
    if not features_list:
        print("you haven't select any feature!!!")
    else:
        features_list.append(target)
    fraud_FT = fraud[features_list]
    not_fraud_FT = not_fraud[features_list]

    return fraud_FT, not_fraud_FT
