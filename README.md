# 🚨 FraudGuard: Credit Card Fraud Detection 🚨

### Project Overview 🎯
Welcome to the FraudGuard project! Our mission is to detect fraudulent credit card transactions using cutting-edge machine learning techniques. Join us on this exciting journey as we uncover the secrets behind suspicious activities and keep your money safe!

### Dataset Source 📊
This project utilizes the **Credit Card Transactions Fraud Detection Dataset** available on [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection). This rich dataset provides the necessary information to train our models effectively.

## Project Structure
```bash
FraudDetection/
│
├── main.py 🚀
├── src/
│ ├── data_preprocessing.py 🧹
│ ├── feature_engineering.py 🔍
│ ├── model_training.py 🏋️‍♀️
│ └── model_evaluation.py 🧪
├── util/
│ └── azure_connection.py 🌐
├── data/
│ ├── combine.csv 🔀
│ ├── fraudTrain.csv 
│ └── fraudTest.csv 
└── notebooks/
  └── data_exploration.ipynb 🔍
  ```


## Files and Their Superpowers 💫
- **main.py**: The main entry point of the application. It initializes the user interface and orchestrates the workflow of the project.

- **src/**: This directory contains the core functionalities of the project.
  - **data_preprocessing.py**: Handles data cleaning, missing value treatment, and data preprocessing steps such as encoding and scaling.
  - **feature_engineering.py**: Responsible for selecting features for the model training phase.
  - **model_training.py**: Contains functions for splitting the data, preparing it for training, and training the logistic regression model.
  - **model_evaluation.py**: Evaluates the trained model using confusion matrix and classification report.

- **util/**: Contains utility functions for connecting to Azure and managing data storage.
  - **azure_connection.py**: Functions to connect to Azure services and handle data operations.

- **data/**: This folder contains the datasets used in the project.
  - **combine.csv**: A combined dataset for training and testing.
  - **fraudTrain.csv**: The training dataset containing labeled transactions.
  - **fraudTest.csv**: The testing dataset for model evaluation.

- **notebooks/**: Contains Jupyter notebooks for data exploration and analysis.
  - **data_exploration.ipynb**: Notebook for exploring the dataset, visualizing data distributions, and understanding relationships between features.

## Getting Started 🚀

1. **Setup**: Ensure you have the required libraries installed. You can install them using pip:

   ```bash
   pip install -r requirements.txt
    ```
2. **Run the Application**: Execute the main.py file to start the application:
    ```bash
   python main.py
    ```
3. **Data Preprocessing**: Use the buttons in the application to perform data preprocessing tasks such as finding missing values, preprocessing data, and balancing classes.
4. **Feature Engineering**: Select features for model training and visualize correlations.
5. **Model Training**: Split the data, prepare it for training, and train the logistic regression model.
6. **Model Evaluation**: Validate the model on training and validation datasets, and visualize the confusion matrix.