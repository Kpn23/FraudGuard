import customtkinter as ctk
import os
from dotenv import load_dotenv
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.data_preprocessing import find_missing_value, balance_class, preprocess_data
from util.azure_connection import (
    connect_to_azure,
    create_table,
    select_table_for_diff_model,
)
from src.feature_engineering import feature_engineering
from src.model_training import split_data, concat_data, prepare_data, train_model
from src.model_evaluation import evaluate_model


class MainSystem(ctk.CTk):
    def load_combined_data(self):
        """Load the combined dataset from CSV."""
        return pd.read_csv(f"{self.csv_directory}/combine.csv", index_col=0)

    def __init__(self):
        load_dotenv()
        self.csv_directory = os.getenv("your_csv_directory")

        self.df_combined = self.load_combined_data()

        # Initialize the parent class
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("FraudGuard on Credit Card Transaction")
        self.geometry("800x600")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(expand=True, fill="both")

        # Create tabs
        self.preprocessing = self.tabview.add("Data Preprocessing")
        # self.cloud = self.tabview.add("Data Management")
        self.feature_engineering = self.tabview.add("Feature Engineering")
        self.training = self.tabview.add("Model Training")
        self.validation = self.tabview.add("Model Validation")
        self.testing = self.tabview.add("Model Testing")

        self.create_element_preprocessing(self.preprocessing)
        # self.create_element_cloud(self.cloud)
        self.create_element_feature_engineering(self.feature_engineering)
        self.create_element_training(self.training)
        self.create_element_validation(self.validation)
        self.create_element_testing(self.testing)

    def create_element_preprocessing(self, parent):
        # Create a frame for the buttons
        button_frame = ctk.CTkFrame(parent)
        button_frame.grid(row=1, column=0, sticky="nsew")

        # Create a frame for the figure
        figure_frame = ctk.CTkFrame(parent)
        figure_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # Create a frame for the text widget
        text_frame = ctk.CTkFrame(parent)
        text_frame.grid(row=1, column=1, sticky="nsew")
        # Optionally set a minimum size for the text_frame
        text_frame.grid_propagate(False)
        text_frame.configure(height=100)

        # Configure grid weights to make frames resize properly
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=0)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # Create element for each task
        tasks = [
            "Find Missing Value",
            "Preprocess Data",
            "Balance Class ('is_fraud')",
        ]
        functions = [
            self.preprocessing_button1,
            self.preprocessing_button2,
            self.preprocessing_button3,
        ]
        # Add element to the parent frame
        for index, task in enumerate(tasks):
            button = ctk.CTkButton(button_frame, text=task, command=functions[index])
            button.pack(pady=10, padx=20, fill="x")

        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Text widget
        self.preprocessing_text = ScrolledText(text_frame, height=10, width=50)
        self.preprocessing_text.pack(pady=10)

    def create_element_cloud(self, parent):
        # Create element for each task
        tasks = [
            "Upload to Azure Cloud",
            "Download from cloud",
        ]
        functions = [self.cloud_button1, self.cloud_button2]
        # Add element to the parent frame
        for index, task in enumerate(tasks):
            button = ctk.CTkButton(parent, text=task, command=functions[index])
            button.pack(pady=10, padx=20, fill="x")

    def create_element_feature_engineering(self, parent):
        # Create a frame for the buttons
        button_frame3 = ctk.CTkFrame(parent)
        button_frame3.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Create a frame for the figure
        figure_frame3 = ctk.CTkFrame(parent)
        figure_frame3.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # Create a frame for the text widget
        checkbox_frame3 = ctk.CTkScrollableFrame(parent)
        checkbox_frame3.grid(
            row=1, column=1, padx=10, pady=10, sticky="nsew"
        )  # Bottom left

        # Configure grid weights to make frames resize properly
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=0)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # Create element for each task
        tasks = ["Feature Engineering (Allow Selection)", "Plot correation heatmap"]
        functions = [self.feature_engineering_button1, self.feature_engineering_button2]
        # Add element to the parent frame
        for index, task in enumerate(tasks):
            button = ctk.CTkButton(button_frame3, text=task, command=functions[index])
            button.pack(pady=10, padx=20, fill="x")

        self.select_all_var = ctk.StringVar(value="off")
        self.select_all_checkbox = ctk.CTkCheckBox(
            master=checkbox_frame3,
            text="Select All",
            variable=self.select_all_var,
            command=self.toggle_all,
            onvalue="on",
            offvalue="off",
        )
        self.select_all_checkbox.pack(pady=10)

        # store the created checkbox name
        self.check_vars = []
        self.target = "is_fraud"

        # Get the feature names from the DataFrame and remove index and target
        self.feature_header = self.df_combined.columns.tolist()
        for column in ["Unnamed: 0", self.target]:
            if column in self.feature_header:
                self.feature_header.remove(column)
        self.feature_header[self.feature_header.index("dob")] = "age"
        for feature in self.feature_header:
            var = ctk.StringVar()
            self.check_vars.append(var)
            features_selection = ctk.CTkCheckBox(
                master=checkbox_frame3,
                text=feature,
                variable=var,
                onvalue="on",
                offvalue="off",
            )
            features_selection.pack(anchor="w", pady=5)

        self.figure3 = plt.Figure(figsize=(4, 4), dpi=100)
        self.canvas3 = FigureCanvasTkAgg(self.figure3, master=figure_frame3)
        self.canvas3.get_tk_widget().pack(fill="both", expand=True)

    def create_element_training(self, parent):
        # Create element for each task
        tasks = [
            "Split Data",
            "Combine Data",
            "Prepare x/y data",
            "Train Model",
        ]

        functions = [
            self.training_button1,
            self.training_button2,
            self.training_button3,
            self.training_button4,
        ]
        # Add element to the parent frame
        for index, task in enumerate(tasks):
            button = ctk.CTkButton(parent, text=task, command=functions[index])
            button.pack(pady=10, padx=20, fill="x")

    def create_element_validation(self, parent):
        # Create a frame for the buttons
        button_frame5 = ctk.CTkFrame(parent)
        button_frame5.grid(row=1, column=0, sticky="nsew")

        # Create a frame for the figure
        figure_frame5 = ctk.CTkFrame(parent)
        figure_frame5.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # Create a frame for the text widget
        text_frame5 = ctk.CTkFrame(parent)
        text_frame5.grid(row=1, column=1, sticky="nsew")
        # Optionally set a minimum size for the text_frame
        text_frame5.grid_propagate(False)
        text_frame5.configure(height=100)

        # Configure grid weights to make frames resize properly
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=0)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # Create element for each task
        tasks = ["Validate Model on Training data", "Validate Model on Validation data"]
        functions = [self.validation_button1, self.validation_button2]
        # Add element to the parent frame
        for index, task in enumerate(tasks):
            button = ctk.CTkButton(button_frame5, text=task, command=functions[index])
            button.pack(pady=10, padx=20, fill="x")

        self.figure5 = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas5 = FigureCanvasTkAgg(self.figure5, master=figure_frame5)
        self.canvas5.get_tk_widget().pack(fill="both", expand=True)

        # Text widget
        self.validation_text = ScrolledText(text_frame5, height=10, width=70)
        self.validation_text.pack(pady=10)

    def create_element_testing(self, parent):
        # Create a frame for the buttons
        button_frame6 = ctk.CTkFrame(parent)
        button_frame6.grid(row=1, column=0, sticky="nsew")

        # Create a frame for the figure
        figure_frame6 = ctk.CTkFrame(parent)
        figure_frame6.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # Create a frame for the text widget
        text_frame6 = ctk.CTkFrame(parent)
        text_frame6.grid(row=1, column=1, sticky="nsew")
        # Optionally set a minimum size for the text_frame
        text_frame6.grid_propagate(False)
        text_frame6.configure(height=100)

        # Configure grid weights to make frames resize properly
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=0)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # Create element for each task
        tasks = ["Final Evaluation"]
        functions = [self.testing_button1]
        # Add element to the parent frame
        for index, task in enumerate(tasks):
            button = ctk.CTkButton(button_frame6, text=task, command=functions[index])
            button.pack(pady=10, padx=20, fill="x")

        self.figure6 = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas6 = FigureCanvasTkAgg(self.figure6, master=figure_frame6)
        self.canvas6.get_tk_widget().pack(fill="both", expand=True)

        # Text widget
        self.testing_text = ScrolledText(text_frame6, height=10, width=70)
        self.testing_text.pack(pady=10)

    def preprocessing_button1(self):
        missing_values = find_missing_value(self.df_combined)
        if missing_values.any():
            for index, value in missing_values.items():
                if value:  # Check if the value is True
                    self.preprocessing_text.insert(
                        tk.END, f"Missing value at index {index}: {value}\n"
                    )
            self.preprocessing_text.yview(tk.END)
        else:
            self.preprocessing_text.insert(tk.END, "No missing value found\n")
            self.preprocessing_text.yview(tk.END)

    def preprocessing_button2(self):
        self.df_encoded = preprocess_data(self.df_combined)
        self.preprocessing_text.insert(
            tk.END, "LabelEncoder and StandardScaler are applied!\n"
        )
        self.preprocessing_text.yview(tk.END)

    def preprocessing_button3(self):
        self.df_fraud, self.df_not_fraud = balance_class(self.df_encoded)

        df_balance = pd.concat(
            [self.df_not_fraud, self.df_encoded[self.df_encoded.is_fraud == 1]],
            ignore_index=True,
        )

        ax = self.figure.add_subplot(111)

        df_balance.groupby("is_fraud").count()["cc_num"].plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        ax.set_xlabel("Is_Fraud")
        ax.set_title("Count of Fraudulent vs Non-Fraudulent Transactions")
        ax.set_xticklabels(["negative", "positive"], rotation=0)
        self.canvas.draw()

    # def cloud_button1(self):
    #     create_table(
    #         self.df_fraud, self.df_not_fraud, connect_to_azure(), "LogisticRegression"
    #     )

    # def cloud_button2(self):
    #     self.cloud

    def feature_engineering_button1(self):
        features_list = [
            feature
            for feature, var in zip(self.feature_header, self.check_vars)
            if var.get() == "on"
        ]
        self.df_fraud_FT, self.df_not_fraud_FT = feature_engineering(
            self.df_fraud, self.df_not_fraud, features_list, self.target
        )

    def feature_engineering_button2(self):
        df_temp_fraud = self.df_fraud_FT
        df_temp_not_fraud = self.df_not_fraud_FT
        df_temp_fraud = df_temp_fraud.drop(self.target, axis=1)
        df_temp_not_fraud = df_temp_not_fraud.drop(self.target, axis=1)
        self.df_heatmap = pd.concat(
            [df_temp_fraud, df_temp_not_fraud], ignore_index=True
        )

        self.figure3.clf()

        ax = self.figure3.add_subplot(111)

        correlation_matrix = self.df_heatmap.corr()

        sns.heatmap(
            correlation_matrix,
            annot=False,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title("Correlation between Features")
        ax.tick_params(axis="x", labelsize=6, rotation=45)
        ax.tick_params(axis="y", labelsize=6)
        self.canvas3.draw()

    def toggle_all(self):
        if self.select_all_var.get() == "on":
            for var in self.check_vars:
                var.set("on")
        else:
            for var in self.check_vars:
                var.set("off")

    def training_button1(self):
        (
            self.train_fraud_FT,
            self.test_fraud_FT,
            self.valid_fraud_FT,
            self.train_not_fraud_FT,
            self.test_not_fraud_FT,
            self.valid_not_fraud_FT,
        ) = split_data(self.df_fraud_FT, self.df_not_fraud_FT)

    def training_button2(self):
        self.train_concat = concat_data(self.train_fraud_FT, self.train_not_fraud_FT)
        self.test_concat = concat_data(self.test_fraud_FT, self.test_not_fraud_FT)
        self.valid_concat = concat_data(self.valid_fraud_FT, self.valid_not_fraud_FT)

    def training_button3(self):
        self.x_train, self.y_train = prepare_data(self.train_concat)
        self.x_test, self.y_test = prepare_data(self.test_concat)
        self.x_valid, self.y_valid = prepare_data(self.valid_concat)

    def training_button4(self):
        self.logistic_regression_model = train_model(self.x_train, self.y_train)

    def validation_button1(self):
        tn, fp, fn, tp, report = evaluate_model(
            self.logistic_regression_model, self.x_train, self.y_train
        )

        cm = np.array([[tn, fp], [fn, tp]])

        self.figure5.clf()

        ax = self.figure5.add_subplot(111)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax,
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix on Training Data")

        self.canvas5.draw()

        self.validation_text.insert(tk.END, report)
        self.validation_text.yview(tk.END)

    def validation_button2(self):
        tn, fp, fn, tp, report = evaluate_model(
            self.logistic_regression_model, self.x_valid, self.y_valid
        )
        cm = np.array([[tn, fp], [fn, tp]])

        self.figure5.clf()

        ax = self.figure5.add_subplot(111)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax,
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix on Training Data")

        self.canvas5.draw()

        self.validation_text.insert(tk.END, report)
        self.validation_text.yview(tk.END)

    def testing_button1(self):
        tn, fp, fn, tp, report = evaluate_model(
            self.logistic_regression_model, self.x_valid, self.y_valid
        )

        cm = np.array([[tn, fp], [fn, tp]])

        self.figure6.clf()

        # Create a heatmap
        ax = self.figure6.add_subplot(111)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax,
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix on Testing data")
        self.canvas6.draw()

        self.testing_text.insert(tk.END, report)
        self.testing_text.yview(tk.END)


if __name__ == "__main__":
    app = MainSystem()
    app.mainloop()
