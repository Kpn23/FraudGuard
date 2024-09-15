import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv


def load_env_variables():
    load_dotenv()
    return os.getenv("your_password")


def map_dtype_to_sql(dtype):
    """Map pandas dtypes to SQL data types."""
    if pd.api.types.is_integer_dtype(dtype):
        return "INT"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_string_dtype(dtype):
        return "VARCHAR(255)"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"
    else:
        return "VARCHAR(255)"


def connect_to_azure():
    password = load_env_variables()
    server = "mysqlserverkpn23.database.windows.net"
    database = "mySampleDatabase"
    username = "azureuser"
    driver = "{ODBC Driver 18 for SQL Server}"

    connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    conn = pyodbc.connect(connection_string)
    return conn


def create_table(df_fraud, df_not_fraud, conn, table_name):
    cursor = conn.cursor()
    df_fraud.loc[:, "is_fraud"] = df_fraud["is_fraud"].astype(int)
    df_not_fraud.loc[:, "is_fraud"] = df_not_fraud["is_fraud"].astype(int)
    # Create the CREATE TABLE statement dynamically for df_fraud
    columns_fraud = ", ".join(
        [f"{col} {map_dtype_to_sql(df_fraud[col].dtype)}" for col in df_fraud.columns]
    )
    create_table_query_fraud = f"CREATE TABLE {table_name}_fraud ({columns_fraud})"

    # Check if the table exists before creating it
    cursor.execute(
        f"SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}_fraud'"
    )
    if not cursor.fetchone():
        cursor.execute(create_table_query_fraud)

    # Create the CREATE TABLE statement dynamically for df_not_fraud
    columns_not_fraud = ", ".join(
        [
            f"{col} {map_dtype_to_sql(df_not_fraud[col].dtype)}"
            for col in df_not_fraud.columns
        ]
    )
    create_table_query_not_fraud = (
        f"CREATE TABLE {table_name}_not_fraud ({columns_not_fraud})"
    )

    # Check if the table exists before creating it
    cursor.execute(
        f"SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}_not_fraud'"
    )
    if not cursor.fetchone():
        cursor.execute(create_table_query_not_fraud)

    # Insert data into the fraud table
    # Insert data into the fraud table

    for index, row in df_fraud.iterrows():
        values = ", ".join(
            [
                str(value) if isinstance(value, (int, float)) else f"'{str(value)}'"
                for value in row
            ]
        )
        insert_query = f"INSERT INTO {table_name}_fraud VALUES ({values})"
        cursor.execute(insert_query)

    # Insert data into the not fraud table
    for index, row in df_not_fraud.iterrows():
        values = ", ".join(
            [
                str(value) if isinstance(value, (int, float)) else f"'{str(value)}'"
                for value in row
            ]
        )
        insert_query = f"INSERT INTO {table_name}_not_fraud VALUES ({values})"
        cursor.execute(insert_query)

    conn.commit()
    cursor.close()


if __name__ == "__main__":
    # Sample DataFrames for testing
    df_fraud = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
    df_not_fraud = pd.DataFrame({"Name": ["Charlie", "David"], "Age": [35, 40]})

    conn = connect_to_azure()
    create_table(df_fraud, df_not_fraud, conn, "trial")
    conn.close()


def select_table_for_diff_model(df):
    df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)


def detect_data_inside(conn):
    # Step 4: Create a cursor
    cursor = conn.cursor()

    # Step 5: Retrieve and print the list of tables
    print("List of tables in the database:")
    if not cursor.tables():
        print("Table does not exist")
    else:
        for table in cursor.tables(tableType="TABLE"):
            print(table.table_name)
    cursor.close()
    conn.close()


def insert_into_table(conn, table_name, data):
    # Prepare the column names and values for the SQL statement
    columns = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))  # Create placeholders for each value
    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    # Execute the insert statement
    cursor = conn.cursor()
    cursor.execute(sql, tuple(data.values()))  # Pass values as a tuple
    conn.commit()  # Commit the transaction
