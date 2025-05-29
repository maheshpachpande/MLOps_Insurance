import os
import sys
from src.insurance_fraud_detection.exception import CustomException
from src.insurance_fraud_detection.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle



load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv('db')

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established")
        # Read data from the 'students' table
        df=pd.read_sql_query('Select * from insurancefraud_dataset',mydb)
        print(df.head())

        return df

    except Exception as e:
        raise CustomException(e, sys)