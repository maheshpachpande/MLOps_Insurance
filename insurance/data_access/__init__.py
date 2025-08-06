import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import json

from dotenv import load_dotenv
from insurance.configuration.mongodb_connection import MongoDBClient
from insurance.constants.database_variables import DATABASE_NAME, COLLECTION_NAME
from insurance.exception import CustomException

load_dotenv()  # Load MONGO_DB_URL from .env


class InsuranceData:
    """
    This class helps to export entire MongoDB collection as a pandas DataFrame
    and to insert CSV data into MongoDB.
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise CustomException(e, sys)

    def save_csv_file(self, file_path: str, collection_name: str, database_name: Optional[str] = None) -> int:
        """
        Load CSV file and insert it into MongoDB collection.
        """
        try:
            # Load CSV as DataFrame
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)

            # Convert to list of dictionaries
            records = list(json.loads(df.T.to_json()).values())

            # Get collection reference
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.database[collection_name]

            # Insert records
            if records:
                collection.insert_many(records)

            return len(records)

        except Exception as e:
            raise CustomException(e,sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Export entire MongoDB collection as a pandas DataFrame.
        """
        try:
            # Get collection reference
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.database[collection_name]

            # Convert to DataFrame
            df = pd.DataFrame(list(collection.find()))

            # Drop MongoDB _id column
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            # Replace "na" with np.nan
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise CustomException(e,sys)


# Run this as a script
if __name__ == "__main__":
    try:
        visa = InsuranceData()
        df = visa.export_collection_as_dataframe(database_name=DATABASE_NAME, collection_name=COLLECTION_NAME)
        print(df.head())  # Preview data
    except Exception as e:
        raise CustomException(e,sys)
