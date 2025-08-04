import pymongo
from insurance.constants.database_variables import DATABASE_NAME
from insurance.constants.env_variable import MONGODB_URL_KEY
from insurance.logger import logging

import certifi
import os
ca = certifi.where()

class MongoDBClient:
    client = None
    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:

            mongo_db_url = os.getenv(MONGODB_URL_KEY)
            if mongo_db_url is not None and "localhost" in mongo_db_url:
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url) 
            else:
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            
            logging.info(f"MongoDB Client created for database: {database_name}")
            
        except Exception as e:
            raise e


# if __name__ == "__main__":
#     mongo_client = MongoDBClient()
#     print(mongo_client.client)