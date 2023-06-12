from abc import ABCMeta
from sqlalchemy import URL, create_engine, text
import pandas as pd


class DBManager(metaclass=ABCMeta):
    def __init__(self, db_url: URL):
        self.connected = False
        self.connection = None
        self.engine = create_engine(db_url)

    def connect(self):
        if self.connected:
            self.connection.close()
        self.connection = self.engine.connect()
        self.connected = True

    def disconnect(self):
        if self.connected:
            self.connection.close()
        self.connection = None
        self.connected = False

    def get_query_as_dataframe(self, query: str) -> pd.DataFrame:
        df = None
        if self.connected:
            df = pd.read_sql(text(query), con=self.connection)
        return df

    def execute_query(self, query: str):
        if self.connected:
            self.connection.execute(text(query))
