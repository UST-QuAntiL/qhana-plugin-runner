from typing import Optional
from sqlalchemy import URL, create_engine, text, MetaData
import pandas as pd


class DBManager:
    def __init__(self, db_url: URL):
        self.connected = False
        self.connection = None
        self.engine = create_engine(db_url)
        self.metadata: Optional[MetaData] = None

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

    def _reflect(self):
        self.metadata = MetaData()
        self.metadata.reflect(self.engine)

    def get_tables_and_columns(self):
        if self.metadata is None:
            self._reflect()
        return {
            table_name: [column.name for column in table.columns]
            for table_name, table in self.metadata.tables.items()
        }

    def get_query_as_dataframe(self, query: str) -> pd.DataFrame:
        df = None
        if self.connected:
            df = pd.read_sql(text(query), con=self.connection)
        return df

    def execute_query(self, query: str):
        if self.connected:
            self.connection.execute(text(query))
