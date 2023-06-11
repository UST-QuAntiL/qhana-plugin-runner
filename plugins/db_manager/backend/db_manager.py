from abc import ABCMeta, abstractmethod
from sqlite3 import connect as sqlite_connect
from mysql.connector import connect as mysql_connect
from psycopg2 import connect as psycopg_connect


class DBManager(metaclass=ABCMeta):
    def __init__(self):
        self.connected = False
        self.connection = None

    @abstractmethod
    def _connect(self, host: str, port: int, user: str, password: str, database: str):
        raise NotImplementedError("This function is not implemented!")

    def connect(self, host: str, port: int, user: str, password: str, database: str):
        if self.connected:
            self.connection.close()
        self.connection = self._connect(host, port, user, password, database)
        self.connected = True

    def disconnect(self):
        if self.connected:
            self.connection.close()
        self.connected = False

    def execute_query(self, query: str):
        cursor = self.connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        self.connection.commit()
        return results


class SQLiteManager(DBManager):
    def __init__(self):
        super().__init__()

    def _connect(self, host: str, port: int, user: str, password: str, database: str):
        return sqlite_connect(database)


class MySQLManager(DBManager):
    def __init__(self):
        super().__init__()

    def _connect(self, host: str, port: int, user: str, password: str, database: str):
        return mysql_connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )


class PostgreSQLManager(DBManager):
    def __init__(self):
        super().__init__()

    def _connect(self, host: str, port: int, user: str, password: str, database: str):
        return psycopg_connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
        )
