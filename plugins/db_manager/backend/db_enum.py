from enum import Enum
from collections import OrderedDict
from sqlalchemy import URL
from .db_manager import DBManager

from celery.utils.log import get_task_logger


TASK_LOGGER = get_task_logger(__name__)


class DBEnum(Enum):
    auto = "Auto"
    sqlite = "SQLite"
    mysql = "MySQL"
    postgresql = "PostgreSQL"

    @staticmethod
    def _get_db_dialect_drivers() -> dict:
        # Keep SQLite as the last entry, since it likes to connect to every database during the guessing phase
        # and throwing an error later!
        return OrderedDict(
            [
                ("mysql", "mysql+pymysql"),
                ("postgresql", "postgresql+psycopg2"),
                ("sqlite", "sqlite"),
            ]
        )

    def _get_db_url(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        dialect_drivers: str = None,
    ) -> URL:
        dialect_drivers = (
            DBEnum._get_db_dialect_drivers()[self.name]
            if dialect_drivers is None
            else dialect_drivers
        )
        if dialect_drivers == "sqlite":
            return URL.create(
                dialect_drivers,
                database=database,
            )
        return URL.create(
            dialect_drivers,
            username=user,
            password=password,  # plain (unescaped) text
            host=host,
            port=port,
            database=database,
        )

    def get_connected_db_manager(
        self, host: str, port: int, user: str, password: str, database: str
    ) -> DBManager:
        if self == DBEnum.auto:
            return self._guess_db_manager(host, port, user, password, database)
        manager = DBManager(self._get_db_url(host, port, user, password, database))
        manager.connect()
        return manager

    def _guess_db_manager(
        self, host: str, port: int, user: str, password: str, database: str
    ) -> DBManager:
        for manager_name, dialect_driver in DBEnum._get_db_dialect_drivers().items():
            try:
                manager = DBManager(
                    self._get_db_url(host, port, user, password, database, dialect_driver)
                )
                manager.connect()
                TASK_LOGGER.info(f"Connected using {manager_name}")
                return manager
            except:
                continue
        raise NotImplementedError(
            "The database type (e.g. MySQL, SQLite, ...) could not be identified or is not implemented or "
            "the given information contains an error/is incomplete!"
        )
