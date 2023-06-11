from enum import Enum
from .db_manager import DBManager, SQLiteManager, MySQLManager, PostgreSQLManager


class DBEnum(Enum):
    auto = "Auto"
    sqlite = "SQLite"
    mysql = "MySQL"
    postgresql = "PostgreSQL"

    def _get_db_managers(self) -> dict:
        # Keep SQLite as the last entry, since it likes to connect to every database during the guessing phase
        # and throwing an error later!
        return {
            "postgresql": PostgreSQLManager,
            "mysql": MySQLManager,
            "sqlite": SQLiteManager,
        }

    def get_connected_db_manager(
        self, host: str, port: int, user: str, password: str, database: str
    ) -> DBManager:
        if self == DBEnum.auto:
            return self._guess_db_manager(host, port, user, password, database)
        manager: DBManager = self._get_db_managers()[str(self.name)]()
        manager.connect(host, port, user, password, database)
        return manager

    def _guess_db_manager(
        self, host: str, port: int, user: str, password: str, database: str
    ) -> DBManager:
        for manager_name, manager_class in self._get_db_managers().items():
            try:
                manager: DBManager = manager_class()
                manager.connect(host, port, user, password, database)
                print(f"Connected using {manager_name}")
                return manager
            except:
                continue
        raise NotImplementedError(
            "The database type (e.g. MySQL, SQLite, ...) could not be identified or is not implemented or "
            "the given information contains an error/is incomplete!"
        )
