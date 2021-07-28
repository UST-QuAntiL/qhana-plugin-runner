import logging
from typing import Any

from mysql.connector import MySQLConnection
from configparser import ConfigParser

from sqlalchemy import create_engine, table, column, select, func
from sqlalchemy.engine import Engine
from sqlalchemy.ext.automap import automap_base, AutomapBase
from sqlalchemy.orm import Session

from plugins.costume_loader_pkg.backend.singleton import Singleton
import os


class Database(Singleton):
    """
    Represents the database class for db connection.
    """

    config_file_default = "config.ini"
    """
    Specifies the default for the config file
    """

    def __init__(self) -> None:
        """
        Initializes the database singleton.
        """
        self.__connection = None
        self.__cursor = None
        self.connected = False
        self.session: Session = None
        self.base: AutomapBase = None
        self.other_tables = {}

        return

    def __del__(self) -> None:
        """
        Deletes the database singleton.
        """
        self.close()
        return

    def open(self, filename=None) -> None:
        """
        Opens the database using the config file.
        """

        # if already connected do nothing
        if self.__connection is not None and self.__connection.is_connected():
            return

        if filename is None:
            filename = Database.config_file_default

        if not os.path.exists(filename):
            logging.error("Couldn't find config file for database connection.")

        section = "mysql"
        parser = ConfigParser()
        parser.read(filename)
        connection_string = {}

        if parser.has_section(section):
            items = parser.items(section)
            for item in items:
                connection_string[item[0]] = item[1]
        else:
            logging.error("{0} not found in the {1} file".format(section, filename))
            raise Exception("{0} not found in the {1} file".format(section, filename))

        self.databaseName = parser.get(section, "database")
        self.user = parser.get(section, "user")
        self.host = parser.get(section, "host")

        self.__connection = MySQLConnection(**connection_string)

        if self.__connection.is_connected():
            self.connected = True
            logging.debug("Successfully connected to database")

    def open_with_params(self, host: str, user: str, password: str, database: str):
        self.__connection = MySQLConnection(
            host=host, user=user, password=password, database=database
        )

        if self.__connection.is_connected():
            self.connected = True
            logging.debug("Successfully connected to database")

        engine: Engine = create_engine(
            "mysql+mysqlconnector://"
            + user
            + ":"
            + password
            + "@"
            + host
            + "/"
            + database
        )

        Base: AutomapBase = automap_base()
        Base.prepare(engine, reflect=True)
        self.base = Base

        self.session = Session(engine)

        self.other_tables["FilmFarbkonzept"] = table(
            "FilmFarbkonzept", column("FilmID"), column("Farbkonzept")
        )
        self.other_tables["RolleDominanteCharaktereigenschaft"] = table(
            "RolleDominanteCharaktereigenschaft",
            column("RollenID"),
            column("FilmID"),
            column("DominanteCharaktereigenschaft"),
        )
        self.other_tables["FilmGenre"] = table(
            "FilmGenre", column("FilmID"), column("Genre")
        )
        self.other_tables["KostuemTageszeit"] = table(
            "KostuemTageszeit",
            column("KostuemID"),
            column("RollenID"),
            column("FilmID"),
            column("Tageszeit"),
        )
        self.other_tables["KostuemKoerpermodifikation"] = table(
            "KostuemKoerpermodifikation",
            column("KostuemID"),
            column("RollenID"),
            column("FilmID"),
            column("Koerpermodifikationname"),
        )
        self.other_tables["KostuemCharaktereigenschaft"] = table(
            "KostuemCharaktereigenschaft",
            column("KostuemID"),
            column("RollenID"),
            column("FilmID"),
            column("Charaktereigenschaft"),
        )
        self.other_tables["KostuemBasiselement"] = table(
            "KostuemBasiselement",
            column("KostuemID"),
            column("RollenID"),
            column("FilmID"),
            column("BasiselementID"),
        )
        self.other_tables["BasiselementDesign"] = table(
            "BasiselementDesign", column("BasiselementID"), column("Designname")
        )
        self.other_tables["BasiselementForm"] = table(
            "BasiselementForm", column("BasiselementID"), column("Formname")
        )
        self.other_tables["BasiselementTrageweise"] = table(
            "BasiselementTrageweise", column("BasiselementID"), column("Trageweisename")
        )
        self.other_tables["BasiselementZustand"] = table(
            "BasiselementZustand", column("BasiselementID"), column("Zustandsname")
        )
        self.other_tables["BasiselementFunktion"] = table(
            "BasiselementFunktion", column("BasiselementID"), column("Funktionsname")
        )

    def close(self) -> None:
        """
        Closes the database connection.
        """
        if self.__connection is not None and self.__connection.is_connected():
            self.__connection.close()
            self.__cursor = None
            self.connected = False
            logging.debug("Successfully disconnected from database")

    def get_cursor(self):
        """
        Returns a cursor to the database.
        """
        return self.__connection.cursor()

    def get_group_column_cte(
        self, table, group_column: str, table_type: str, separator="|"
    ):
        table_columns: Any
        if isinstance(table, self.base):
            table_columns = table
        else:
            table_columns = table.c
        if table_type not in ("film", "role", "costume"):
            raise ValueError("Wrong table type.")
        group_by_rows = [getattr(table_columns, "FilmID")]
        if table_type in ("role", "costume"):
            group_by_rows.append(getattr(table_columns, "RollenID"))
        if table_type == "costume":
            group_by_rows.append(getattr(table_columns, "KostuemID"))
        return (
            select(
                *group_by_rows,
                func.group_concat(
                    getattr(table_columns, group_column),
                    separator,
                ).label(group_column),
            )
            .group_by(*group_by_rows)
            .cte(f"{table.name}CTE")
        )

    def get_film_cte(self, table, group_column, separator="|"):
        return self.get_group_column_cte(table, group_column, "film", separator=separator)

    def get_role_cte(self, table, group_column, separator="|"):
        return self.get_group_column_cte(table, group_column, "role", separator=separator)

    def get_costume_cte(self, table, group_column, separator="|"):
        return self.get_group_column_cte(
            table, group_column, "costume", separator=separator
        )
