from typing import Any, Union, List

from sqlalchemy import create_engine, table, column, select, func
from sqlalchemy.engine import Engine
from sqlalchemy.ext.automap import automap_base, AutomapBase
from sqlalchemy.orm import Session, DeclarativeMeta

from plugins.costume_loader_pkg.backend.singleton import Singleton


class Database(Singleton):
    """
    Represents the database class for db connection.
    """

    def __init__(self):
        """
        Initializes the database singleton.
        """
        self.connected = False
        self.session: Session = None
        self.base: AutomapBase = None
        self.other_tables = {}

    def open_with_params(self, host: str, user: str, password: str, database: str):
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

        self.connected = True

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

    def _get_group_column_cte(
        self,
        table,
        group_columns: Union[str, List[str]],
        table_type: str,
        name_suffix: str = "",
    ):
        table_columns: Any
        if isinstance(table, DeclarativeMeta):
            table_columns = table
        else:
            table_columns = table.c
        if table_type not in ("film", "role", "costume", "base_element"):
            raise ValueError("Wrong table type.")

        group_by_rows = []

        if table_type in ["film", "role", "costume"]:
            group_by_rows = [getattr(table_columns, "FilmID")]

            if table_type in ("role", "costume"):
                group_by_rows.append(getattr(table_columns, "RollenID"))
            if table_type == "costume":
                group_by_rows.append(getattr(table_columns, "KostuemID"))

        if table_type == "base_element":
            group_by_rows = [getattr(table_columns, "BasiselementID")]

        name = (
            getattr(table, "name", getattr(table, "__name__", ""))
            + "_"
            + name_suffix
            + "_CTE"
        )

        concats = []

        if isinstance(group_columns, str):
            concats.append(
                func.group_concat(getattr(table_columns, group_columns)).label(
                    group_columns
                )
            )
        elif isinstance(group_columns, list):
            for group_column in group_columns:
                concats.append(
                    func.group_concat(getattr(table_columns, group_column)).label(
                        group_column
                    )
                )

        return select(*group_by_rows, *concats).group_by(*group_by_rows).cte(name)

    def get_film_cte(
        self, table, group_columns: Union[str, List[str]], name_suffix: str = ""
    ):
        return self._get_group_column_cte(table, group_columns, "film", name_suffix)

    def get_role_cte(
        self, table, group_columns: Union[str, List[str]], name_suffix: str = ""
    ):
        return self._get_group_column_cte(table, group_columns, "role", name_suffix)

    def get_costume_cte(
        self, table, group_columns: Union[str, List[str]], name_suffix: str = ""
    ):
        return self._get_group_column_cte(table, group_columns, "costume", name_suffix)

    def get_base_element_cte(
        self, table, group_columns: Union[str, List[str]], name_suffix: str = ""
    ):
        return self._get_group_column_cte(
            table, group_columns, "base_element", name_suffix
        )
