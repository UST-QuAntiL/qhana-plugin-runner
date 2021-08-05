import logging
import time
from enum import Enum, auto
from typing import Any, Dict, Tuple
from typing import List

from sqlalchemy import select, and_, join, func
from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy.sql import expression

from plugins.costume_loader_pkg.backend.attribute import Attribute
from plugins.costume_loader_pkg.backend.database import Database

MUSE_URL = "http://129.69.214.108/"


class Entity:
    """
    This class represents a entity such as a costume or a basic element.
    """

    def __init__(self, name: str):
        """
        Initializes the entity object.
        """
        self.name = name
        self.id = 0
        self.basiselementID = 0
        self.kostuemId = 0
        self.rollenId = 0
        self.filmId = 0
        # format: (attribute, None)
        self.attributes = {}
        # format: (attribute, value)
        self.values = {}

    def get_film_url(self) -> str:
        """
        Gets the film url of the entity.
        """
        return MUSE_URL + "#/filme/" + str(self.id)

    def get_rollen_url(self) -> str:
        """
        Gets the rollen url of the entity.
        """
        return self.get_film_url() + "/rollen/" + str(self.rollenId)

    def get_kostuem_url(self) -> str:
        """
        Gets the kostuem url of the entity.
        """
        return self.get_rollen_url() + "/kostueme/" + str(self.kostuemId)

    def add_attribute(self, attribute: Attribute):
        """
        Adds an attribute to the attributes list.
        """
        self.attributes[attribute] = None

    def add_value(self, attribute: Attribute, value: Any):
        """
        Adds a value to the values list. If the associated
        attribute is not added yet, it will be added.
        """
        if attribute not in self.attributes:
            self.add_attribute(attribute)
        self.values[attribute] = value

    def get_value(self, attribute: Attribute):
        """
        Gets the value of the given attribute.
        """
        return self.values[attribute]

    def remove_attribute(self, attribute: Attribute):
        """
        Removes an attribute from the attributes list.
        """
        del self.attributes[attribute]

    def remove_value(self, attribute: Attribute):
        """
        Removes a value from the values list.
        """
        del self.values[attribute]

    def __str__(self) -> str:
        """
        Returns the entity in a single string.
        """
        output = "name: " + self.name + ", id: " + str(self.id) + ", "
        output += " basiselementID: " + str(self.basiselementID) + ", "
        output += " kostuemId: " + str(self.kostuemId) + ", "
        output += " rollenId: " + str(self.rollenId) + ", "
        output += " filmId: " + str(self.filmId) + ", "
        for attribute_key in self.values:
            output += (
                Attribute.get_name(attribute_key)
                + ": "
                + str(self.values[attribute_key])
                + ", "
            )
        return output[:-2]


class TableType(Enum):
    FILM = 0
    ROLE = 1
    COSTUME = 2
    BASE_ELEMENT = 3


class EntityFactory:
    """
    This class creates entities based on the given attributes.
    """

    @staticmethod
    def _create_and_expression(
        costume_table: Any,
        costume_base_element_table: Any,
        table_to_join: Any,
        table_type: TableType,
    ) -> expression:
        if isinstance(table_to_join, DeclarativeMeta):
            table_columns = table_to_join
        else:
            table_columns = table_to_join.c

        columns_to_join_on = []

        if table_type in [TableType.FILM, TableType.ROLE, TableType.COSTUME]:
            columns_to_join_on.append(costume_table.FilmID == table_columns.FilmID)

            if table_type in [TableType.ROLE, TableType.COSTUME]:
                columns_to_join_on.append(
                    costume_table.RollenID == table_columns.RollenID
                )

                if table_type == TableType.COSTUME:
                    columns_to_join_on.append(
                        costume_table.KostuemID == table_columns.KostuemID
                    )

        if table_type == TableType.BASE_ELEMENT:
            columns_to_join_on = [
                costume_base_element_table.c.BasiselementID
                == table_columns.BasiselementID
            ]

        return and_(*columns_to_join_on)

    @staticmethod
    def _create_table_column_dicts(
        database: Database,
    ) -> Tuple[
        Dict[Attribute, Any], Dict[Attribute, Any], Dict[Attribute, Any], Any, Any
    ]:
        join_on: Dict[Attribute, Any] = {}
        tables: Dict[Attribute, Any] = {}
        columns: Dict[Attribute, Any] = {}

        #################
        # Kostuem table #
        #################

        costume_table = database.base.classes.Kostuem
        join_on[costume_table] = EntityFactory._create_and_expression(
            costume_table, None, costume_table, TableType.COSTUME
        )

        columns[Attribute.ortsbegebenheit] = costume_table.Ortsbegebenheit
        columns[Attribute.dominanteFarbe] = costume_table.DominanteFarbe
        columns[Attribute.stereotypRelevant] = costume_table.StereotypRelevant
        columns[Attribute.dominanteFunktion] = costume_table.DominanteFunktion
        columns[Attribute.dominanterZustand] = costume_table.DominanterZustand

        #########################
        # FilmFarbkonzept table #
        #########################

        color_concept_table = database.get_film_cte(
            database.other_tables["FilmFarbkonzept"], "Farbkonzept"
        )
        join_on[color_concept_table] = EntityFactory._create_and_expression(
            costume_table, None, color_concept_table, TableType.FILM
        )

        tables[Attribute.farbkonzept] = color_concept_table
        columns[Attribute.farbkonzept] = color_concept_table.c.Farbkonzept

        ############################################
        # RolleDominanteCharaktereigenschaft table #
        ############################################

        dominant_character_trait_table = database.get_role_cte(
            database.other_tables["RolleDominanteCharaktereigenschaft"],
            "DominanteCharaktereigenschaft",
        )
        join_on[dominant_character_trait_table] = EntityFactory._create_and_expression(
            costume_table, None, dominant_character_trait_table, TableType.ROLE
        )

        tables[Attribute.dominanteCharaktereigenschaft] = dominant_character_trait_table
        columns[
            Attribute.dominanteCharaktereigenschaft
        ] = dominant_character_trait_table.c.DominanteCharaktereigenschaft

        ###################
        # Stereotyp table #
        ###################

        role_stereotype_table = database.get_role_cte(
            database.base.classes.RolleStereotyp, "Stereotyp"
        )
        join_on[role_stereotype_table] = EntityFactory._create_and_expression(
            costume_table, None, role_stereotype_table, TableType.ROLE
        )

        tables[Attribute.stereotyp] = role_stereotype_table
        columns[Attribute.stereotyp] = role_stereotype_table.c.Stereotyp

        ###############
        # Rolle table #
        ###############

        role_table = database.base.classes.Rolle
        join_on[role_table] = EntityFactory._create_and_expression(
            costume_table, None, role_table, TableType.ROLE
        )

        tables[Attribute.rollenberuf] = role_table
        columns[Attribute.rollenberuf] = role_table.Rollenberuf

        tables[Attribute.geschlecht] = role_table
        columns[Attribute.geschlecht] = role_table.Geschlecht

        tables[Attribute.dominanterAlterseindruck] = role_table
        columns[Attribute.dominanterAlterseindruck] = role_table.DominanterAlterseindruck

        tables[Attribute.dominantesAlter] = role_table
        columns[Attribute.dominantesAlter] = role_table.DominantesAlter

        tables[Attribute.rollenrelevanz] = role_table
        columns[Attribute.rollenrelevanz] = role_table.Rollenrelevanz

        ###################
        # FilmGenre table #
        ###################

        genre_table = database.get_film_cte(database.other_tables["FilmGenre"], "Genre")
        join_on[genre_table] = EntityFactory._create_and_expression(
            costume_table, None, genre_table, TableType.FILM
        )

        tables[Attribute.genre] = genre_table
        columns[Attribute.genre] = genre_table.c.Genre

        ###################
        # KostuemSpielzeit table #
        ###################

        costume_playtime_table = database.get_costume_cte(
            database.base.classes.KostuemSpielzeit, "Spielzeit"
        )
        join_on[costume_playtime_table] = EntityFactory._create_and_expression(
            costume_table, None, costume_playtime_table, TableType.COSTUME
        )

        tables[Attribute.spielzeit] = costume_playtime_table
        columns[Attribute.spielzeit] = costume_playtime_table.c.Spielzeit

        ##########################
        # KostuemTageszeit table #
        ##########################

        costume_daytime_table = database.get_costume_cte(
            database.other_tables["KostuemTageszeit"], "Tageszeit"
        )
        join_on[costume_daytime_table] = EntityFactory._create_and_expression(
            costume_table, None, costume_daytime_table, TableType.COSTUME
        )

        tables[Attribute.tageszeit] = costume_daytime_table
        columns[Attribute.tageszeit] = costume_daytime_table.c.Tageszeit

        ####################################
        # KostuemKoerpermodifikation table #
        ####################################

        body_modification_table = database.get_costume_cte(
            database.other_tables["KostuemKoerpermodifikation"], "Koerpermodifikationname"
        )
        join_on[body_modification_table] = EntityFactory._create_and_expression(
            costume_table, None, body_modification_table, TableType.COSTUME
        )

        tables[Attribute.koerpermodifikation] = body_modification_table
        columns[
            Attribute.koerpermodifikation
        ] = body_modification_table.c.Koerpermodifikationname

        #########################
        # KostuemTimecode table #
        #########################

        timecode_table = database.base.classes.KostuemTimecode

        duration_table = (
            select(
                timecode_table.KostuemID,
                timecode_table.RollenID,
                timecode_table.FilmID,
                func.sum(
                    func.time_to_sec(
                        func.timediff(
                            timecode_table.Timecodeende, timecode_table.Timecodeanfang
                        )
                    )
                ).label("Duration"),
            )
            .group_by(
                timecode_table.FilmID, timecode_table.RollenID, timecode_table.KostuemID
            )
            .cte("DurationCTE")
        )
        join_on[duration_table] = EntityFactory._create_and_expression(
            costume_table, None, duration_table, TableType.COSTUME
        )

        tables[Attribute.kostuemZeit] = duration_table
        columns[Attribute.kostuemZeit] = duration_table.c.Duration

        ############################
        # RolleFamilienstand table #
        ############################

        status_table = database.get_role_cte(
            database.base.classes.RolleFamilienstand, "Familienstand"
        )
        join_on[status_table] = EntityFactory._create_and_expression(
            costume_table, None, status_table, TableType.ROLE
        )

        tables[Attribute.familienstand] = status_table
        columns[Attribute.familienstand] = status_table.c.Familienstand

        #####################################
        # KostuemCharaktereigenschaft table #
        #####################################

        trait_table = database.get_costume_cte(
            database.other_tables["KostuemCharaktereigenschaft"], "Charaktereigenschaft"
        )
        join_on[trait_table] = EntityFactory._create_and_expression(
            costume_table, None, trait_table, TableType.COSTUME
        )

        tables[Attribute.charaktereigenschaft] = trait_table
        columns[Attribute.charaktereigenschaft] = trait_table.c.Charaktereigenschaft

        #########################
        # KostuemSpielort table #
        #########################

        location_table = database.get_costume_cte(
            database.base.classes.KostuemSpielort, ["Spielort", "SpielortDetail"]
        )
        join_on[location_table] = EntityFactory._create_and_expression(
            costume_table, None, location_table, TableType.COSTUME
        )

        tables[Attribute.spielort] = location_table
        columns[Attribute.spielort] = location_table.c.Spielort

        tables[Attribute.spielortDetail] = location_table
        columns[Attribute.spielortDetail] = location_table.c.SpielortDetail

        ###############################
        # KostuemAlterseindruck table #
        ###############################

        age_impression_table = database.get_costume_cte(
            database.base.classes.KostuemAlterseindruck, ["Alterseindruck", "NumAlter"]
        )
        join_on[age_impression_table] = EntityFactory._create_and_expression(
            costume_table, None, age_impression_table, TableType.COSTUME
        )

        tables[Attribute.alterseindruck] = age_impression_table
        columns[Attribute.alterseindruck] = age_impression_table.c.Alterseindruck

        tables[Attribute.alter] = age_impression_table
        columns[Attribute.alter] = age_impression_table.c.NumAlter

        #############################
        # KostuemBasiselement table #
        #############################

        costume_base_element_table = database.other_tables["KostuemBasiselement"]
        join_on[costume_base_element_table] = EntityFactory._create_and_expression(
            costume_table, None, costume_base_element_table, TableType.COSTUME
        )

        ######################
        # Basiselement table #
        ######################

        base_element_table = database.base.classes.Basiselement
        join_on[base_element_table] = EntityFactory._create_and_expression(
            costume_table,
            costume_base_element_table,
            base_element_table,
            TableType.BASE_ELEMENT,
        )

        tables[Attribute.basiselement] = base_element_table
        columns[Attribute.basiselement] = base_element_table.Basiselementname

        ############################
        # BasiselementDesign table #
        ############################

        design_table = database.get_base_element_cte(
            database.other_tables["BasiselementDesign"], "Designname"
        )
        join_on[design_table] = EntityFactory._create_and_expression(
            costume_table,
            costume_base_element_table,
            design_table,
            TableType.BASE_ELEMENT,
        )

        tables[Attribute.design] = design_table
        columns[Attribute.design] = design_table.c.Designname

        ##########################
        # BasiselementForm table #
        ##########################

        form_table = database.get_base_element_cte(
            database.other_tables["BasiselementForm"], "Formname"
        )
        join_on[form_table] = EntityFactory._create_and_expression(
            costume_table, costume_base_element_table, form_table, TableType.BASE_ELEMENT
        )

        tables[Attribute.form] = form_table
        columns[Attribute.form] = form_table.c.Formname

        ##########################
        # BasiselementTrageweise #
        ##########################

        wear_table = database.get_base_element_cte(
            database.other_tables["BasiselementTrageweise"], "Trageweisename"
        )
        join_on[wear_table] = EntityFactory._create_and_expression(
            costume_table, costume_base_element_table, wear_table, TableType.BASE_ELEMENT
        )

        tables[Attribute.trageweise] = wear_table
        columns[Attribute.trageweise] = wear_table.c.Trageweisename

        #############################
        # BasiselementZustand table #
        #############################

        condition_table = database.get_base_element_cte(
            database.other_tables["BasiselementZustand"], "Zustandsname"
        )
        join_on[condition_table] = EntityFactory._create_and_expression(
            costume_table,
            costume_base_element_table,
            condition_table,
            TableType.BASE_ELEMENT,
        )

        tables[Attribute.zustand] = condition_table
        columns[Attribute.zustand] = condition_table.c.Zustandsname

        ##############################
        # BasiselementFunktion table #
        ##############################

        function_table = database.get_base_element_cte(
            database.other_tables["BasiselementFunktion"], "Funktionsname"
        )
        join_on[function_table] = EntityFactory._create_and_expression(
            costume_table,
            costume_base_element_table,
            function_table,
            TableType.BASE_ELEMENT,
        )

        tables[Attribute.funktion] = function_table
        columns[Attribute.funktion] = function_table.c.Funktionsname

        ##############################
        # BasiselementMaterial table #
        ##############################

        material_table = database.get_base_element_cte(
            database.base.classes.BasiselementMaterial,
            ["Materialname", "Materialeindruck"],
        )
        join_on[material_table] = EntityFactory._create_and_expression(
            costume_table,
            costume_base_element_table,
            material_table,
            TableType.BASE_ELEMENT,
        )

        tables[Attribute.material] = material_table
        columns[Attribute.material] = material_table.c.Materialname

        tables[Attribute.materialeindruck] = material_table
        columns[Attribute.materialeindruck] = material_table.c.Materialeindruck

        ###########################
        # BasiselementFarbe table #
        ###########################

        color_table = database.get_base_element_cte(
            database.base.classes.BasiselementFarbe, ["Farbname", "Farbeindruck"]
        )
        join_on[color_table] = EntityFactory._create_and_expression(
            costume_table,
            costume_base_element_table,
            color_table,
            TableType.BASE_ELEMENT,
        )

        tables[Attribute.farbe] = color_table
        columns[Attribute.farbe] = color_table.c.Farbname

        tables[Attribute.farbeindruck] = color_table
        columns[Attribute.farbeindruck] = color_table.c.Farbeindruck

        return tables, columns, join_on, costume_table, costume_base_element_table

    @staticmethod
    def create(attributes: List[Attribute], database: Database) -> List[Entity]:
        """
        Creates a entity based on the given list of attributes.
        """
        columns_to_select = []
        tables_to_join = set()
        (
            tables,
            columns,
            join_on,
            costume_table,
            costume_base_element_table,
        ) = EntityFactory._create_table_column_dicts(database)

        is_base_element = False

        for attr in attributes:
            columns_to_select.append(columns[attr])

            if attr in tables:
                tables_to_join.add(tables[attr])

            is_base_element = is_base_element or attr in [
                Attribute.basiselement,
                Attribute.design,
                Attribute.form,
                Attribute.trageweise,
                Attribute.zustand,
                Attribute.funktion,
                Attribute.material,
                Attribute.materialeindruck,
                Attribute.farbe,
                Attribute.farbeindruck,
            ]

        # add base element id if base elements need to be loaded
        if is_base_element:
            columns_to_select = [
                costume_base_element_table.c.BasiselementID
            ] + columns_to_select

        columns_to_select = [
            costume_table.KostuemID,
            costume_table.RollenID,
            costume_table.FilmID,
        ] + columns_to_select

        query = select(*columns_to_select)

        if is_base_element:
            j = join(
                costume_table,
                costume_base_element_table,
                join_on[costume_base_element_table],
            )
        else:
            j = None

        for table in tables_to_join:
            if j is None:
                j = join(costume_table, table, join_on[table])
            else:
                j = j.join(table, join_on[table])

        if j is not None:
            query = query.select_from(j)

        query = query.order_by(
            costume_table.FilmID,
            costume_table.RollenID,
            costume_table.KostuemID,
        )

        if is_base_element:
            query = query.order_by(
                costume_base_element_table.c.BasiselementID,
            )

        query_result = database.session.execute(query).all()

        # Convert the result of the query into entity instances
        entities = EntityFactory._entities_from_query_result(
            attributes, is_base_element, query_result
        )

        return entities

    @staticmethod
    def _entities_from_query_result(
        attributes, is_base_element, query_result
    ) -> List[Entity]:
        entities = []

        for row in query_result:
            entity = Entity("Entity")

            costume_id = row[0]
            role_id = row[1]
            film_id = row[2]

            entity.kostuemId = costume_id
            entity.rollenId = role_id
            entity.filmId = film_id

            if is_base_element:
                base_element_id = row[3]
                entity.basiselementID = base_element_id
                row = row[4:]
            else:
                row = row[3:]

            for i, attr in enumerate(attributes):
                value: str = row[i]

                if value is None or value == "":
                    entity.add_value(attr, None)
                    logging.warning("Found entry with " + attr.value + ' = None or = ""')
                    break

                entity.add_attribute(attr)

                if isinstance(value, str):
                    entity.add_value(attr, value.split(","))
                else:
                    entity.add_value(attr, value)

            entities.append(entity)

        return entities


def main():
    db = Database()
    db.open_with_params(
        host="localhost",
        user="test",
        password="test",
        database="KostuemRepo",
    )

    time1 = time.time()

    EntityFactory.create(
        [
            # Attribute.ortsbegebenheit,
            # Attribute.dominanteFarbe,
            # Attribute.stereotypRelevant,
            # Attribute.dominanteFunktion,
            # Attribute.dominanterZustand,
            #
            # Attribute.farbkonzept,
            #
            # Attribute.dominanteCharaktereigenschaft,
            #
            # Attribute.stereotyp,
            #
            # Attribute.rollenberuf,
            # Attribute.geschlecht,
            # Attribute.dominanterAlterseindruck,
            # Attribute.dominantesAlter,
            # Attribute.rollenrelevanz,
            #
            # Attribute.genre,
            #
            # Attribute.spielzeit,
            #
            # Attribute.tageszeit,
            #
            # Attribute.koerpermodifikation,
            #
            # Attribute.kostuemZeit,
            #
            # Attribute.familienstand,
            #
            # Attribute.charaktereigenschaft,
            #
            # Attribute.spielort,
            # Attribute.spielortDetail,
            #
            Attribute.alterseindruck,
            Attribute.alter,
            #
            # Attribute.basiselement,
            #
            # Attribute.design,
            #
            # Attribute.form,
            #
            # Attribute.trageweise,
            #
            # Attribute.zustand,
            #
            # Attribute.funktion,
            #
            # Attribute.material,
            # Attribute.materialeindruck,
            #
            # Attribute.farbe,
            # Attribute.farbeindruck,
        ],
        db,
    )

    time2 = time.time()

    print(time2 - time1)


if __name__ == "__main__":
    main()
