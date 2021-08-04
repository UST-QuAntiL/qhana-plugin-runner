import logging
import time
from typing import Any, Dict, Tuple
from typing import List

from sqlalchemy import select, tuple_, and_, join, cast, String
from sqlalchemy.orm import Bundle

from plugins.costume_loader_pkg.backend.database import Database
from plugins.costume_loader_pkg.backend.taxonomy import Taxonomie
from datetime import datetime, timedelta
from plugins.costume_loader_pkg.backend.attribute import Attribute
import copy

MUSE_URL = "http://129.69.214.108/"


"""
This class represents a entity such as a costume or a basic element.
"""


class Entity:
    """
    Initializes the entity object.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.id = 0
        self.kostuemId = 0
        self.rollenId = 0
        self.filmId = 0
        # format: (attribute, None)
        self.attributes = {}
        # format: (attribute, value)
        self.values = {}

        return

    """
    Sets the id of the entity.
    """

    def set_id(self, id: int) -> None:
        self.id = id
        return

    """
    Gets the id of the entity.
    """

    def get_id(self) -> int:
        return self.id

    """
    Sets the basiselementId of the entity.
    """

    def set_basiselement_id(self, basiselementId: int) -> None:
        self.basiselementId = basiselementId
        return

    """
    Gets the basiselementId of the entity.
    """

    def get_basiselement_id(self) -> int:
        return self.basiselementId

    """
    Sets the kostuemId of the entity.
    """

    def set_kostuem_id(self, kostuemId: int) -> None:
        self.kostuemId = kostuemId
        return

    """
    Gets the kostuemId of the entity.
    """

    def get_kostuem_id(self) -> int:
        return self.kostuemId

    """
    Sets the rollenId of the entity.
    """

    def set_rollen_id(self, rollenId: int) -> None:
        self.rollenId = rollenId
        return

    """
    Gets the rollenId of the entity.
    """

    def get_rollen_id(self) -> int:
        return self.rollenId

    """
    Sets the filmId of the entity.
    """

    def set_film_id(self, filmId: int) -> None:
        self.filmId = filmId
        return

    """
    Gets the filmid of the entity.
    """

    def get_film_id(self) -> int:
        return self.filmId

    """
    Gets the film url of the entity.
    """

    def get_film_url(self) -> str:
        return MUSE_URL + "#/filme/" + str(self.get_film_id())

    """
    Gets the rollen url of the entity.
    """

    def get_rollen_url(self) -> str:
        return self.get_film_url() + "/rollen/" + str(self.get_rollen_id())

    """
    Gets the kostuem url of the entity.
    """

    def get_kostuem_url(self) -> str:
        return self.get_rollen_url() + "/kostueme/" + str(self.get_kostuem_id())

    """
    Adds an attribute to the attributes list.
    """

    def add_attribute(self, attribute: Attribute) -> None:
        self.attributes[attribute] = None
        return

    """
    Adds a value to the values list. If the associated
    attribute is not added yet, it will be added.
    """

    def add_value(self, attribute: Attribute, value: Any) -> None:
        if attribute not in self.attributes:
            self.add_attribute(attribute)
        self.values[attribute] = value
        return

    """
    Gets the value of the given attribute.
    """

    def get_value(self, attribute: Attribute):
        return self.values[attribute]

    """
    Removes an attribute from the attributes list.
    """

    def remove_attribute(self, attribute: Attribute) -> None:
        del self.attributes[attribute]
        return

    """
    Removes a value from the values list.
    """

    def remove_attribute(self, attribute: Attribute) -> None:
        del self.values[attribute]
        return

    """
    Returns the entity in a single string.
    """

    def __str__(self) -> str:
        output = "name: " + self.name + ", id: " + str(self.id) + ", "
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


"""
This class creats entities based on the given attributes.
"""


class EntityFactory:
    """
    Creates a entity based on the given list of attributes.
    The default for amount is max int, all entities that are found will be returned.
    A filter for the keys can be specified which is a set with the format
    { (KostuemID, RollenID, FilmID) , ... }
    """

    @staticmethod
    def _is_value_in_list_of_values(v: str, lv: List[str]) -> bool:
        for v2 in lv:
            if v.lower() == v2.lower():
                return True

        return False

    @staticmethod
    def _expand_filter_term(
        attribute: Attribute, filter_term: str, db: Database
    ) -> List[str]:
        if filter_term.startswith("*"):
            value = filter_term[1:].lower()
            taxonomy_type = Attribute.get_taxonomie_type(attribute)
            taxonomy = Taxonomie.create_from_db(taxonomy_type, db)

            lowercase_adj = {}

            k: str
            v: Dict

            for k, v in taxonomy.graph.adj.items():
                lowercase_adj[k.lower()] = [item.lower() for item in v.keys()]

            processed_nodes = []
            new_nodes = [value]

            while len(new_nodes) > 0:
                current_node = new_nodes.pop()

                new_nodes.extend(lowercase_adj[current_node])
                processed_nodes.append(current_node)

            return processed_nodes
        else:
            return [filter_term]

    @staticmethod
    def _is_accepted_by_filter(
        entity: Entity, filter_rules: Dict[Attribute, List[str]], db: Database
    ) -> bool:
        """
        Tests if the entity is accepted by the provided filter rules or if it should be filtered out.
        :param entity: entity
        :param filter_rules: filter rules
        :return: True if accepted, false if not.
        """
        for attribute in filter_rules:
            attr_accepted = False
            expanded_rules = [
                new_term
                for term in filter_rules[attribute]
                for new_term in EntityFactory._expand_filter_term(attribute, term, db)
            ]

            for value in expanded_rules:
                if (
                    EntityFactory._is_value_in_list_of_values(
                        value, entity.get_value(attribute)
                    )
                    or value == ""
                ):
                    attr_accepted = True

            if not attr_accepted:
                return False

        return True

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
        join_on[costume_table] = and_(
            costume_table.KostuemID == costume_table.KostuemID,
            costume_table.RollenID == costume_table.RollenID,
            costume_table.FilmID == costume_table.FilmID,
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
        join_on[color_concept_table] = and_(
            costume_table.FilmID == color_concept_table.c.FilmID
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
        join_on[dominant_character_trait_table] = and_(
            costume_table.RollenID == dominant_character_trait_table.c.RollenID,
            costume_table.FilmID == dominant_character_trait_table.c.FilmID,
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
        join_on[role_stereotype_table] = and_(
            costume_table.RollenID == role_stereotype_table.c.RollenID,
            costume_table.FilmID == role_stereotype_table.c.FilmID,
        )

        tables[Attribute.stereotyp] = role_stereotype_table
        columns[Attribute.stereotyp] = role_stereotype_table.c.Stereotyp

        ###############
        # Rolle table #
        ###############

        role_table = database.base.classes.Rolle
        join_on[role_table] = and_(
            costume_table.RollenID == role_table.RollenID,
            costume_table.FilmID == role_table.FilmID,
        )

        tables[Attribute.rollenberuf] = role_table
        columns[Attribute.rollenberuf] = role_table.Rollenberuf

        tables[Attribute.geschlecht] = role_table
        columns[Attribute.geschlecht] = role_table.Geschlecht

        ###################
        # FilmGenre table #
        ###################

        genre_table = database.get_film_cte(database.other_tables["FilmGenre"], "Genre")
        join_on[genre_table] = and_(costume_table.FilmID == genre_table.c.FilmID)

        tables[Attribute.genre] = genre_table
        columns[Attribute.genre] = genre_table.c.Genre

        ###################
        # KostuemSpielzeit table #
        ###################

        costume_playtime_table = database.get_costume_cte(
            database.base.classes.KostuemSpielzeit, "Spielzeit"
        )
        join_on[costume_playtime_table] = and_(
            costume_table.KostuemID == costume_playtime_table.c.KostuemID,
            costume_table.RollenID == costume_playtime_table.c.RollenID,
            costume_table.FilmID == costume_playtime_table.c.FilmID,
        )

        tables[Attribute.spielzeit] = costume_playtime_table
        columns[Attribute.spielzeit] = costume_playtime_table.c.Spielzeit

        ##########################
        # KostuemTageszeit table #
        ##########################

        costume_daytime_table = database.get_costume_cte(
            database.other_tables["KostuemTageszeit"], "Tageszeit"
        )
        join_on[costume_daytime_table] = and_(
            costume_table.KostuemID == costume_daytime_table.c.KostuemID,
            costume_table.RollenID == costume_daytime_table.c.RollenID,
            costume_table.FilmID == costume_daytime_table.c.FilmID,
        )

        tables[Attribute.tageszeit] = costume_daytime_table
        columns[Attribute.tageszeit] = costume_daytime_table.c.Tageszeit

        ####################################
        # KostuemKoerpermodifikation table #
        ####################################

        body_modification_table = database.get_costume_cte(
            database.other_tables["KostuemKoerpermodifikation"], "Koerpermodifikationname"
        )
        join_on[body_modification_table] = and_(
            costume_table.KostuemID == body_modification_table.c.KostuemID,
            costume_table.RollenID == body_modification_table.c.RollenID,
            costume_table.FilmID == body_modification_table.c.FilmID,
        )

        tables[Attribute.koerpermodifikation] = body_modification_table
        columns[
            Attribute.koerpermodifikation
        ] = body_modification_table.c.Koerpermodifikationname

        #########################
        # KostuemTimecode table #
        #########################

        timecode_table = database.base.classes.KostuemTimecode
        timecode_cte = select(
            timecode_table.KostuemID,
            timecode_table.RollenID,
            timecode_table.FilmID,
            (
                cast(timecode_table.Timecodeanfang, String)
                + "|"
                + cast(timecode_table.Timecodeende, String)
            ).label("Zeiten"),
        ).cte("TimecodeCTE")

        merged_timecode_table = database.get_costume_cte(timecode_cte, "Zeiten")
        join_on[merged_timecode_table] = and_(
            costume_table.KostuemID == merged_timecode_table.c.KostuemID,
            costume_table.RollenID == merged_timecode_table.c.RollenID,
            costume_table.FilmID == merged_timecode_table.c.FilmID,
        )

        tables[Attribute.kostuemZeit] = merged_timecode_table
        columns[Attribute.kostuemZeit] = merged_timecode_table.c.Zeiten

        ############################
        # RolleFamilienstand table #
        ############################

        status_table = database.get_role_cte(
            database.base.classes.RolleFamilienstand, "Familienstand"
        )
        join_on[status_table] = and_(
            costume_table.RollenID == status_table.c.RollenID,
            costume_table.FilmID == status_table.c.FilmID,
        )

        tables[Attribute.familienstand] = status_table
        columns[Attribute.familienstand] = status_table.c.Familienstand

        #####################################
        # KostuemCharaktereigenschaft table #
        #####################################

        trait_table = database.get_costume_cte(
            database.other_tables["KostuemCharaktereigenschaft"], "Charaktereigenschaft"
        )
        join_on[trait_table] = and_(
            costume_table.KostuemID == trait_table.c.KostuemID,
            costume_table.RollenID == trait_table.c.RollenID,
            costume_table.FilmID == trait_table.c.FilmID,
        )

        tables[Attribute.charaktereigenschaft] = trait_table
        columns[Attribute.charaktereigenschaft] = trait_table.c.Charaktereigenschaft

        #########################
        # KostuemSpielort table #
        #########################

        location_table = database.base.classes.KostuemSpielort
        location_cte = select(
            location_table.KostuemID,
            location_table.RollenID,
            location_table.FilmID,
            (location_table.Spielort + "|" + location_table.SpielortDetail).label(
                "Spielort"
            ),
        ).cte("SpielortCTE")
        merged_location_table = database.get_costume_cte(location_cte, "Spielort")
        join_on[merged_location_table] = and_(
            costume_table.KostuemID == merged_location_table.c.KostuemID,
            costume_table.RollenID == merged_location_table.c.RollenID,
            costume_table.FilmID == merged_location_table.c.FilmID,
        )

        tables[Attribute.spielort] = merged_location_table
        columns[Attribute.spielort] = merged_location_table.c.Spielort

        tables[Attribute.spielortDetail] = merged_location_table
        columns[Attribute.spielortDetail] = merged_location_table.c.Spielort

        ###############################
        # KostuemAlterseindruck table #
        ###############################

        age_impression_table = database.base.classes.KostuemAlterseindruck
        age_impression_cte = select(
            age_impression_table.KostuemID,
            age_impression_table.RollenID,
            age_impression_table.FilmID,
            (
                age_impression_table.Alterseindruck + "|" + age_impression_table.NumAlter
            ).label("Alter"),
        ).cte("AlterCTE")
        merged_age_impression_table = database.get_costume_cte(
            age_impression_cte, "Alter"
        )
        join_on[merged_age_impression_table] = and_(
            costume_table.KostuemID == merged_age_impression_table.c.KostuemID,
            costume_table.RollenID == merged_age_impression_table.c.RollenID,
            costume_table.FilmID == merged_age_impression_table.c.FilmID,
        )

        tables[Attribute.alterseindruck] = merged_age_impression_table
        columns[Attribute.alterseindruck] = merged_age_impression_table.c.Alter

        tables[Attribute.alter] = merged_age_impression_table
        columns[Attribute.alter] = merged_age_impression_table.c.Alter

        #############################
        # KostuemBasiselement table #
        #############################

        costume_base_element_table = database.other_tables["KostuemBasiselement"]
        join_on[costume_base_element_table] = and_(
            costume_table.KostuemID == costume_base_element_table.c.KostuemID,
            costume_table.RollenID == costume_base_element_table.c.RollenID,
            costume_table.FilmID == costume_base_element_table.c.FilmID,
        )

        ######################
        # Basiselement table #
        ######################

        base_element_table = database.base.classes.Basiselement
        join_on[base_element_table] = and_(
            costume_base_element_table.c.BasiselementID
            == base_element_table.BasiselementID
        )

        tables[Attribute.basiselement] = base_element_table
        columns[Attribute.basiselement] = base_element_table.Basiselementname

        ############################
        # BasiselementDesign table #
        ############################

        design_table = database.get_base_element_cte(
            database.other_tables["BasiselementDesign"], "Designname"
        )
        join_on[design_table] = and_(
            costume_base_element_table.c.BasiselementID == design_table.c.BasiselementID
        )

        tables[Attribute.design] = design_table
        columns[Attribute.design] = design_table.c.Designname

        ##########################
        # BasiselementForm table #
        ##########################

        form_table = database.get_base_element_cte(
            database.other_tables["BasiselementForm"], "Formname"
        )
        join_on[form_table] = and_(
            costume_base_element_table.c.BasiselementID == form_table.c.BasiselementID
        )

        tables[Attribute.form] = form_table
        columns[Attribute.form] = form_table.c.Formname

        ##########################
        # BasiselementTrageweise #
        ##########################

        wear_table = database.get_base_element_cte(
            database.other_tables["BasiselementTrageweise"], "Trageweisename"
        )
        join_on[wear_table] = and_(
            costume_base_element_table.c.BasiselementID == wear_table.c.BasiselementID
        )

        tables[Attribute.trageweise] = wear_table
        columns[Attribute.trageweise] = wear_table.c.Trageweisename

        #############################
        # BasiselementZustand table #
        #############################

        condition_table = database.get_base_element_cte(
            database.other_tables["BasiselementZustand"], "Zustandsname"
        )
        join_on[condition_table] = and_(
            costume_base_element_table.c.BasiselementID
            == condition_table.c.BasiselementID
        )

        tables[Attribute.zustand] = condition_table
        columns[Attribute.zustand] = condition_table.c.Zustandsname

        ##############################
        # BasiselementFunktion table #
        ##############################

        function_table = database.get_base_element_cte(
            database.other_tables["BasiselementFunktion"], "Funktionsname"
        )
        join_on[function_table] = and_(
            costume_base_element_table.c.BasiselementID == function_table.c.BasiselementID
        )

        tables[Attribute.funktion] = function_table
        columns[Attribute.funktion] = function_table.c.Funktionsname

        ##############################
        # BasiselementMaterial table #
        ##############################

        material_table = database.base.classes.BasiselementMaterial
        material_cte = select(
            material_table.BasiselementID,
            (material_table.Materialname + "|" + material_table.Materialeindruck).label(
                "Material"
            ),
        ).cte("MaterialCTE")

        merged_material_table = database.get_base_element_cte(material_cte, "Material")
        join_on[merged_material_table] = and_(
            costume_base_element_table.c.BasiselementID
            == merged_material_table.c.BasiselementID
        )

        tables[Attribute.material] = merged_material_table
        columns[Attribute.material] = merged_material_table.c.Material

        tables[Attribute.materialeindruck] = merged_material_table
        columns[Attribute.materialeindruck] = merged_material_table.c.Material

        ###########################
        # BasiselementFarbe table #
        ###########################

        color_table = database.base.classes.BasiselementFarbe
        color_cte = select(
            color_table.BasiselementID,
            (color_table.Farbname + "|" + color_table.Farbeindruck).label("Farbe"),
        ).cte("FarbeCTE")

        merged_color_table = database.get_base_element_cte(color_cte, "Farbe")
        join_on[merged_color_table] = and_(
            costume_base_element_table.c.BasiselementID
            == merged_color_table.c.BasiselementID
        )

        tables[Attribute.farbe] = merged_color_table
        columns[Attribute.farbe] = merged_color_table.c.Farbe

        tables[Attribute.farbeindruck] = merged_color_table
        columns[Attribute.farbeindruck] = merged_color_table.c.Farbe

        return tables, columns, join_on, costume_table, costume_base_element_table

    @staticmethod
    def create(attributes: List[Attribute], database: Database) -> List[Entity]:
        invalid_entries = 0

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

            # load base element if needed
            # this also means that we are now treating
            # each datapoint as a base element
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
            attributes, invalid_entries, is_base_element, query_result
        )

        return entities

    @staticmethod
    def _entities_from_query_result(
        attributes, invalid_entries, is_base_element, query_result
    ) -> List[Entity]:
        entities = []

        for row in query_result:
            entity = Entity("Entity")

            costume_id = row[0]
            role_id = row[1]
            film_id = row[2]

            entity.set_kostuem_id(costume_id)
            entity.set_rollen_id(role_id)
            entity.set_film_id(film_id)

            offset = 3

            if is_base_element:
                base_element_id = row[3]
                entity.set_basiselement_id(base_element_id)
                offset = 4

            is_invalid = False

            for i, attr in enumerate(attributes):
                value: str = row[i + offset]

                if attr in [
                    Attribute.spielort,
                    Attribute.alterseindruck,
                    Attribute.material,
                    Attribute.farbe,
                ]:
                    value = ",".join([elem.split("|")[0] for elem in value.split(",")])
                if attr in [
                    Attribute.spielortDetail,
                    Attribute.alter,
                    Attribute.materialeindruck,
                    Attribute.farbeindruck,
                ]:
                    value = ",".join([elem.split("|")[1] for elem in value.split(",")])

                if attr == Attribute.kostuemZeit:
                    costume_time = 0

                    for time_pair in value.split(","):
                        start_time = time_pair.split("|")[0].split(":")
                        end_time = time_pair.split("|")[1].split(":")

                        start_time_delta = timedelta(
                            hours=int(start_time[0]),
                            minutes=int(start_time[1]),
                            seconds=int(start_time[2]),
                        )

                        end_time_delta = timedelta(
                            hours=int(end_time[0]),
                            minutes=int(end_time[1]),
                            seconds=int(end_time[2]),
                        )

                        costume_time += (
                            end_time_delta - start_time_delta
                        ).total_seconds()

                    value = str(costume_time)

                if value is None or value == "":
                    invalid_entries += 1
                    is_invalid = True
                    logging.warning("Found entry with " + attr.value + ' = None or = ""')
                    break

                entity.add_attribute(attr)
                entity.add_value(attr, value.split(","))

            if is_invalid:
                continue

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
            Attribute.genre,
            Attribute.dominanteFarbe,
            # Attribute.farbe,
            # Attribute.farbeindruck,
            Attribute.kostuemZeit,
            # Attribute.alter,
            # Attribute.alterseindruck,
        ],
        db,
    )

    time2 = time.time()

    print(time2 - time1)


if __name__ == "__main__":
    main()
