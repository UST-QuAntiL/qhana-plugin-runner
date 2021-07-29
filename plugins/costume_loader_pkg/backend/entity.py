import logging
from typing import Any, Dict
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
    def create(
        attributes: List[Attribute],
        database: Database,
        amount: int = 2147483646,
        keys=None,
        filter_rules: Dict[Attribute, List[str]] = {},
    ) -> List[Entity]:

        entities = []
        invalid_entries = 0

        columns_to_select = []
        tables_to_join = set()
        join_on = {}

        # First, run as we are just collecting
        # costumes. If there is an attribute set
        # that is just part of a basiselement,
        # this flag will be changed and all
        # entities will be trated as basiselements.
        be_basiselement = False

        costume_table = database.base.classes.Kostuem
        join_on[costume_table] = and_(
            costume_table.KostuemID == costume_table.KostuemID,
            costume_table.RollenID == costume_table.RollenID,
            costume_table.FilmID == costume_table.FilmID,
        )

        if Attribute.ortsbegebenheit in attributes:
            columns_to_select.append(costume_table.Ortsbegebenheit)
        if Attribute.dominanteFarbe in attributes:
            columns_to_select.append(costume_table.DominanteFarbe)
        if Attribute.stereotypRelevant in attributes:
            columns_to_select.append(costume_table.StereotypRelevant)
        if Attribute.dominanteFunktion in attributes:
            columns_to_select.append(costume_table.DominanteFunktion)
        if Attribute.dominanterZustand in attributes:
            columns_to_select.append(costume_table.DominanterZustand)

        color_concept_table = db.get_film_cte(
            database.other_tables["FilmFarbkonzept"], "Farbkonzept"
        )
        join_on[color_concept_table] = and_(
            costume_table.FilmID == color_concept_table.c.FilmID
        )

        if Attribute.farbkonzept in attributes:
            columns_to_select.append(color_concept_table.c.Farbkonzept)
            tables_to_join.add(color_concept_table)

        dominant_character_trait_table = db.get_role_cte(
            database.other_tables["RolleDominanteCharaktereigenschaft"],
            "DominanteCharaktereigenschaft",
        )
        join_on[dominant_character_trait_table] = and_(
            costume_table.RollenID == dominant_character_trait_table.c.RollenID,
            costume_table.FilmID == dominant_character_trait_table.c.FilmID,
        )

        if Attribute.dominanteCharaktereigenschaft in attributes:
            columns_to_select.append(
                dominant_character_trait_table.c.DominanteCharaktereigenschaft
            )
            tables_to_join.add(dominant_character_trait_table)

        role_stereotype_table = db.get_role_cte(
            database.base.classes.RolleStereotyp, "Stereotyp"
        )
        join_on[role_stereotype_table] = and_(
            costume_table.RollenID == role_stereotype_table.c.RollenID,
            costume_table.FilmID == role_stereotype_table.c.FilmID,
        )

        if Attribute.stereotyp in attributes:
            columns_to_select.append(role_stereotype_table.c.Stereotyp)
            tables_to_join.add(role_stereotype_table)

        role_table = database.base.classes.Rolle
        join_on[role_table] = and_(
            costume_table.RollenID == role_table.RollenID,
            costume_table.FilmID == role_table.FilmID,
        )

        if Attribute.rollenberuf in attributes:
            columns_to_select.append(role_table.Rollenberuf)
            tables_to_join.add(role_table)
        if Attribute.geschlecht in attributes:
            columns_to_select.append(role_table.Geschlecht)
            tables_to_join.add(role_table)
        if Attribute.dominanterAlterseindruck in attributes:
            columns_to_select.append(role_table.DominanterAlterseindruck)
            tables_to_join.add(role_table)
        if Attribute.dominantesAlter in attributes:
            columns_to_select.append(role_table.DominantesAlter)
            tables_to_join.add(role_table)
        if Attribute.rollenrelevanz in attributes:
            columns_to_select.append(role_table.Rollenrelevanz)
            tables_to_join.add(role_table)

        genre_table = db.get_film_cte(database.other_tables["FilmGenre"], "Genre")
        join_on[genre_table] = and_(costume_table.FilmID == genre_table.c.FilmID)

        if Attribute.genre in attributes:
            columns_to_select.append(genre_table.c.Genre)
            tables_to_join.add(genre_table)

        costume_playtime_table = db.get_costume_cte(
            database.base.classes.KostuemSpielzeit, "Spielzeit"
        )
        join_on[costume_playtime_table] = and_(
            costume_table.KostuemID == costume_playtime_table.c.KostuemID,
            costume_table.RollenID == costume_playtime_table.c.RollenID,
            costume_table.FilmID == costume_playtime_table.c.FilmID,
        )

        if Attribute.spielzeit in attributes:
            columns_to_select.append(costume_playtime_table.c.Spielzeit)
            tables_to_join.add(costume_playtime_table)

        costume_daytime_table = db.get_costume_cte(
            database.other_tables["KostuemTageszeit"], "Tageszeit"
        )
        join_on[costume_daytime_table] = and_(
            costume_table.KostuemID == costume_daytime_table.c.KostuemID,
            costume_table.RollenID == costume_daytime_table.c.RollenID,
            costume_table.FilmID == costume_daytime_table.c.FilmID,
        )

        if Attribute.tageszeit in attributes:
            columns_to_select.append(costume_daytime_table.c.Tageszeit)
            tables_to_join.add(costume_daytime_table)

        body_modification_table = db.get_costume_cte(
            database.other_tables["KostuemKoerpermodifikation"], "Koerpermodifikationname"
        )
        join_on[body_modification_table] = and_(
            costume_table.KostuemID == body_modification_table.c.KostuemID,
            costume_table.RollenID == body_modification_table.c.RollenID,
            costume_table.FilmID == body_modification_table.c.FilmID,
        )

        if Attribute.koerpermodifikation in attributes:
            columns_to_select.append(body_modification_table.c.Koerpermodifikationname)
            tables_to_join.add(body_modification_table)

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

        merged_timecode_table = db.get_costume_cte(timecode_cte, "Zeiten")
        join_on[merged_timecode_table] = and_(
            costume_table.KostuemID == merged_timecode_table.c.KostuemID,
            costume_table.RollenID == merged_timecode_table.c.RollenID,
            costume_table.FilmID == merged_timecode_table.c.FilmID,
        )

        if Attribute.kostuemZeit in attributes:
            columns_to_select.append(merged_timecode_table.c.Zeiten)
            tables_to_join.add(merged_timecode_table)

        status_table = db.get_role_cte(
            database.base.classes.RolleFamilienstand, "Familienstand"
        )
        join_on[status_table] = and_(
            costume_table.RollenID == status_table.c.RollenID,
            costume_table.FilmID == status_table.c.FilmID,
        )

        if Attribute.familienstand in attributes:
            columns_to_select.append(status_table.c.Familienstand)
            tables_to_join.add(status_table)

        trait_table = db.get_costume_cte(
            database.other_tables["KostuemCharaktereigenschaft"], "Charaktereigenschaft"
        )
        join_on[trait_table] = and_(
            costume_table.KostuemID == trait_table.c.KostuemID,
            costume_table.RollenID == trait_table.c.RollenID,
            costume_table.FilmID == trait_table.c.FilmID,
        )

        if Attribute.charaktereigenschaft in attributes:
            columns_to_select.append(trait_table.c.Charaktereigenschaft)
            tables_to_join.add(trait_table)

        location_table = database.base.classes.KostuemSpielort
        location_cte = select(
            location_table.KostuemID,
            location_table.RollenID,
            location_table.FilmID,
            (location_table.Spielort + "|" + location_table.SpielortDetail).label(
                "Spielort"
            ),
        ).cte("SpielortCTE")

        merged_location_table = db.get_costume_cte(location_cte, "Spielort")
        join_on[merged_location_table] = and_(
            costume_table.KostuemID == merged_location_table.c.KostuemID,
            costume_table.RollenID == merged_location_table.c.RollenID,
            costume_table.FilmID == merged_location_table.c.FilmID,
        )

        if Attribute.spielort in attributes or Attribute.spielortDetail in attributes:
            columns_to_select.append(merged_location_table.c.Spielort)
            tables_to_join.add(merged_location_table)

        age_impression_table1 = db.get_costume_cte(
            database.base.classes.KostuemAlterseindruck, "Alterseindruck", "1"
        )
        join_on[age_impression_table1] = and_(
            costume_table.KostuemID == age_impression_table1.c.KostuemID,
            costume_table.RollenID == age_impression_table1.c.RollenID,
            costume_table.FilmID == age_impression_table1.c.FilmID,
        )
        age_impression_table2 = db.get_costume_cte(
            database.base.classes.KostuemAlterseindruck, "NumAlter", "2"
        )
        join_on[age_impression_table2] = and_(
            costume_table.KostuemID == age_impression_table2.c.KostuemID,
            costume_table.RollenID == age_impression_table2.c.RollenID,
            costume_table.FilmID == age_impression_table2.c.FilmID,
        )

        if Attribute.alterseindruck in attributes:
            columns_to_select.append(age_impression_table1.c.Alterseindruck)
            tables_to_join.add(age_impression_table1)
        if Attribute.alter in attributes:
            columns_to_select.append(age_impression_table2.c.NumAlter)
            tables_to_join.add(age_impression_table2)

        ############################
        # basis element attributes #
        ############################

        # TODO: CTE
        # TODO: one row per basiselement
        is_basiselement = False

        # load basiselement if needed
        # this also means that we are now treat
        # each datapoint as a basiselement
        if (
            Attribute.basiselement in attributes
            or Attribute.design in attributes
            or Attribute.form in attributes
            or Attribute.trageweise in attributes
            or Attribute.zustand in attributes
            or Attribute.funktion in attributes
            or Attribute.material in attributes
            or Attribute.materialeindruck in attributes
            or Attribute.farbe in attributes
            or Attribute.farbeindruck in attributes
        ):
            is_basiselement = True
            costume_base_element_table = database.other_tables["KostuemBasiselement"]
            join_on[costume_base_element_table] = and_(
                costume_table.KostuemID == costume_base_element_table.c.KostuemID,
                costume_table.RollenID == costume_base_element_table.c.RollenID,
                costume_table.FilmID == costume_base_element_table.c.FilmID,
            )

            base_element_table = database.base.classes.Basiselement
            join_on[base_element_table] = and_(
                costume_base_element_table.c.BasiselementID
                == base_element_table.BasiselementID
            )

            if Attribute.basiselement in attributes:
                columns_to_select.append(base_element_table.Basiselementname)
                tables_to_join.add(base_element_table)

            design_table = database.other_tables["BasiselementDesign"]
            join_on[design_table] = and_(
                costume_base_element_table.c.BasiselementID
                == design_table.c.BasiselementID
            )

            if Attribute.design in attributes:
                columns_to_select.append(design_table.c.Designname)
                tables_to_join.add(design_table)

            form_table = database.other_tables["BasiselementForm"]
            join_on[form_table] = and_(
                costume_base_element_table.c.BasiselementID == form_table.c.BasiselementID
            )

            if Attribute.form in attributes:
                columns_to_select.append(form_table.c.Formname)
                tables_to_join.add(form_table)

            wear_table = database.other_tables["BasiselementTrageweise"]
            join_on[wear_table] = and_(
                costume_base_element_table.c.BasiselementID == wear_table.c.BasiselementID
            )

            if Attribute.trageweise in attributes:
                columns_to_select.append(wear_table.c.Trageweisename)
                tables_to_join.add(wear_table)

            condition_table = database.other_tables["BasiselementZustand"]
            join_on[condition_table] = and_(
                costume_base_element_table.c.BasiselementID
                == condition_table.c.BasiselementID
            )

            if Attribute.zustand in attributes:
                columns_to_select.append(condition_table.c.Zustandsname)
                tables_to_join.add(condition_table)

            function_table = database.other_tables["BasiselementFunktion"]
            join_on[function_table] = and_(
                costume_base_element_table.c.BasiselementID
                == function_table.c.BasiselementID
            )

            if Attribute.funktion in attributes:
                columns_to_select.append(function_table.c.Funktionsname)
                tables_to_join.add(function_table)

            material_table = database.base.classes.BasiselementMaterial
            join_on[material_table] = and_(
                costume_base_element_table.c.BasiselementID
                == material_table.BasiselementID
            )

            if Attribute.material in attributes:
                columns_to_select.append(material_table.Materialname)
                tables_to_join.add(material_table)
            if Attribute.materialeindruck in attributes:
                columns_to_select.append(material_table.Materialeindruck)
                tables_to_join.add(material_table)

            color_table = database.base.classes.BasiselementFarbe
            join_on[color_table] = and_(
                costume_base_element_table.c.BasiselementID == color_table.BasiselementID
            )

            if Attribute.farbe in attributes:
                columns_to_select.append(color_table.Farbname)
                tables_to_join.add(color_table)
            if Attribute.farbeindruck in attributes:
                columns_to_select.append(color_table.Farbeindruck)
                tables_to_join.add(color_table)

        if is_basiselement:
            columns_to_select = [
                costume_base_element_table.c.BasiselementID
            ] + columns_to_select

        columns_to_select = [
            costume_table.KostuemID,
            costume_table.RollenID,
            costume_table.FilmID,
        ] + columns_to_select

        query = select(*columns_to_select)

        if is_basiselement:
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

        query = query.select_from(j)
        query = query.order_by(
            costume_table.FilmID,
            costume_table.RollenID,
            costume_table.KostuemID,
        )

        if is_basiselement:
            query = query.order_by(
                costume_base_element_table.c.BasiselementID,
            )

        result = database.session.execute(query).all()

        return entities


"""
This class represents a costume in the simplified model
NOTE: We use stereotyp although it is not defined in the simplified model
"""


class Costume(Entity):
    """
    Initialize a costume object with the predefined attributes from the
    simplified model.
    """

    def __init__(self, id) -> None:
        super().__init__("Kostüm", id)

        # add all the attributes
        self.add_attribute(Attribute.dominanteFarbe)
        self.add_attribute(Attribute.dominanteCharaktereigenschaft)
        self.add_attribute(Attribute.dominanterZustand)
        self.add_attribute(Attribute.stereotyp)
        self.add_attribute(Attribute.geschlecht)
        self.add_attribute(Attribute.dominanterAlterseindruck)
        self.add_attribute(Attribute.genre)

        return


""" 
Represents the factory to create costumes
"""


class CostumeFactory:
    """
    Creates one single costume.
    """

    @staticmethod
    def __create_costume(
        id: Any,
        dominanteFarbe: [str],
        dominanteCharaktereigenschaft: [str],
        dominanterZustand: [str],
        stereotyp: [str],
        geschlecht: [str],
        dominanterAlterseindruck: [str],
        genre: [str],
        database: Database,
    ) -> Costume:

        costume = Costume(id)

        # add all the values
        costume.add_value(Attribute.dominanteFarbe, dominanteFarbe)
        costume.add_value(
            Attribute.dominanteCharaktereigenschaft, dominanteCharaktereigenschaft
        )
        costume.add_value(Attribute.dominanterZustand, dominanterZustand)
        costume.add_value(Attribute.stereotyp, stereotyp)
        costume.add_value(Attribute.geschlecht, geschlecht)
        costume.add_value(Attribute.dominanterAlterseindruck, dominanterAlterseindruck)
        costume.add_value(Attribute.genre, genre)

        # add all the bases, i.e. taxonomies
        costume.add_base(
            Attribute.dominanteFarbe,
            Attribute.get_base(Attribute.dominanteFarbe, database),
        )
        costume.add_base(
            Attribute.dominanteCharaktereigenschaft,
            Attribute.get_base(Attribute.dominanteCharaktereigenschaft, database),
        )
        costume.add_base(
            Attribute.dominanterZustand,
            Attribute.get_base(Attribute.dominanterZustand, database),
        )
        costume.add_base(
            Attribute.stereotyp, Attribute.get_base(Attribute.stereotyp, database)
        )
        costume.add_base(
            Attribute.geschlecht, Attribute.get_base(Attribute.geschlecht, database)
        )
        costume.add_base(
            Attribute.dominanterAlterseindruck,
            Attribute.get_base(Attribute.dominanterAlterseindruck, database),
        )
        costume.add_base(Attribute.genre, Attribute.get_base(Attribute.genre, database))

        return costume

    """
    Create all costumes from the database.
    NOTE: Currently there are clones of costumes but I dont know why
    NOTE: Mybe b.c. there are several costume entries in table Kostuem
    NOTE: that have the same ID (but why?)
    TODO: Implement filter possibility
    """

    @staticmethod
    def create(database: Database, amount: int = 0) -> List[Costume]:
        # NOTE: This apprach does not perform good!
        # NOTE: Here, we do 4 transactions to the db for
        # NOTE: each costume entry!
        # TODO: Use different approach to load the costume attributes,
        # TODO: i.e. safe a keys list [KostuemID, RollenID, FilmID]
        # TODO: and iteraite throught this list afterwards.
        costumes: List[Costume] = []
        invalid_entries = 0

        cursor = database.get_cursor()

        # Get dominant_color and dominant_condition directly from Kostuem table
        query_costume = "SELECT KostuemID, RollenID, FilmID, DominanteFarbe, DominanterZustand FROM Kostuem"
        cursor.execute(query_costume)
        rows_costume = cursor.fetchall()

        count = 0
        if amount <= 0:
            amount = 2147483646

        for row_costume in rows_costume:
            if count == amount:
                break
            if row_costume[0] == None:
                invalid_entries += 1
                logging.warning(
                    "Found entry with KostuemID = None. This entry will be skipped."
                )
                continue
            if row_costume[1] == None:
                invalid_entries += 1
                logging.warning(
                    "Found entry with RollenID = None. This entry will be skipped."
                )
                continue
            if row_costume[2] == None:
                invalid_entries += 1
                logging.warning(
                    "Found entry with FilmID = None. This entry will be skipped."
                )
                continue
            if row_costume[3] == None:
                invalid_entries += 1
                logging.warning(
                    "Found entry with DominanteFarbe = None. This entry will be skipped."
                )
                continue
            if row_costume[4] == None:
                invalid_entries += 1
                logging.warning(
                    "Found entry with DominanterZustand = None. This entry will be skipped."
                )
                continue

            kostuemId = row_costume[0]
            rollenId = row_costume[1]
            filmId = row_costume[2]
            dominanteFarbe = [row_costume[3]]
            dominanterZustand = [row_costume[4]]
            dominanteCharaktereigenschaft = []
            stereotyp = []
            geschlecht = []
            dominanterAlterseindruck = []
            genre = []

            # Get dominant_traits from table RolleDominanteCharaktereigenschaften
            query_trait = "SELECT DominanteCharaktereigenschaft FROM RolleDominanteCharaktereigenschaft "
            query_trait += "WHERE RollenID = %s AND FilmID = %s"
            cursor.execute(query_trait, (row_costume[1], row_costume[2]))
            rows_trait = cursor.fetchall()

            if len(rows_trait) == 0:
                invalid_entries += 1
                logging.warning(
                    "Found entry with no DominanteCharaktereigenschaft. Associated entries are: "
                    + "RollenID = "
                    + str(row_costume[1])
                    + ", "
                    + "FilmID = "
                    + str(row_costume[2])
                    + ". "
                    + "This entry will be skipped."
                )
                continue

            for row_trait in rows_trait:
                dominanteCharaktereigenschaft.append(row_trait[0])

            # Get stereotypes from table RolleStereotyp
            query_stereotype = (
                "SELECT Stereotyp FROM RolleStereotyp WHERE RollenID = %s AND FilmID = %s"
            )
            cursor.execute(query_stereotype, (row_costume[1], row_costume[2]))
            rows_stereotype = cursor.fetchall()

            if len(rows_stereotype) == 0:
                invalid_entries += 1
                logging.warning(
                    "Found entry with no Stereotyp. Associated entries are: "
                    + "RollenID = "
                    + str(row_costume[1])
                    + ", "
                    + "FilmID = "
                    + str(row_costume[2])
                    + ". "
                    + "This entry will be skipped."
                )
                continue

            for row_stereotype in rows_stereotype:
                stereotyp.append(row_stereotype[0])

            # Get gender and dominant_age_impression from table Rolle
            query_gender_age = (
                "SELECT Geschlecht, DominanterAlterseindruck FROM Rolle WHERE "
            )
            query_gender_age += "RollenID = %s AND FilmID = %s"
            cursor.execute(query_gender_age, (row_costume[1], row_costume[2]))
            rows_gender_age = cursor.fetchall()

            if len(rows_gender_age) == 0:
                invalid_entries += 1
                logging.warning(
                    "Found entry with no Geschlecht and no DominanterAlterseindruck. Associated entries are: "
                    + "RollenID = "
                    + str(row_costume[1])
                    + ", "
                    + "FilmID = "
                    + str(row_costume[2])
                    + ". "
                    + "This entry will be skipped."
                )
                continue

            for row_gender_age in rows_gender_age:
                if row_gender_age[0] == None:
                    invalid_entries += 1
                    logging.warning(
                        "Found entry with no Geschlecht. Associated entries are: "
                        + "RollenID = "
                        + str(row_costume[1])
                        + ", "
                        + "FilmID = "
                        + str(row_costume[2])
                        + ". "
                        + "This entry will be skipped."
                    )
                    continue
                if row_gender_age[1] == None:
                    invalid_entries += 1
                    logging.warning(
                        "Found entry with no DominanterAlterseindruck. Associated entries are: "
                        + "RollenID = "
                        + str(row_costume[1])
                        + ", "
                        + "FilmID = "
                        + str(row_costume[2])
                        + ". "
                        + "This entry will be skipped."
                    )
                    continue

                geschlecht.append(row_gender_age[0].pop())
                dominanterAlterseindruck.append(row_gender_age[1])

            # Get genres from table FilmGenre
            query_genre = "SELECT Genre FROM FilmGenre WHERE FilmID = %s"
            cursor.execute(query_genre, (row_costume[2],))
            rows_genre = cursor.fetchall()

            if len(rows_genre) == 0:
                invalid_entries += 1
                logging.warning(
                    "Found entry with no Genre. Associated entry is: "
                    + "FilmID = "
                    + str(row_costume[2])
                    + ". "
                    + "This entry will be skipped."
                )
                continue

            for row_genre in rows_genre:
                genre.append(row_genre[0])

            costume_id = str(kostuemId) + ":" + str(rollenId) + ":" + str(filmId)

            costume = CostumeFactory.__create_costume(
                costume_id,
                dominanteFarbe,
                dominanteCharaktereigenschaft,
                dominanterZustand,
                stereotyp,
                geschlecht,
                dominanterAlterseindruck,
                genre,
                database,
            )

            costumes.append(costume)
            count += 1

        cursor.close()

        if invalid_entries > 0:
            logging.warning(
                str(invalid_entries)
                + " costumes from "
                + str(count)
                + " are invalid. "
                + "These invalid costumes won't be used for further application."
            )

        return costumes

    # Compares the costumes and their data with the
    # taxonomies and check for inconsistencies
    # file: specifies a separate file for the output
    # if None: log/yyyy_mm_dd_hh_ss_db_validation.log
    # will be used
    @staticmethod
    def validate_all_costumes(self, file=None) -> None:
        logging.info("Running validation of the database.")

        if self.connected == False:
            logging.error(
                "Validation not possible because we are currently not connected to a database."
            )
        if file == None:
            file = (
                "log/"
                + datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
                + "_db_validation.log"
            )

        # load all taxonomies
        color = self.get_color()
        traits = self.get_traits()
        condition = self.get_condition()
        stereotype = self.get_stereotype()
        gender = self.get_gender()
        age_impression = self.get_age_impression()
        genre = self.get_genre()

        # load all costumes
        costumes = self.get_costumes()

        output_string = ""

        found_errors = []

        invalid_costumes = 0
        is_current_costume_invalid = False

        # compare all costumes with taxonomies
        for costume in costumes:
            if not color.has_node(costume.dominant_color):
                error = (
                    costume.dominant_color + " could not be found in color taxonomie. "
                )
                is_current_costume_invalid = True
                if error not in found_errors:
                    found_errors.append(error)
                    output_string += error + " costume: " + str(costume) + "\n"

            for trait in costume.dominant_traits:
                if not traits.has_node(trait):
                    error = trait + " could not be found in traits taxonomie. "
                    is_current_costume_invalid = True
                    if error not in found_errors:
                        found_errors.append(error)
                        output_string += error + " costume: " + str(costume) + "\n"

            if not condition.has_node(costume.dominant_condition):
                error = (
                    costume.dominant_condition
                    + " could not be found in condition taxonomie. "
                )
                is_current_costume_invalid = True
                if error not in found_errors:
                    found_errors.append(error)
                    output_string += error + " costume: " + str(costume) + "\n"

            for _stereotype in costume.stereotypes:
                if not stereotype.has_node(_stereotype):
                    error = _stereotype + " could not be found in stereotype taxonomie. "
                    is_current_costume_invalid = True
                    if error not in found_errors:
                        found_errors.append(error)
                        output_string += error + " costume: " + str(costume) + "\n"

            if not gender.has_node(costume.gender):
                error = costume.gender + " could not be found in gender taxonomie. "
                is_current_costume_invalid = True
                if error not in found_errors:
                    found_errors.append(error)
                    output_string += error + " costume: " + str(costume) + "\n"

            if not age_impression.has_node(costume.dominant_age_impression):
                error = (
                    costume.dominant_age_impression
                    + " could not be found in age_impression taxonomie. "
                )
                is_current_costume_invalid = True
                if error not in found_errors:
                    found_errors.append(error)
                    output_string += error + " costume: " + str(costume) + "\n"

            for _genre in costume.genres:
                if not genre.has_node(_genre):
                    error = _genre + " could not be found in genre taxonomie. "
                    is_current_costume_invalid = True
                    is_current_costume_invalid = True
                    if error not in found_errors:
                        found_errors.append(error)
                        output_string += error + " costume: " + str(costume) + "\n"

            if is_current_costume_invalid:
                invalid_costumes += 1
                is_current_costume_invalid = False

        with open(file, "w+") as file_object:
            file_object.write(output_string)

        logging.info(str(invalid_costumes) + " / " + str(len(costumes)) + " are invalid")
        logging.info("Validation output has been written to " + file)

        return
