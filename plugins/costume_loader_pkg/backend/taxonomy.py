import logging
from enum import Enum
from typing import List, Dict
from typing import Tuple

from sqlalchemy import text
from sqlalchemy.sql.elements import TextClause

from plugins.costume_loader_pkg.backend.database import Database


class TaxonomyType(Enum):
    """
    Represents a taxonomy type, i.e. all possible
    taxonomies in the database. This should not be
    confused with the Attribute, which describes
    any special property of an entity with which
    is comparable.
    """

    alterseindruck = "alterseindruck"
    basiselement = "basiselement"
    charaktereigenschaft = "charaktereigenschaft"
    design = "design"
    farbeindruck = "farbeindruck"
    farbe = "farbe"
    farbkonzept = "farbkonzept"
    form = "form"
    funktion = "funktion"
    genre = "genre"
    koerpermodifikation = "koerpermodifikation"
    koerperteil = "koerperteil"
    material = "material"
    materialeindruck = "materialeindruck"
    operator = "operator"
    produktionsort = "produktionsort"
    rollenberuf = "rollenberuf"
    spielortDetail = "spielortDetail"
    spielort = "spielort"
    spielzeit = "spielzeit"
    stereotyp = "stereotyp"
    tageszeit = "tageszeit"
    teilelement = "teilelement"
    trageweise = "trageweise"
    typus = "typus"
    zustand = "zustand"
    geschlecht = "geschlecht"
    ortsbegebenheit = "ortsbegebenheit"
    stereotypRelevant = "stereotypRelevant"
    rollenrelevanz = "rollenrelevanz"
    familienstand = "familienstand"

    @staticmethod
    def get_name(taxonomy_type) -> str:
        if taxonomy_type == TaxonomyType.alterseindruck:
            return "Alterseindruck"
        elif taxonomy_type == TaxonomyType.basiselement:
            return "Basiselement"
        elif taxonomy_type == TaxonomyType.charaktereigenschaft:
            return "Charaktereigenschaft"
        elif taxonomy_type == TaxonomyType.design:
            return "Design"
        elif taxonomy_type == TaxonomyType.farbeindruck:
            return "Farbeindruck"
        elif taxonomy_type == TaxonomyType.farbe:
            return "Farbe"
        elif taxonomy_type == TaxonomyType.farbkonzept:
            return "Farbkonzept"
        elif taxonomy_type == TaxonomyType.form:
            return "Form"
        elif taxonomy_type == TaxonomyType.funktion:
            return "Funktion"
        elif taxonomy_type == TaxonomyType.genre:
            return "Genre"
        elif taxonomy_type == TaxonomyType.koerpermodifikation:
            return "Körpermodifikation"
        elif taxonomy_type == TaxonomyType.koerperteil:
            return "Köerperteil"
        elif taxonomy_type == TaxonomyType.material:
            return "Material"
        elif taxonomy_type == TaxonomyType.materialeindruck:
            return "Materialeindruck"
        elif taxonomy_type == TaxonomyType.operator:
            return "Operator"
        elif taxonomy_type == TaxonomyType.produktionsort:
            return "Produktionsort"
        elif taxonomy_type == TaxonomyType.rollenberuf:
            return "Rollenberuf"
        elif taxonomy_type == TaxonomyType.spielortDetail:
            return "SpielortDetail"
        elif taxonomy_type == TaxonomyType.spielort:
            return "Spielort"
        elif taxonomy_type == TaxonomyType.spielzeit:
            return "Spielzeit"
        elif taxonomy_type == TaxonomyType.stereotyp:
            return "Stereotyp"
        elif taxonomy_type == TaxonomyType.tageszeit:
            return "Tageszeit"
        elif taxonomy_type == TaxonomyType.teilelement:
            return "Teilelement"
        elif taxonomy_type == TaxonomyType.trageweise:
            return "Trageweise"
        elif taxonomy_type == TaxonomyType.typus:
            return "Typus"
        elif taxonomy_type == TaxonomyType.zustand:
            return "Zustand"
        elif taxonomy_type == TaxonomyType.geschlecht:
            return "Geschlecht"
        elif taxonomy_type == TaxonomyType.ortsbegebenheit:
            return "Ortsbegebenheit"
        elif taxonomy_type == TaxonomyType.stereotypRelevant:
            return "StereotypRelevant"
        elif taxonomy_type == TaxonomyType.rollenrelevanz:
            return "Rollenrelevanz"
        elif taxonomy_type == TaxonomyType.familienstand:
            return "Familienstand"
        else:
            logging.error(
                'No name for taxonomyType "' + str(taxonomy_type) + '" specified'
            )
            raise ValueError(
                'No name for taxonomyType "' + str(taxonomy_type) + '" specified'
            )

    @staticmethod
    def get_database_table_name(taxonomy_type) -> str:
        if taxonomy_type == taxonomy_type.alterseindruck:
            return "AlterseindruckDomaene"
        elif taxonomy_type == taxonomy_type.basiselement:
            return "BasiselementDomaene"
        elif taxonomy_type == taxonomy_type.charaktereigenschaft:
            return "CharaktereigenschaftsDomaene"
        elif taxonomy_type == taxonomy_type.design:
            return "DesignDomaene"
        elif taxonomy_type == taxonomy_type.farbeindruck:
            return None
        elif taxonomy_type == taxonomy_type.farbe:
            return "FarbenDomaene"
        elif taxonomy_type == taxonomy_type.farbkonzept:
            return "FarbkonzeptDomaene"
        elif taxonomy_type == taxonomy_type.form:
            return "FormenDomaene"
        elif taxonomy_type == taxonomy_type.funktion:
            return "FunktionsDomaene"
        elif taxonomy_type == taxonomy_type.genre:
            return "GenreDomaene"
        elif taxonomy_type == taxonomy_type.koerpermodifikation:
            return "KoerpermodifikationsDomaene"
        elif taxonomy_type == taxonomy_type.koerperteil:
            return None
        elif taxonomy_type == taxonomy_type.material:
            return "MaterialDomaene"
        elif taxonomy_type == taxonomy_type.materialeindruck:
            return None
        elif taxonomy_type == taxonomy_type.operator:
            return "OperatorDomaene"
        elif taxonomy_type == taxonomy_type.produktionsort:
            return "ProduktionsortDomaene"
        elif taxonomy_type == taxonomy_type.rollenberuf:
            return "RollenberufDomaene"
        elif taxonomy_type == taxonomy_type.spielortDetail:
            return "SpielortDetailDomaene"
        elif taxonomy_type == taxonomy_type.spielort:
            return "SpielortDomaene"
        elif taxonomy_type == taxonomy_type.spielzeit:
            return "SpielzeitDomaene"
        elif taxonomy_type == taxonomy_type.stereotyp:
            return "StereotypDomaene"
        elif taxonomy_type == taxonomy_type.tageszeit:
            return "TageszeitDomaene"
        elif taxonomy_type == taxonomy_type.teilelement:
            return "TeilelementDomaene"
        elif taxonomy_type == taxonomy_type.trageweise:
            return "TrageweisenDomaene"
        elif taxonomy_type == taxonomy_type.typus:
            return "TypusDomaene"
        elif taxonomy_type == taxonomy_type.zustand:
            return "ZustandsDomaene"
        elif taxonomy_type == taxonomy_type.geschlecht:
            return None
        elif taxonomy_type == taxonomy_type.ortsbegebenheit:
            return None
        elif taxonomy_type == taxonomy_type.stereotypRelevant:
            return None
        elif taxonomy_type == taxonomy_type.rollenrelevanz:
            return None
        elif taxonomy_type == taxonomy_type.familienstand:
            return None
        else:
            logging.error(
                'No name for taxonomy_type "' + str(taxonomy_type) + '" specified'
            )
            raise ValueError(
                'No name for taxonomy_type "' + str(taxonomy_type) + '" specified'
            )


class Taxonomy:
    @staticmethod
    def create_from_db(taxonomy_type: TaxonomyType, db: Database) -> Dict:
        """
        Returns the entities and relations,
        given the name of the taxonomy table
        """
        name = taxonomy_type.get_database_table_name(taxonomy_type)

        entities = set()
        relations = []
        # Format is (child, parent)
        rows = Taxonomy.__get_taxonomy_table(name, db)

        root_node = None

        for row in rows:
            child = row[0]
            parent = row[1]

            entities.add(child)

            # If parent is none, this is the root node
            if parent is None or parent == "":
                root_node = child

            relations.append({"source": parent, "target": child})

        if root_node is None:
            logging.error("No root node found in taxonomy")
            raise Exception("No root node found in taxonomy")

        return {
            "GRAPH_ID": "tax_" + name,
            "type": "tree",
            "ref-target": "entities.json",
            "entities": list(entities),
            "relations": relations,
        }

    @staticmethod
    def __get_taxonomy_table(name: str, database: Database) -> List[Tuple[str, str]]:
        """
        Returns the table of a taxonomy.
        In the kostuemrepo a taxonomie table is a "domaene" table with
        the structure "Child", "Parent"
        """
        query: TextClause = text("SELECT * FROM " + name)
        rows: List[Tuple[str, str]] = database.session.execute(query).fetchall()

        return rows


def main():
    db = Database()
    db.open_with_params(
        host="localhost",
        user="test",
        password="test",
        database="KostuemRepo",
    )

    taxonomy = Taxonomy.create_from_db(TaxonomyType.farbe, db)


if __name__ == "__main__":
    main()
