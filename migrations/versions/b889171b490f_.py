"""empty message

Revision ID: b889171b490f
Revises: a09e03db46a1
Create Date: 2023-11-09 12:33:37.697251

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b889171b490f"
down_revision = "a09e03db46a1"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column("DataBlob", "id")
    op.add_column("DataBlob", sa.Column("key", sa.String(500), primary_key=True))
    op.create_primary_key("pk_DataBlob", "DataBlob", ["plugin_id", "key"])


def downgrade():
    op.drop_column("DataBlob", "key")
    op.add_column("DataBlob", sa.Column("id", sa.Integer(), primary_key=True))
    op.create_primary_key("pk_DataBlob", "DataBlob", ["plugin_id", "id"])
