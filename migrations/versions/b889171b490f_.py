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
    with op.batch_alter_table("DataBlob") as batch_op:
        batch_op.drop_column("id")
        batch_op.add_column(sa.Column("key", sa.String(500), primary_key=True))
        batch_op.create_primary_key("pk_DataBlob", ["plugin_id", "key"])


def downgrade():
    with op.batch_alter_table("DataBlob") as batch_op:
        batch_op.drop_column("key")
        batch_op.add_column(sa.Column("id", sa.Integer(), primary_key=True))
        batch_op.create_primary_key("pk_DataBlob", ["plugin_id", "id"])
