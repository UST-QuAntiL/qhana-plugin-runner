"""Change id column in Table DataBlob to a key column for text based primary keys.

Revision ID: b889171b490f
Revises: a09e03db46a1
Create Date: 2023-11-09 12:33:37.697251

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b889171b490f"
down_revision = "a09e03db46a1"
branch_labels = None
depends_on = None


def upgrade():
    # drop and recreate table in cse some data is already present
    # should not be the case before this migration
    op.drop_table("DataBlob")
    op.create_table(
        "DataBlob",
        sa.Column("plugin_id", sa.String(length=550), nullable=False, primary_key=True),
        sa.Column("key", sa.String(500), primary_key=True),
        sa.Column("value", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint("plugin_id", "key", name=op.f("pk_DataBlob")),
    )


def downgrade():
    # drop new table and recreate old version
    op.drop_table("DataBlob")
    op.create_table(
        "DataBlob",
        sa.Column("id", sa.INTEGER(), nullable=False),
        sa.Column("plugin_id", sa.String(length=550), nullable=False),
        sa.Column("value", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint("id", "plugin_id", name=op.f("pk_DataBlob")),
    )
