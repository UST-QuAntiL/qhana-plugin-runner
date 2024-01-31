"""Add virtual Plugins

Revision ID: a09e03db46a1
Revises: ae51830d0cb5
Create Date: 2023-05-05 05:48:10.261809

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = "a09e03db46a1"
down_revision = "ae51830d0cb5"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "DataBlob",
        sa.Column("id", sa.INTEGER(), nullable=False),
        sa.Column("plugin_id", sa.String(length=550), nullable=False),
        sa.Column("value", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint("id", "plugin_id", name=op.f("pk_DataBlob")),
    )
    op.create_table(
        "PluginState",
        sa.Column("plugin_id", sa.String(length=550), nullable=False),
        sa.Column("key", sa.String(length=500), nullable=False),
        sa.Column("value", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("plugin_id", "key", name=op.f("pk_PluginState")),
    )
    op.create_table(
        "VirtualPlugin",
        sa.Column("id", sa.INTEGER(), nullable=False),
        sa.Column("parent_id", sa.String(length=550), nullable=False),
        sa.Column("name", sa.String(length=500), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("tags", sa.Text(), nullable=False),
        sa.Column("href", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_VirtualPlugin")),
    )

    op.drop_table("TaskData")


def downgrade():
    op.create_table(
        "TaskData",
        sa.Column("id", sa.INTEGER(), nullable=False),
        sa.Column("key", sa.VARCHAR(length=500), nullable=False),
        sa.Column("value", sqlite.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["id"],
            ["ProcessingTask.id"],
        ),
        sa.PrimaryKeyConstraint("id", "key"),
    )
    op.drop_table("VirtualPlugin")
    op.drop_table("PluginState")
    op.drop_table("DataBlob")
