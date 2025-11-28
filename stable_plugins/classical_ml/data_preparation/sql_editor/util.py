import duckdb


def execute_sql(sql: str):
    with duckdb.connect(
        config={
            "python_enable_replacements": False,
            "allow_community_extensions": False,
            "allow_persistent_secrets": False,
            # "disabled_filesystems": "LocalFileSystem",
        }
    ) as con:
        con.install_extension("httpfs")
        con.load_extension("httpfs")
        result = con.sql(sql)
        print(result)
        print(result.fetchall())


if __name__ == "__main__":
    execute_sql("SELECT 42;")
