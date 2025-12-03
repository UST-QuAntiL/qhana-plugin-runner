import duckdb


def check_sql_syntax(sql: str) -> str | None:
    try:
        # workaround to only check SQL for syntax errors without executing it
        duckdb.extract_statements(sql)
    except duckdb.ParserException as err:
        return err.args[0]
    return None


def execute_sql(sql: str):
    with duckdb.connect(
        config={
            "python_enable_replacements": False,
            "allow_community_extensions": False,
            "allow_persistent_secrets": False,
            # "disabled_filesystems": "LocalFileSystem",
        }
    ) as con:
        # FIXME: https://duckdb.org/docs/stable/operations_manual/securing_duckdb/overview
        con.install_extension("httpfs")
        con.load_extension("httpfs")
        result = con.sql(sql)
        print(result)
        print(result.fetchall())


if __name__ == "__main__":
    print(check_sql_syntax("SELLLECT 42;"))
    execute_sql("SELECT 42;")
