from flask.templating import render_template


def get_checkbox_list_dict(tables_and_columns: dict):
    checkbox_list_dict = {}
    for key, value in tables_and_columns.items():
        checkbox_list_dict[key] = render_template(
            "sql_loader_checkbox_list.html",
            list_content=[
                {"label": entry, "id": entry, "name": entry} for entry in value
            ],
        )
    return checkbox_list_dict
