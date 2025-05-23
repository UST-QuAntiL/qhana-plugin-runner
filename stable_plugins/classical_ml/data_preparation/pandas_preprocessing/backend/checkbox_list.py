from flask.templating import render_template


def get_checkbox_list_dict(dictionary: dict):
    checkbox_list_dict = {}
    for key, value in dictionary.items():
        checkbox_list_dict[key] = render_template(
            "pd_preprocessing_checkbox_list.html",
            list_content=[
                {"label": entry, "id": entry, "name": entry} for entry in value
            ],
        )
    return checkbox_list_dict
