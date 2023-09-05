from typing import List
from flask.templating import render_template


def get_optionlist(options: List[str]) -> str:
    return render_template(
        "optionlist_template.html",
        options=options,
    )
