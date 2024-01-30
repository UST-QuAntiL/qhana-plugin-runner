from flask import url_for

PLACEHOLDER = -4242424242


def url_for_ie(endpoint):
    return url_for(endpoint=endpoint, _external=True, db_id=PLACEHOLDER)


def ie_replace_task_id(url, task_id):
    # make it more secure the placeholder has to be between two slashes
    return url.replace(f"/{PLACEHOLDER}/", f"/{task_id}/")
