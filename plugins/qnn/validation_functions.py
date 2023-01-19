import marshmallow as ma


def validate_layer_input(value):
    if value != "":
        value = value.replace(" ", "")
        value = value.split(",")
        for entry in value:
            if not entry.isdigit():
                raise ma.ValidationError("Not numbers separated by commas.")
            entry = int(entry)
            if entry <= 0:
                raise ma.ValidationError("Numbers must be greater than 0.")
