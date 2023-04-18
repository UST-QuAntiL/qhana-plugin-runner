import marshmallow as ma


def validate_float_in_interval_else_int(
    min: float, max: float,
    min_inclusive: bool = False, max_inclusive: bool = False
):
    def validation_function(value: float):
        value_in_interval = min < value and value < max or (value == min and min_inclusive) or (value == max and max_inclusive)
        if not value_in_interval:
            if int(value) - value != 0:
                interval_str = f"{'[' if min_inclusive else '('}{min}, {max}{']' if max_inclusive else ')'}"
                raise ma.ValidationError(f"Value must be an integer outside of the interval {interval_str}")

    return validation_function
