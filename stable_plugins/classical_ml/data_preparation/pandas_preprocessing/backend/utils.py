def get_number_if_possible(s: str):
    if s.isdigit():
        return int(s)

    temp = None
    try:
        temp = float(s)
    except ValueError:
        pass

    if temp is not None:
        s = temp

    return s
