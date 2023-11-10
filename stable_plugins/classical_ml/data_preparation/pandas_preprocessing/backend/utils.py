def get_number_if_possible(s: str):
    if s.isdigit():
        return int(s)

    try:
        return float(s)
    except ValueError:
        pass

    return s
