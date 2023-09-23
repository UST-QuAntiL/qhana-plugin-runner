def get_number_if_possible(s: str):
    if s.isdigit():
        return int(s)

    try:
        temp = float(s)
        return temp
    except ValueError:
        pass
        
    return s
