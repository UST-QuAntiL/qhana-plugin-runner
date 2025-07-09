def format_complex_vectors(vectors):
    formatted_vectors = []
    for vec in vectors:
        fv = []
        for cnum in vec:
            fv.append([cnum.real, cnum.imag])
        formatted_vectors.append(fv)

    return formatted_vectors
