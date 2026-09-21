MIMETYPES = {
    "csv": "text/csv",
    "json": "application/json",
    "lines": "application/X-lines+json",
}

EXTENSIONS = {
    "csv": "csv",
    "json": "json",
    "lines": "jsonl",
}

TEST_DATA = {
    "a.csv": "ID,href,dim0,dim1\ne1,h1,1,2",
    "a.json": '[{"ID":"e1","href":"h1","dim0":1,"dim1":2}]',
    "a_lines.json": '{"ID":"e1","href":"h1","dim0":1,"dim1":2}',
    "b.csv": "ID,href,dim0,dim1\ne1,h1,3,4",
    "b.json": '[{"ID":"e1","href":"h1","dim0":3,"dim1":4}]',
    "b_lines.json": '{"ID":"e1","href":"h1","dim0":3,"dim1":4}',
    "ragged.json": '[{"ID":"e1","href":"h1","dim0":1,"dim1":2},{"ID":"e2","href":"h2","dim0":3}]',
    "single.csv": "ID,href,dim0\ne1,h1,5",
    "single.json": '[{"ID":"e1","href":"h1","dim0":5}]',
    "single_lines.json": '{"ID":"e1","href":"h1","dim0":5}',
}
