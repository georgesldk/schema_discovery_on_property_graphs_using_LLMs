def detect_file_role(df):
    """
    Determines if a CSV represents a Node list or an Edge list.

    Logic is based on column name heuristics:
    - Edge file: has both START and END identifiers
    - Node file: has an ID column
    """
    cols = df.columns.tolist()

    start_col = next(
        (c for c in cols if ':START_ID' in c or 'source' in c.lower()),
        None
    )
    end_col = next(
        (c for c in cols if ':END_ID' in c or 'target' in c.lower()),
        None
    )
    id_col = next(
        (c for c in cols if ':ID' in c or ('id' in c.lower() and not start_col)),
        None
    )

    if start_col and end_col:
        return 'edge', {'start': start_col, 'end': end_col}
    elif id_col:
        return 'node', {'id': id_col}
    else:
        return 'unknown', {}
