"""SQL snippets shared by the pipeline scripts."""


def haversine_km(lon_col, lat_col, lon0, lat0):
    """Great-circle distance in km from (lon0, lat0) to the given columns.

    Coordinates are parenthesised: a bare negative longitude would otherwise
    render as `longitude--95.12`, and `--` opens a SQL comment.
    """
    return (f"2*6371*asin(sqrt(pow(sin(radians({lat_col}-({lat0}))/2),2)"
            f" + cos(radians(({lat0})))*cos(radians({lat_col}))"
            f"*pow(sin(radians({lon_col}-({lon0}))/2),2)))")
