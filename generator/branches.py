"""Branch master generator for the synthetic retail banking universe.

Defines the static branches across major metro and urban Indian locations.
"""

from datetime import date
import polars as pl


# List of pre-defined branches
BRANCH_DATA = [
    {
        "branch_code": "B001",
        "branch_name": "Mumbai Main",
        "city": "Mumbai",
        "state": "Maharashtra",
        "region": "West",
        "branch_type": "Metro",
        "open_date": date(2010, 1, 1),
        "closure_date": None,
    },
    {
        "branch_code": "B002",
        "branch_name": "Bengaluru MG Road",
        "city": "Bengaluru",
        "state": "Karnataka",
        "region": "South",
        "branch_type": "Metro",
        "open_date": date(2012, 6, 1),
        "closure_date": None,
    },
    {
        "branch_code": "B003",
        "branch_name": "Delhi Connaught Place",
        "city": "Delhi",
        "state": "Delhi",
        "region": "North",
        "branch_type": "Metro",
        "open_date": date(2011, 3, 1),
        "closure_date": None,
    },
    {
        "branch_code": "B004",
        "branch_name": "Kolkata Salt Lake",
        "city": "Kolkata",
        "state": "West Bengal",
        "region": "East",
        "branch_type": "Metro",
        "open_date": date(2013, 9, 1),
        "closure_date": None,
    },
    {
        "branch_code": "B005",
        "branch_name": "Chennai T-Nagar",
        "city": "Chennai",
        "state": "Tamil Nadu",
        "region": "South",
        "branch_type": "Metro",
        "open_date": date(2014, 7, 1),
        "closure_date": None,
    },
    {
        "branch_code": "B006",
        "branch_name": "Pune Shivaji Nagar",
        "city": "Pune",
        "state": "Maharashtra",
        "region": "West",
        "branch_type": "Urban",
        "open_date": date(2015, 2, 1),
        "closure_date": None,
    },
    {
        "branch_code": "B007",
        "branch_name": "Hyderabad Gachibowli",
        "city": "Hyderabad",
        "state": "Telangana",
        "region": "South",
        "branch_type": "Urban",
        "open_date": date(2016, 11, 1),
        "closure_date": None,
    },
    {
        "branch_code": "B008",
        "branch_name": "Ahmedabad Satellite",
        "city": "Ahmedabad",
        "state": "Gujarat",
        "region": "West",
        "branch_type": "Urban",
        "open_date": date(2017, 5, 1),
        "closure_date": None,
    },
    {
        "branch_code": "B009",
        "branch_name": "Lucknow Hazratganj",
        "city": "Lucknow",
        "state": "Uttar Pradesh",
        "region": "North",
        "branch_type": "Urban",
        "open_date": date(2018, 8, 1),
        "closure_date": None,
    },
    {
        "branch_code": "B010",
        "branch_name": "Jaipur C-Scheme",
        "city": "Jaipur",
        "state": "Rajasthan",
        "region": "West",
        "branch_type": "Urban",
        "open_date": date(2019, 10, 1),
        "closure_date": None,
    },
]


def generate_branches() -> pl.DataFrame:
    """Generates the static branch master DataFrame.

    Returns:
        pl.DataFrame: The branch master table.
    """
    return pl.DataFrame(
        BRANCH_DATA,
        schema={
            "branch_code": pl.String,
            "branch_name": pl.String,
            "city": pl.String,
            "state": pl.String,
            "region": pl.String,
            "branch_type": pl.String,
            "open_date": pl.Date,
            "closure_date": pl.Date,
        },
    )
