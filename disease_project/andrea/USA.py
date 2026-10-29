import geopandas as gpd
import numpy as np
import polars as pl

_jan_df = pl.read_csv("../data/USA/weekly_county2county_2019_01_07.csv")
_feb_df = pl.read_csv("../data/USA/weekly_county2county_2019_02_04.csv")
_mar_df = pl.read_csv("../data/USA/weekly_county2county_2019_03_04.csv")
_apr_df = pl.read_csv("../data/USA/weekly_county2county_2019_04_01.csv")
_may_df = pl.read_csv("../data/USA/weekly_county2county_2019_05_06.csv")
_jun_df = pl.read_csv("../data/USA/weekly_county2county_2019_06_03.csv")
_jul_df = pl.read_csv("../data/USA/weekly_county2county_2019_07_01.csv")

_trips_df = pl.concat([_jan_df, _feb_df, _mar_df, _apr_df, _may_df, _jun_df, _jul_df])
# remove columns "lng_o", "lat_o", "lng_d", "lat_d", "date_range", "visitor_flows"
_trips_df = _trips_df.drop(
    ["lng_o", "lat_o", "lng_d", "lat_d", "date_range", "visitor_flows"]
)
# drop rows where geoid_o and geoid_d are the same
_trips_df = _trips_df.filter(_trips_df["geoid_o"] != _trips_df["geoid_d"])
# accumulate the "pop_flows" column by matching "geoid_o" and "geoid_d"
_trips_df = _trips_df.group_by(["geoid_o", "geoid_d"]).agg(
    pl.sum("pop_flows").alias("pop_flows")
)
# divide pop_flows by 7*7 to get daily average
_trips_df = _trips_df.with_columns((pl.col("pop_flows") / (7 * 7)).alias("pop_flows"))

_unique_geoids = pl.concat([_trips_df["geoid_o"], _trips_df["geoid_d"]]).unique()

_population_df = pl.read_csv("../data/USA/counties.csv")
# drop all columns except "geoid" and "population"
_population_df = _population_df.select(["geoid", "population"])
# filter by unique_geoids
_population_df = _population_df.filter(pl.col("geoid").is_in(_unique_geoids.implode()))

_unique_geoids = _population_df["geoid"].unique()

# filter trips_df by unique_geoids
_trips_df = _trips_df.filter(
    pl.col("geoid_o").is_in(_unique_geoids.implode())
    & pl.col("geoid_d").is_in(_unique_geoids.implode())
)

_unique_geoids_trips = pl.concat([_trips_df["geoid_o"], _trips_df["geoid_d"]]).unique()
_unique_geoids_population = _population_df["geoid"].unique()
assert (
    _unique_geoids_trips.sort().to_list() == _unique_geoids_population.sort().to_list()
), "Unique geoids in trips_df and population_df do not match"
_unique_geoids = _unique_geoids_trips.sort()

_county_to_index = {geoid: i for i, geoid in enumerate(_unique_geoids.to_list())}

_od_matrix = np.zeros((len(_unique_geoids), len(_unique_geoids)))
for _row in _trips_df.iter_rows():
    _geoid_o = _row[0]
    _geoid_d = _row[1]
    _pop_flows = _row[2]
    _idx_o = _county_to_index[_geoid_o]
    _idx_d = _county_to_index[_geoid_d]
    _pop_o = _population_df[_idx_o, 1]
    _pop_d = _population_df[_idx_d, 1]
    _od_matrix[_idx_d, _idx_o] += 0.5 * _pop_flows / _pop_o
    _od_matrix[_idx_o, _idx_d] += 0.5 * _pop_flows / _pop_d

_gdf = gpd.read_file("../data/USA/cb_2025_us_county_20m/cb_2025_us_county_20m.shp")
# filter by
# us_counties_filtered = us_counties[us_counties["GEOID"].astype(int).isin(unique_geoids.astype(int))]
_gdf = _gdf[_gdf["GEOID"].astype(int).isin(_unique_geoids.to_numpy().astype(int))]
# sort by GEOID
_gdf = _gdf.sort_values("GEOID")


def get_od_matrix():
    return _od_matrix.copy()


def make_initial_condition(
    *,
    initial_county: int | None = None,
    initial_vaccination_rate: float = 0.0,
    random_seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)

    if initial_county is not None:
        initial_infected_county = _county_to_index[initial_county]
    else:
        initial_infected_county = rng.choice(len(_unique_geoids))

    S0 = _population_df[:, 1].to_numpy().astype(float)
    E0 = np.zeros_like(S0)
    I0 = np.zeros_like(S0)
    I0[initial_infected_county] = 1
    R0 = initial_vaccination_rate * S0
    S0 -= R0 + I0
    return np.concatenate([S0, E0, I0, R0])


def get_gdf():
    return _gdf.copy()
