import zipfile

import geopandas as gpd
import numpy as np
import polars as pl

try:
    with zipfile.ZipFile("../data/trip_count/trip_count_matrix_county_by_age_activity_final_2024.zip", 'r') as zip_ref:
        zip_ref.extractall("../data/trip_count/")

    with zipfile.ZipFile("../data/vg250/DE_VG250.zip", 'r') as zip_ref:
        zip_ref.extractall("../data/vg250/")
except Exception as e:
    print(f"Error extracting zip files: {e}")

_trips_df = pl.read_csv("../data/trip_count/trip_count_matrix_county_by_age_activity_final_2024.csv")
_trips_df = _trips_df.group_by(["start_county", "end_county"]).agg(pl.sum("trip_cnt").alias("trip_cnt"))

population_df = pl.read_csv("../data/population/12411-0016_de.csv", separator=";", skip_rows=5)
population_df = population_df.select([pl.col("_duplicated_0").alias("county"),
                                     pl.col("_duplicated_1").alias("county_name"),
                                     pl.col("Insgesamt").alias("population")])
population_df = population_df.drop_nulls()
population_df = population_df.filter(pl.col("population") != "-")
population_df = population_df.with_columns(pl.col("population").cast(pl.Int64))
population_df = population_df.sort("county")

# sorted list of counties
_start_counties = sorted(_trips_df["start_county"].unique())
_end_counties = sorted(_trips_df["end_county"].unique())
assert _start_counties == _end_counties, "Start and end counties do not match"
# create map county id -> index
county_to_index = {county: i for i, county in enumerate(_start_counties)}

OD_matrix = np.zeros((len(_start_counties), len(_end_counties)))
for row in _trips_df.iter_rows():
    start_county, end_county, trip_cnt = row
    if start_county == end_county:
        continue
    i = county_to_index[start_county]
    j = county_to_index[end_county]
    start_county_population = population_df[i]["population"][0]
    end_county_population = population_df[j]["population"][0]
    OD_matrix[j, i] += 0.5 * trip_cnt / start_county_population
    OD_matrix[i, j] += 0.5 * trip_cnt / end_county_population

gdf = gpd.read_file("../data/vg250/DE_VG250.gpkg", layer="vg250_krs")
gdf = gdf.iloc[:401]
gdf = gdf[gdf["AGS"].astype(int).isin([int(c) for c in _start_counties])]

def make_initial_condition(*,
                           initial_county: int | None = None,
                           initial_vaccination_rate: float = 0.0,
                           random_seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(random_seed)

    if initial_county is not None:
        initial_infected_county = county_to_index[initial_county]
    else:
        initial_infected_county = rng.choice(len(_start_counties))

    S0 = population_df["population"].to_numpy().astype(float)
    I0 = np.zeros_like(S0)
    I0[initial_infected_county] = 1.0
    R0 = initial_vaccination_rate * S0
    S0 -= I0 + R0
    return np.concatenate((S0, I0, R0))
