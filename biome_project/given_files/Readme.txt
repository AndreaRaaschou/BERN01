All data is collected into the data folder. 

1. There are five climate variables which are daily average values during 1961-1990. They are:
Tswrf (Total shortwave radiation flux, W m-2), 
Pre (Precipitation, mm day-1), 
Tmp (Daily mean temperature, K), 
Tmax (Daily maximum temperature, K)
Tmin (Daily minimum temperature, K). 

All of these variables have an accompanying file with the ending '_stats', which contains aggregated statistics of the climate variables, for instance, seasonal mean, median and standard deviation.


2. In the file LPJ-GUESS_output BERN1.csv, the following variables are collected:
NPP: net primary productivity (kg C m-2 year-1)
SoilR: soil respiration (kg C m-2 year-1)
MaxBiomeCmass: The maximum biomass from a single biome (kg C m-2)
MxbiomeLAI: The maximum leaf area index from a single biome (unitless)
VegC: Vegetation carbon poo (kg C m-2)l
LitterC: Litter carbon pool (kg C m-2)
SoilC: Soil carbon pool (kg C m-2)
Biome_Cmass: The biome type based on the maximum biomass (category)
Biome_LAI: The biome type based on the maximum LAI (category)
Biome_obs: The observed biome type (category)

3. soilmap.dat: the texture inforamtion for each grid cell (Percentage). 

4. gridlist_pan_gfed_ISO3_UN_ecoregion.txt
This file contains gridded data for countries, ecoregions, regions used by 
GFED (Global Fire emissions database) and carbon sink studes (Pan et al., 2011: doi:10.1126/science.1201609)


In addition, there is a legend of the biomes so that you can figure out which number is associated with what biome.
This file contains both the number, names and a suggested RGB colour for this biome.
