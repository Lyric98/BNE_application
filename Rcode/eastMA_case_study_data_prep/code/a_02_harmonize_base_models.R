# File: STR_b_01c_create_refGrid_eastMA.R
# Author: Sebastian Rowland <sr3463@cumc.columbia.edu>
# Date: 10/27/2021
#
# Contents:
# N. Notes
# 0. Import Packages and Set Global Variables
# 1. Read Reference Grid
# 2. Join AV Base Model
# 3. Join GBD Base Model
# 4. Join CACES Base Model
# 5. Save

#----------------#
#### N: NOTES ####
#----------------#

# This script takes about 16 minutes to run. 

#---------------------------------------------------#
#### 0: IMPORT PACKAGES AND SET GLOBAL VARIABLES ####
#---------------------------------------------------#

# 0a. Load packages and functions required for this script
if(!exists("Ran_a_00")){
  here::i_am("bne_eastMA_app.Rproj")
  source(here::here('code', '0_00_set_up_env.R'))
}

#------------------------------#
#### 1. READ REFERENCE GRID ####
#------------------------------#

# 1a. bring in the reference grid 
refGrid.eastMA <- fst::read_fst(here::here('data', 'ancillary',  
                            'refGrid_eastMA.fst'))

# 1b. make spatial 
refGrid.eastMA <- refGrid.eastMA %>% 
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4326")) %>% 
  sf::st_transform(crs = st_crs("epsg:4326"))

#-----------------------------#
#### 2. JOIN AV BASE MODEL ####
#-----------------------------#

# 2a. read in the raster
av2011 <- raster::raster(here::here('data', 'base_models', 'raw',
                                       'V4NA03_PM25_NA_201101_201112-RH35.nc'))

# 2b. keep only data within Eastern MA area
eastMAExtent <- raster::extent(c(eastMA.bb$xMin, eastMA.bb$xMax,
                                 eastMA.bb$yMin, eastMA.bb$yMax))
av2011.eastMA <- raster::crop(av2011, eastMAExtent)

plot(av2011.eastMA)

# 2c. extract values that correspond to points
refGrid.eastMA.sp <- refGrid.eastMA %>% 
  sf::as_Spatial()

refGrid.eastMA$pred_av <- raster::extract(av2011, refGrid.eastMA.sp)

# 2d. plot to confirm success
plot(refGrid.eastMA)

#------------------------------#
#### 3. JOIN GBD BASE MODEL ####
#------------------------------#

# 3a. load data 
load(here::here('data', 'base_models', 'raw', 'GBD2016_PREDPOP_FINAL.rdta'))  
# note that GS has extra columns and allows us to explicitly filter by country

# 3b. filter to area of interest 
gbd2011 <- mydata2 %>%
  tibble::as_tibble() %>%
  dplyr::filter(Country == "USA") %>%
  dplyr::select(!(tidyselect::contains("log") | tidyselect::contains("pop"))) %>%
  tidyr::pivot_longer(cols = tidyselect::contains("_PM2.5_"), 
                      names_to = c("measurement", "year"),
                      names_pattern = "(.*)_PM2.5_(.*)",
                      values_to = "pred_gs") %>%
  dplyr::rename(lat = Latitude, lon = Longitude) %>%
  dplyr::select(lat, lon, measurement, pred_gs) %>%
  dplyr::filter(measurement == "Mean") %>%
  dplyr::select(-measurement) %>% 
  dplyr::filter(lat > eastMA.bb$yMin & lat < eastMA.bb$yMax & 
                  lon > eastMA.bb$xMin & lon < eastMA.bb$xMax)

# 3c. convert to simple features 
gbd2011 <- gbd2011 %>% 
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4326")) 

# 3d. spatial join by nearest neighbor
refGrid.eastMA$index_gbd <- nngeo::st_nn(refGrid.eastMA, gbd2011, k = 1) %>% unlist()
gbd2011 <- gbd2011 %>% 
  dplyr::mutate(index_gbd = row_number()) %>% 
  as.data.frame() %>% 
  dplyr::select(-geometry)
refGrid.eastMA <- refGrid.eastMA %>% 
  dplyr::inner_join(gbd2011, by = 'index_gbd')

#--------------------------------#
#### 4. JOIN CACES BASE MODEL ####
#--------------------------------#

# 4a. load data
caces2011 <- readr::read_csv(here::here('data', 'base_models', 'raw', 
                                        'CACES_annual_2011_blockGrp_raw.csv'), 
                             col_types = "cccdcdd") %>%
  dplyr::rename(pred_caces = pred_wght) %>%
  dplyr::select(lat, lon, pred_caces)

# 4b. restrict to eastern MA area
caces2011 <- caces2011 %>% 
  dplyr::filter(lat > eastMA.bb$yMin & lat < eastMA.bb$yMax & 
                  lon > eastMA.bb$xMin & lon < eastMA.bb$xMax)

# 4c. convert to simple features
caces2011 <- caces2011 %>% 
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4326")) 

# 4d. spatial join by nearest neighbor
refGrid.eastMA$index_caces <- nngeo::st_nn(refGrid.eastMA, caces2011, k = 1) %>% unlist()
caces2011 <- caces2011 %>% 
  dplyr::mutate(index_caces = row_number()) %>% 
  as.data.frame() %>% 
  dplyr::select(-geometry)
refGrid.eastMA <- refGrid.eastMA %>% 
  dplyr::inner_join(caces2011, by = 'index_caces')

#---------------#
#### 5. SAVE ####
#---------------#

# 5a. convert to dataframe
refGrid.eastMA <- refGrid.eastMA %>% 
  dplyr::mutate(lon = st_coordinates(refGrid.eastMA)[,1], 
                lat = st_coordinates(refGrid.eastMA)[,2]) %>% 
  as.data.frame() %>% 
  dplyr::select(-geometry, -contains('index'))

# 5b. plot to confirm success
#dta.eastMA %>% 
# sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4269")) %>% 
#  plot()

# 5b. save the points
refGrid.eastMA %>% 
  readr::write_csv(here::here('data', 'prediction_dataset',  
                            'base_model_predictions_eastMA.csv'))
