# File: STR_b_01c_create_refGrid_eastMA.R
# Author: Sebastian Rowland <sr3463@cumc.columbia.edu>
# Date: 10/28/2021

##-----------------------##
#### Table of Contents ####
##-----------------------##

# N. Notes
# 0. Import Packages and Set Global Variables
# 1. Identify Grid Centroids within East MA Bounding Box
# 2. Identify Grid Centroids within CONUS-Inscribed Rectangle
# 3. Restrict Centroids to CONUS

#----------------#
#### N. NOTES ####
#----------------#

# This script takes about 16 minutes to run. 

#---------------------------------------------#
#### 0. PACKAGE IMPORTS & GLOBAL VARIABLES ####
#---------------------------------------------#

# 0a. Load packages and functions required for this script
if(!exists("Ran_a_00")){
  here::i_am("bne_eastMA_app.Rproj")
  source(here::here('code', '0_00_set_up_env.R'))
}

#--------------------------------------------------------------#
#### 1. IDENTIFY GRID CENTROIDS WITHIN EAST MA BOUNDING BOX ####
#--------------------------------------------------------------#

# 1a. make grid of all the possible coordinate combinations 
# we determine the size of the grids with the by argument; units are degrees
refGrid <- expand.grid(lat = seq(eastMA.bb$yMin, eastMA.bb$yMax, by = 0.01), 
                   lon = seq(eastMA.bb$xMin, eastMA.bb$xMax, by = 0.01))

# 1b. convert to spatial (simple feature)
refGrid <- refGrid %>% 
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4326"))

#----------------------------------------#
#### 2. RESTRICT POINTS TO EASTERN MA ####
#----------------------------------------#

# 2a. load the base map from our shapefile
usa <- sf::st_read(here::here('data', 'ancillary', 'cb_2015_us_state_500k', 
                              'cb_2015_us_state_500k.shp')) %>% 
  st_transform(crs = st_crs("epsg:4326"))

# 2b. keep states in Eastern MA area
eastMAstates <- c('MA', 'RI', 'CT', 'NY', 'NH', 'VT')
eastMA <- usa %>% 
  dplyr::filter(STUSPS %in% eastMAstates) %>% 
  dplyr::select(NAME, STUSPS)

# 2c. intersect points with eastern MA states
refGrid.eastMA <- refGrid %>% 
  sf::st_join(eastMA, join = st_within) %>% 
  dplyr::filter(!is.na(NAME)) %>% 
  sf::st_transform(crs=st_crs("epsg:4326"))

# 2d. confirm success
#plot(refGrid.eastMA)

#---------------#
#### 3. Save ####
#---------------#

# 3a. convert to dataframe
refGrid.eastMA <- refGrid.eastMA %>% 
  dplyr::mutate(lon = st_coordinates(refGrid.eastMA)[,1], 
         lat = st_coordinates(refGrid.eastMA)[,2]) %>% 
  as.data.frame() %>% 
  dplyr::select(-geometry)

# 3b. plot to confirm success
refGrid.eastMA %>% 
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4326")) %>% 
  plot()

# 3c. save the points
refGrid.eastMA %>% 
  dplyr::select(lat, lon) %>%
  fst::write_fst(here::here('data', 'ancillary',  
                            'refGrid_eastMA.fst'))

