# File: STR_b_01_process_AQS_data.R
# Author: Sebastian Rowland <sr3463@cumc.columbia.edu>
# Date: 10/28/2021
#
# Contents:
# N. Notes
# 0. Import Packages and Set Global Variables
# 1. Read Dataset of Annual Monitors 
# 2. Curate Monitors 
# 3. Save

#----------------#
#### N. NOTES ####
#----------------#



#---------------------------------------------------#
#### 0: IMPORT PACKAGES AND SET GLOBAL VARIABLES ####
#---------------------------------------------------#

# 0a Load packages and functions required for this script
if(!exists("Ran_a_00")){
  here::i_am("bne_eastMA_app.Rproj")
  source(here::here('code',  "0_00_set_up_env.R"))
}

#### Do not run 
# remove unused variables from aqs data to save memory 

readr::read_csv(here::here('data', 'ground_truth', 'raw', 
                                  'daily_88101_2011.csv')) %>% 
  dplyr::select(Latitude, Longitude, 'Arithmetic Mean', 'Datum', 
                'State Name', 'County Name', 'Site Num', 'Parameter Code', 'POC') %>% 
  readr::write_csv(here::here('data', 'ground_truth', 'raw', 
                              'daily_88501_2011_selected_var.csv'))
readr::read_csv(here::here('data', 'ground_truth', 'raw', 
                           'daily_88502_2011.csv')) %>% 
  dplyr::select(Latitude, Longitude, 'Arithmetic Mean', 'Datum', 
                'State Name', 'County Name', 'Site Num', 'Parameter Code', 'POC') %>% 
  readr::write_csv(here::here('data', 'ground_truth', 'raw', 
                              'daily_88502_2011_selected_var.csv'))


#------------------------------------------#
#### 1. READ DATASET OF ANNUAL MONITORS ####
#------------------------------------------#

# 1a. read the aqs data 
# downloaded from https://aqs.epa.gov/aqsweb/airdata/download_files.html
aqs <- readr::read_csv(here::here('data', 'ground_truth', 'raw', 
                           'daily_88101_2011_selected_var.csv')) %>% 
  dplyr::bind_rows(read_csv(here::here('data', 'ground_truth', 'raw', 
                           'daily_88502_2011_selected_var.csv')))
# 1b. clean up the names 
aqs <- aqs %>% 
  janitor::clean_names() %>% 
  dplyr::rename(lat = latitude, lon = longitude, aqs = arithmetic_mean) 

# 1c. fix the datums 
aqs.nad <- aqs %>% 
  dplyr::filter(datum == 'NAD83') %>% 
  sf::st_as_sf(., coords = c("lon", "lat"),
           crs=st_crs("epsg:4269")) %>% 
  sf::st_transform(crs=st_crs("epsg:4326"))

aqs.wgs <- aqs %>% 
  dplyr::filter(datum == 'WGS84') %>% 
  sf::st_as_sf(., coords = c("lon", "lat"),
           crs=st_crs("epsg:4326")) 

aqs.sf <- aqs.wgs %>%
  dplyr::bind_rows(aqs.nad) 

# 1d. return to dataframe format
aqs <- aqs.sf %>%
  dplyr::mutate(lon = st_coordinates(aqs.sf)[,1], 
              lat = st_coordinates(aqs.sf)[,2]) %>% 
  as.data.frame() %>% 
  dplyr::select(-geometry)

#--------------------------#
#### 2. CURATE MONITORS ####
#--------------------------#

# 2a. keep only monitors within the bounding box 
aqs <- aqs %>% 
  dplyr::filter(lat > eastMA.bb$yMin & lat < eastMA.bb$yMax & 
                  lon > eastMA.bb$xMin  & lon < eastMA.bb$xMax)

# 2d. get annual averages
aqs.ann <- aqs %>% 
  dplyr::mutate(mon_id = paste0(state_name, '-', county_name, '-', 
                         site_num, '-', parameter_code, '-', 
                         poc)) %>% 
  dplyr::group_by(mon_id, lat, lon) %>% 
  dplyr::summarize(aqs = mean(aqs))

# 2e. keep only 1 monitor per location 
aqs.ann.unique <- aqs.ann %>% 
  dplyr::group_by(lat, lon) %>% 
  dplyr::slice(n=1) %>% 
  dplyr::ungroup()

# plot to check 
#aqs.ann.unique <- aqs %>% 
 # st_as_sf(., coords = c("lon", "lat"), crs=st_crs("epsg:4326"))  %>% 
  #dplyr::select(aqs)
#plot(aqs2.sf)

#---------------#
#### 3. SAVE ####
#---------------#

# 3A. Save as csv
aqs.ann.unique %>% 
  readr::write_csv(here::here('data', 'ground_truth', 'curated', 
                              'aqs_curated_2011.csv'))

