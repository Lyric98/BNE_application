# File: STR_b_01c_create_refGrid_eastMA.R
# Author: Sebastian Rowland <sr3463@cumc.columbia.edu>
# Date: 10/27/2021
#
# Contents:
# N. Notes
# 0. Import Packages and Set Global Variables
# 1. Combine Base Models and Monitoring Data
# 2. Save

#----------------#
#### N. NOTES ####
#----------------#

# This script takes about 1 minute to run. 


#---------------------------------------------------#
#### 0: IMPORT PACKAGES AND SET GLOBAL VARIABLES ####
#---------------------------------------------------#

# 0a Load packages and functions required for this script
if(!exists("Ran_a_00")){
  here::i_am("bne_eastMA_app.Rproj")
  source(here::here('code',  "0_00_set_up_env.R"))
}

#-----------------------------------------------#
#### 1. COMBINE BASE MODELS AND MONITOR DATA ####
#-----------------------------------------------#

# 1a. read harmonized base models 
base_models <- readr::read_csv(here::here('data', 'prediction_dataset',  
                                           'base_model_predictions_eastMA.csv'))

# 1b. make simple feature 
base_models <- base_models %>% 
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4326")) 

# 1c. read monitoring data 
aqs <- readr::read_csv(here::here('data', 'ground_truth', 'curated', 
                                     'aqs_curated_2011.csv'))

# 1d. make simple features 
aqs <- aqs %>% 
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4326")) 

# 1e. spatial join by nearest neighbor 

aqs$index_pred <- nngeo::st_nn(aqs, base_models, k = 1) %>% 
  unlist()

base_models <- base_models %>% 
  dplyr::mutate(index_pred = row_number()) %>% 
  as.data.frame() %>% 
  dplyr::select(-geometry)

training <- aqs %>% 
  dplyr::inner_join(base_models, by = 'index_pred')

#---------------#
#### 2. SAVE ####
#---------------#

# 2a. convert to dataframe
training <- training %>% 
  dplyr::mutate(lon = st_coordinates(training)[,1], 
                lat = st_coordinates(training)[,2]) %>% 
  as.data.frame() %>% 
  dplyr::select(-geometry, -contains('index'))

# 2b. plot to confirm success
training %>% 
 sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4269")) %>% 
  plot()

# 2b. save the points
training %>% 
  #dplyr::select(lat, lon) %>%
  readr::write_csv(here::here('data', 'training_dataset',  
                            'training_eastMA.csv'))

#---------------------------------#
#### 3. IDENTIFY SPATIAL FOLDS ####
#---------------------------------#

# 3a. set threshold of cluster in meters
clustThreshold <- 30 * 1000

# 3b. convert data to simple features
training.loc <- training %>% 
  st_as_sf(coords = c('lon', 'lat'), crs=sf::st_crs("epsg:4326")) %>% 
  st_transform(crs=sf::st_crs("epsg:2163")) %>% 
  mutate(lat = st_coordinates(.)[,1], 
         lon = st_coordinates(.)[,2]) %>% 
  dplyr::select(lat, lon) %>% 
  as.data.frame() %>% 
  dplyr::select(-geometry)

# 3c. compute distances among monitors
df.dist <- stats::dist(training.loc, method = "euclidean")
rm(training.loc)

# 3d. Create hierarchial cluster solution
hc.complete <- stats::hclust(df.dist, method = "single")

# 3e. Cut tree
hc.cluster <- stats::cutree(hc.complete, h = clustThreshold)

# 3f. Assign cluster to each training point
training$hc <- hc.cluster

# 3g. Assign Monitors to Folds
# we use the greedy algorithm described by Just et. al. 
# 3g.i. Determine the number of monitors in each spatial cluster
hclust <- training%>% 
  dplyr::group_by(hc) %>% 
  dplyr::summarize(Mon = n()) %>% 
  dplyr::arrange(desc(Mon))
# 3g.ii. Assign each of the largest clusters to a fold
fold_df <- hclust %>% 
  dplyr::slice(1:10) %>% 
  dplyr::mutate(fold = 1:10)
# 3g.iii. Remove those clusters from the dataframe of available clusters 
# which we call hclust.sm
hclust.sm <- hclust %>%
  dplyr::slice(11:nrow(hclust)) %>% 
  dplyr::mutate(fold = 0)

# 3h. Assign fold membership to each cluster over a loop 
for (i in 1:nrow(hclust.sm)){
  #i <- 1
  # 3h.i. identify the current smallest fold
  smallestFold <- fold_df %>% 
    dplyr::group_by(fold) %>% 
    dplyr::summarize(Count = sum(Mon)) %>%
    dplyr::arrange(Count) %>% 
    dplyr::slice(1:1)
  # 3h.ii. add assign the next largest cluster to the smallest fold
  hclust.sm$fold[1] <- smallestFold$fold[1]
  # 3h.iii add that assigned cluster to assigned cluster pile 
  fold_df <- fold_df %>% 
    dplyr::bind_rows(hclust.sm[1,])
  # 3h.iv remove that assigned cluster from unassigned cluster pile
  hclust.sm <- hclust.sm[2:nrow(hclust.sm),]
}

# 3i. assign monitors to folds according to their cluster 
training <- training %>% 
  dplyr::inner_join(fold_df, by = 'hc') 
  
# 3j. create variable to track folds
training <- training %>% 
  dplyr::mutate(fold = paste0('fold', str_pad(fold, 2, 'left', '0')))

# 3k. save 
training %>% 
  readr::write_csv(here::here('data', 'training_dataset',  
                              'training_eastMA_folds.csv'))

#--------------------------------------#
#### 4. PLOT TO CHECK SPATIAL FOLDS ####
#--------------------------------------#

# 4.a. convert back to simple features 
training.sf <- training %>% 
  st_as_sf(coords = c('lon', 'lat'), crs=sf::st_crs("epsg:4326"))

# 4.b. bring in outline of the area
conus <- sf::read_sf(here::here('data', 'conus_outline', 'conus.shp')) %>% 
  sf:: t_transform(crs=sf::st_crs("epsg:4326"))  
conus.ne <- conus %>% 
  st_crop(c(st_bbox(training.sf)[1]-0.5, st_bbox(training.sf)[2]-1, 
            st_bbox(training.sf)[3]+0.5, st_bbox(training.sf)[4]+0.5))

# 4.c. plot folds
png(here::here('outputs', 'training_folds.png'))
ggplot(conus.ne) + 
  geom_sf(fill = 'grey95') + 
  geom_sf(data = training.sf,  aes(color = fold, fill = fold)) + 
  ggsci::scale_color_d3()  + 
  theme_classic()
dev.off()