library(here)
library(sf)
library(magrittr)
library(dplyr)
library(raster)
library(tibble)
library(tidyr)
library(readr)
library(janitor)
library(nngeo) 
library(ggplot2)

# boundaries
eastMA.bb <- list(xMin = -73.5, xMax = -69.9, yMin = 40.49, yMax = 44.3)
current_path <- rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
base_models <- readr::read_csv('../data/prediction_dataset/base_model_predictions_eastMA.csv')

# 1b. make simple feature 
base_models <- base_models %>% 
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4326")) 

# 1c. read monitoring data 
aqs <- readr::read_csv('../data/ground_truth/curated/aqs_curated_2011.csv')

training <- readr::read_csv('../data/training_dataset/training_eastMA.csv')
training51 <- training[c(1:51),]


# 3a. set threshold of cluster in meters
clustThreshold <- 30 * 1000

# 3b. convert data to simple features
training.loc <- training51 %>% 
  st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4326")) %>% 
  st_transform(crs=st_crs("epsg:2163")) %>% 
  st_coordinates() %>% 
  as.data.frame() %>% 
  dplyr::select(X, Y) %>% 
  as.matrix()

# 3c. Compute k-means clustering
set.seed(15)
k <- 6
kmeans_res <- kmeans(training.loc, k)
kmeans_res$size


# 3d. Assign monitors to folds based on k-means clusters
training51$fold <- as.factor(kmeans_res$cluster)

# 4a. convert back to simple features 
training51.sf <- training51 %>% 
  st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4326"))

# Check folds
training51 %>%
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4269")) %>%
  plot()

