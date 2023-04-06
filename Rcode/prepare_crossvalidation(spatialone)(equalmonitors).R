#library(pacman)
library(here)
library(sf)
library(magrittr)
library(dplyr)
library(fst)
library(raster)
library(tibble)
library(tidyr)
library(readr)
library(janitor)
library(nngeo) 

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

# 2b. plot to confirm success
training51 %>% 
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4269")) %>% 
  plot()

base_models %>% 
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4269")) %>% 
  plot()

# 3a. set threshold of cluster in meters
clustThreshold <- 3 * 1000

# 3b. convert data to simple features
training.loc <- training51 %>% 
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
training51$hc <- hc.cluster

# 3g. Assign Monitors to Folds
# we use the modified greedy algorithm described above
# 3g.i. Determine the number of monitors in each spatial cluster
hclust <- training51 %>% 
  group_by(hc) %>% 
  summarize(Mon = n()) %>% 
  arrange(desc(Mon))
# 3g.ii. Assign each of the largest clusters to a fold
fold_df <- hclust %>% 
  dplyr::slice(1:10) %>% 
  dplyr::mutate(fold = 1:10)


num_folds <- 10
fold_size <- ceiling(sum(hclust$Mon) / num_folds)
training51$fold <- 0
assigned_clusters <- 0
fold_id <- 1
for (i in 1:nrow(hclust)) {
  # 3g.iii. Add the current cluster to the current fold
  training51$fold[which(training51$hc == hclust$hc[i])] <- fold_id
  assigned_clusters <- assigned_clusters + hclust$Mon[i]
  # 3g.iv. If the current fold is full, move to the next fold
  if (assigned_clusters >= fold_size) {
    fold_id <- fold_id + 1
    assigned_clusters <- 0
  }
}

# 3g.iii. Remove those clusters from the dataframe of available clusters
# which we call hclust.sm
hclust.sm <- hclust %>%
  dplyr::slice(11:nrow(hclust)) %>%
  dplyr::mutate(fold = 0)

# # 3h. Assign fold membership to each cluster over a loop 
# unassigned_clusters <- hclust %>% 
#   dplyr::select(hc, Mon) %>% 
#   rename(n = Mon)
# 
# while (nrow(unassigned_clusters) > 0) {
#   # 3h.i. Identify the current smallest fold
#   smallestFold <- training51 %>%
#     filter(fold > 0) %>% 
#     group_by(fold) %>% 
#     summarize(Count = n()) %>%
#     arrange(Count) %>% 
#     slice(1:1)
#   # 3h.ii. Assign the next largest cluster to the smallest fold
#   next_cluster <- unassigned_clusters %>% 
#     arrange(desc(n)) %>% 
#     slice(1:1)
#   training51$fold[which(training51$hc == next_cluster$hc)] <- smallestFold$fold[1]
#   # 3h.iii. Remove the assigned cluster from the unassigned cluster pile
#   unassigned_clusters <- unassigned_clusters[-1,]
# }



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
training51 <- training51 %>% 
  dplyr::inner_join(fold_df, by = 'hc') 

# 4.a. convert back to simple features 
training51.sf <- training51 %>% 
  st_as_sf(coords = c('lon', 'lat'), crs=sf::st_crs("epsg:4326"))






training51 %>%
  sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4269")) %>%
  plot()












# # 3g. Assign Monitors to Folds
# # we use the greedy algorithm described by Just et. al. 
# # 3g.i. Determine the number of monitors in each spatial cluster
# hclust <- training51%>% 
#   dplyr::group_by(hc) %>% 
#   dplyr::summarize(Mon = n()) %>% 
#   dplyr::arrange(desc(Mon))
# # 3g.ii. Assign each of the largest clusters to a fold
# fold_df <- hclust %>% 
#   dplyr::slice(1:10) %>% 
#   dplyr::mutate(fold = 1:10)
# # 3g.iii. Remove those clusters from the dataframe of available clusters 
# # which we call hclust.sm
# hclust.sm <- hclust %>%
#   dplyr::slice(11:nrow(hclust)) %>% 
#   dplyr::mutate(fold = 0)
# 
# # 3h. Assign fold membership to each cluster over a loop 
# for (i in 1:nrow(hclust.sm)){
#   #i <- 1
#   # 3h.i. identify the current smallest fold
#   smallestFold <- fold_df %>% 
#     dplyr::group_by(fold) %>% 
#     dplyr::summarize(Count = sum(Mon)) %>%
#     dplyr::arrange(Count) %>% 
#     dplyr::slice(1:1)
#   # 3h.ii. add assign the next largest cluster to the smallest fold
#   hclust.sm$fold[1] <- smallestFold$fold[1]
#   # 3h.iii add that assigned cluster to assigned cluster pile 
#   fold_df <- fold_df %>% 
#     dplyr::bind_rows(hclust.sm[1,])
#   # 3h.iv remove that assigned cluster from unassigned cluster pile
#   hclust.sm <- hclust.sm[2:nrow(hclust.sm),]
# }
# 
# # 3i. assign monitors to folds according to their cluster 
# training51 <- training51 %>% 
#   dplyr::inner_join(fold_df, by = 'hc') 
# 
# # 4.a. convert back to simple features 
# training51.sf <- training51 %>% 
#   st_as_sf(coords = c('lon', 'lat'), crs=sf::st_crs("epsg:4326"))
# 
# # check folds
# training51 %>% 
#   sf::st_as_sf(coords = c("lon", "lat"), crs=st_crs("epsg:4269")) %>% 
#   plot()
# 
# write.csv(training51, "./data/training_dataset/training51.csv", row.names = FALSE)