
# implement this GAM-based ensemble model: fit a GAM regression model
# y ~ f(longitude, latitude, model_1) + f(longitude, latitude, model_2) + f(longitude, latitude, model_3).
# where each f is a thin plate spline. 


library(tidymodels)
library(tidyverse)
library(timetk)
library(lubridate)
library(mgcv)

current_path <- rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))

# load the data
training <- read.csv("../data/training_dataset/training_eastMA.csv")

gam_model <- gam(y ~ s(lon, lat, pred_av) + s(lon, lat, pred_gs) + s(lon, lat, pred_caces), 
           data = training, 
           method = "REML")


preds_train <- predict(gam_model, newdata = training) %>% as.numeric()
#preds_test  <- predict(gam_model, newdata = testing(splits)) %>% as.numeric()
