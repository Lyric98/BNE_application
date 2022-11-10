
# implement this GAM-based ensemble model: fit a GAM regression model
# y ~ f(longitude, latitude, model_1) + f(longitude, latitude, model_2) + f(longitude, latitude, model_3).
# where each f is a thin plate spline. 

library(tidyverse)
library(lubridate)
library(mgcv)

current_path <- rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))

# load the data
training <- read.csv("../data/training_dataset/training_eastMA.csv")
training <- training[0:51,]

X_valid = training[c("lon","lat")]
X_centr = colMeans(X_valid)
X_scale_lon = max(X_valid[1]) - min(X_valid[1])
X_scale_lat = max(X_valid[2]) - min(X_valid[2])

X_valid["lon"] = (X_valid["lon"] - X_centr["lon"]) / X_scale_lon
X_valid["lat"] = (X_valid["lat"] - X_centr["lat"]) / X_scale_lat

training[c("lon","lat")] <- X_valid

# gam_model <- gam(aqs ~ s(pred_av) +
#                    s(pred_gs) +
#                    s(pred_caces)+ te(lon, lat),
#            data = training)

gam_model <- gam(aqs ~ s(lon, lat, by=pred_av, k=4) +
                   s(lon, lat, by=pred_gs, k=4) +
                   s(lon, lat, by=pred_caces, k=4),
                 data = training[0:30,])

# gam_model <- gam(aqs ~ s(lon, lat, by=pred_av, k=4),
#                  data = training[0:30,])

gam_model <- bam(aqs ~ s(lon, lat, by=pred_av) +
                   s(lon, lat, by=pred_gs) +
                   s(lon, lat, by=pred_caces),
                 data = training[0:30,])

preds_train <- predict(gam_model, newdata = training[31:51,]) %>% as.numeric()
#preds_test  <- predict(gam_model, newdata = testing(splits)) %>% as.numeric()
rmse_gam = sqrt(mean((training[31:51,]$aqs - preds_train)^2))
rmse_gam


lr_model <- lm(aqs~lon+lat, data=training)

preds_lr <- predict(lr_model, newdata = training) %>% as.numeric()
#preds_test  <- predict(gam_model, newdata = testing(splits)) %>% as.numeric()
rmse_lr = sqrt(mean((training$aqs - preds_lr)^2))
rmse_lr

library(caret)
set.seed(3)
#Randomly shuffle the data
training<-training[sample(nrow(training)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(training)),breaks=10,labels=FALSE)
rmse_gam = c()
#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- training[testIndexes, ]
  trainData <- training[-testIndexes, ]
  #Use the test and train data partitions however you desire...
  gam_model <- gam(aqs ~ s(lon, lat, by=pred_av, k=4) +
                     s(lon, lat, by=pred_gs, k=4) +
                     s(lon, lat, by=pred_caces, k=4),
                   data = trainData)
  preds_test <- stats::predict(gam_model, newdata =testData, interval="prediction") %>% as.numeric()
  #preds_test  <- predict(gam_model, newdata = testing(splits)) %>% as.numeric()
  rmse = sqrt(mean((testData$aqs - preds_test)^2))
  rmse_gam[i] = rmse
}



