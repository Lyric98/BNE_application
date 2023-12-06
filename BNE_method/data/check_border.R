current_path <- rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))

training_eastMA <- read.csv("../data/training_dataset/training_eastMA.csv")
