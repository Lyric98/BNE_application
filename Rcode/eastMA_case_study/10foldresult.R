
rmse_mean_std <- read.table("test.txt",              # TXT data file indicated as string or full path to the file
           header = TRUE,    # Whether to display the header (TRUE) or not (FALSE)
           sep = "",dec = " ")  
View(rmse_mean_std)

rmse_mean_cos <- read.table("cosine_rmse.txt",              # TXT data file indicated as string or full path to the file
                            header = TRUE,    # Whether to display the header (TRUE) or not (FALSE)
                            sep = "",dec = " ")  
View(rmse_mean_cos)
