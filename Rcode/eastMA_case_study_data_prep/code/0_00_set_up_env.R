# File: 0_00_set_up_env.R
# Author: Sebastian Rowland <sr3463@cumc.columbia.edu>
# Date: 10/28/2021

##-----------------------##
#### Table of Contents ####
##-----------------------##

# 0: Preparation 
# 1: Load Packages for Analysis
# 2: Set Features
# 3: Load Functions 
# 4: Make CONUS Outline

#----------------------#
#### 0: PREPARATION ####
#----------------------#

# 0a. Indicate that the script was run
# although it is intuitive to run this at the end of the script, 
# that creates a recursion with the functions
Ran_a_00 <- "Ran_a_00"

# 0b. Load pacman package
library(pacman)

#-------------------------------------#
#### 1: LOAD PACKAGES FOR ANALYSIS ####
#-------------------------------------#

# 1a. Load Packages
p_load(here, sf, magrittr, dplyr, fst, raster, tibble, tidyr, readr, janitor,
       nngeo) 

#-----------------------#
#### 2: SET FEATURES ####
#-----------------------#

# 2a set the projection string 
projString <- "epsg:2163"
#projStringRas <- "+init=epsg:2163"

# boundaries

eastMA.bb <- list(xMin = -73.5, xMax = -69.9, yMin = 40.49, yMax = 44.3)

