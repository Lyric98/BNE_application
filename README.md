# BNE_application

## Installation

To install requirements:
```
$ conda env create -f environment.yml
or 
$ conda env update -f environment.yml

$ conda activate BNE
```
This environment includes all necessary dependencies.

To create an `BNE` executable to run experiments, run `pip install -e .`.

## Part 1: Data

### Availability

For prediction files of air pollution models (MH2021, GS2018, and SK2020 to be used as base models for BNE ensemble), the dataset can be requested by following instructions of the respective websites:


* MH2021 (Aaron von Donkelaar's group's model)
Hammer, M. S.; van Donkelaar, A.; Li, C.; Lyapustin, A.; Sayer, A. M.; Hsu, N. C.; Levy, R. C.; Garay, M. J.; Kalashnikova, O. V.; Kahn, R. A.; Brauer, M.; Apte, J. S.; Henze, D. K.; Zhang, L.; Zhang, Q.; Ford, B.; Pierce, J. R.; and Martin, R. V., Global Estimates and Long-Term Trends of Fine Particulate Matter Concentrations (1998-2018)., Environ. Sci. Technol, doi: 10.1021/acs.est.0c01764, 2020.
https://sites.wustl.edu/acag/datasets/surface-pm2-5/ 


* GS2018 (Global Burden of Disease)
Shaddick, Gavin, et al. "Data integration for the assessment of population exposure to ambient air pollution for global burden of disease assessment." Environmental science & technology 52.16 (2018): 9069-9078.
https://pubs.acs.org/doi/10.1021/acs.est.8b02864 


* SK2020 (Center for Air, Climate, and Energy Solutions)
Kim S.-Y.; Bechle, M.; Hankey, S.; Sheppard, L.; Szpiro, A. A.; Marshall, J. D. 2020. “Concentrations of criteria pollutants in the contiguous U.S., 1979 – 2015: Role of prediction model parsimony in integrated empirical geographic regression.” PLoS ONE 15(2), e0228535. DOI: 10.1371/journal.pone.0228535
https://www.caces.us/data 

For geographic data mapping, we utilized shapefiles provided by the United States Census Bureau, sourced from their Topologically Integrated Geographic Encoding and Referencing (TIGER) Geographic Information System (GIS) clearinghouse. These files can be accessed at the following URL: https://www2.census.gov/geo/tiger/GENZ2015/shp/. The TIGER/Line shapefiles offer detailed and comprehensive geographic data that are essential for accurate mapping and spatial analysis.


### Description

#### File format(s)

The processed version of the above three base models’ prediction dataset are included in 
https://github.com/Lyric98/BNE_application/blob/master/data/training_dataset/training51.csv . We used this CSV file to processed our case study.

#### Data dictionary

In our case study, we processed the first 7 columns of the above CSV file and the Column Descriptions are as following:

* **mon_id**: Unique identifier for monitors' information (State-County-dentifier);
* **aqs**: This column refers to the United States Environmental Protection Agency (USEPA) AQS data, which was downloaded from the USEPA's Air Data clearinghouse https://aqs.epa.gov/aqsweb/airdata/download_files.html;
* **pred_av**: Predicted values from the base model "MH2021";
* **pred_gs**: Predicted values from the base model "GS2018";
* **pred_caces**: Predicted values from the base model "SK2020";
* **lon**: Longitude coordinates of the monitor stations;
* **lat**: Latitude coordinates of the monitor stations.

## Part 2: Code

The codebase includes modules to train BNE models using base model predictions and air pollution outcome (under [BNE_method](https://github.com/Lyric98/BNE_application/tree/master/BNE_method)), and also jupyer notebooks for generating prediction and visualize results ([BNE_method/case_study_results](https://github.com/Lyric98/BNE_application/tree/master/BNE_method/case_study_results)). The detailed instructions for training code for parameter tuning is in [BNE_method/Tuning/README.me](https://github.com/Lyric98/BNE_application/tree/master/BNE_method/Tuning#readme), and the instructions for executing the jupyter notebook is contained in the notebook itself.

#### Supporting software requirements 
- Python = 3.8
- jupyter notebook
- R = 4.1.2

Libraries and dependencies used by the code are detailed in `environment.yml`.


## Part 3: Reproducibility Workflow

### Reproducibility Instructions
Each Jupyter-notebook file (.ipynb) or Python file (.py) can be run independently to generate the examples, simulation experiment, or the case study. The R code can be run to prepare the data for the case study. Note that base data refers to data that the base models are trained on; train data refers to data that the ensemble is trained on, and test data refers to data that the ensemble is evaluated on. 

`BNE_examine.ipynb` includes all simulation codes (1D/2D toy models).

For the case study experiments, Table 4 and Table S3 are generated diretly from `DirectlyEnsembles.ipynb`; Figure 4, Figure 5 and Figure S1 are generated by `BNE_1213.ipynb` and `BNE_spatialCV.ipynb`.

### Expected run-time

For the Jupyter notebook files we are able to reproduce on a standard desktop machine (Processor: 2.3 GHz Dual-Core Intel Core i5; Memory: 8 GB 2133 MHz LPDDR3) during 1-2 hours. But for tuning parameters, we need to run it on clusters.

### Additional documentation (Tuning Paramater)

For the case studies, the computations in tuning the hyperparameters of the BNE models were run on the FASRC Cannon cluster supported by the FAS Division of Science Research Computing Group at Harvard University.