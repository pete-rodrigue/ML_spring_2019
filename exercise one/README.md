# README

## About

This folder contains all the files needed to make the figures present in the pdf "exercise one.pdf"

## Files

The files in this folder:

* exercise one.pdf
  * This is the pdf with answers to the homework.
* Homework 1 - Diagnostic.pdf
  * This is the original homework assignment
* hw1.py
  * This is the python file that creates figures and statistics for exercise one.pdf
* alleged_crimes_2017.csv
  * This is the Chicago crime data for 2017 downloaded from the city. It gets downloaded in hw1.py (that code is commented out now).
* alleged_crimes_2018.csv
  * This is the Chicago crime data for 2017 downloaded from the city. It gets downloaded in hw1.py (that code is commented out now).
* CommAreas.csv
  * This file contains information about Chicago's community areas. It's used in hw1.py.
* Boundaries - Community Areas (current) (1).geojson
  * This file contains boundaries for Chicago's community areas. It's used in hw1.py
* R12107097_SL140.csv
  * This file contains Census data about tract family income, average household size, and poverty. It's downloaded from Social Explorer. The accompanying textfile in this folder, R12107097.txt, contains the data dictionary explaining what the variable names in this file mean.
* geo_export_131406b6-afd8-46ca-98c9-f3564a20a214.shp (and similarly named files)
  * These files are the shapefile boundaries of all the Census tracts in Cook County.
* spatialJoinCrimesTracts.R
  * This R file takes the files alleged_crimes_2018.csv, alleged_crimes_2017.csv, and geo_export_131406b6-afd8-46ca-98c9-f3564a20a214.shp and does a spatial join of crimes to census tracts. It exports the resulting tables as alleged_crimes_2018_with_tracts.csv and alleged_crimes_2017_with_tracts.csv.
* alleged_crimes_2017_with_tracts
  * This is the crime data with the tract-level Census data about median family income, average household size, and poverty rates merged on.
* alleged_crimes_2017_with_tracts
  * This is the crime data with the tract-level Census data about median family income, average household size, and poverty rates merged on.


## To run the code in hw1.py:

  1. You will want to open the folder “exercise one”
  2. You will want to clone or download the folder “exercise one”
  3. In the folder “exercise one”, run the file “hw1.py”. Be sure that your working directory is the folder “exercise one”.
  4. Things to note: There is an R script in the folder “exercise one” called “spatialJoinCrimesTracts.R”. I did the spatial join of crimes to census tracts in R, using this script. All of the figures are exported to the folder “figures”.
