## Spatial merge crimes and census tracts

library(sp)
library(rgdal)
library(spatialEco)


setwd("C:/Users/edwar.WJM-SONYLAPTOP/Documents/GitHub/ML_spring_2019/exercise one")

df <- read.csv('alleged_crimes_2018.csv')
df <- df[is.na(df$latitude) == F, ]
spdf <- SpatialPointsDataFrame(coords = df[, c('longitude', 'latitude')], data = df)


shape <- readOGR(".","geo_export_131406b6-afd8-46ca-98c9-f3564a20a214")


proj4string(spdf)
proj4string(shape)

proj4string(spdf) <- CRS("+init=epsg:4326")
shape <- spTransform(shape, CRS("+init=epsg:4326"))

spatial_joined <- point.in.poly(spdf, shape)
plot(spatial_joined[1:200,])

new.df <- spatial_joined@data

write.csv(new.df, "alleged_crimes_2018_with_tracts.csv")






df <- read.csv('alleged_crimes_2017.csv')
df <- df[is.na(df$latitude) == F, ]
spdf <- SpatialPointsDataFrame(coords = df[, c('longitude', 'latitude')], data = df)


shape <- readOGR(".","geo_export_131406b6-afd8-46ca-98c9-f3564a20a214")


proj4string(spdf)
proj4string(shape)

proj4string(spdf) <- CRS("+init=epsg:4326")
shape <- spTransform(shape, CRS("+init=epsg:4326"))

spatial_joined <- point.in.poly(spdf, shape)
plot(spatial_joined[1:200,])

new.df <- spatial_joined@data

write.csv(new.df, "alleged_crimes_2017_with_tracts.csv")
