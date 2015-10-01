## Load libraries and set working directory
library(devtools)
library(roxygen2)
setwd("~/Desktop/CAT-Survey") #This will need to be changed to match your git directory
current.code <- as.package("./catSurv")
load_all(current.code)
document(current.code)

## Install the package
install(pkg=current.code, local=TRUE)

## For Testing
setwd("~/Desktop/CATsurv")
library(rjson)
library(CATPack)
cat <- new("CATsurv")
json_cat <- fromJSON(file="eqModel.txt")
cat@guessing <- json_cat$guessing
cat@discrimination <- unlist(json_cat$discrimination)
cat@answers <- as.numeric(json_cat$answers)
cat@priorName <- json_cat$priorName
cat@priorParams <- json_cat$priorParams
cat@lowerBound <- json_cat$lowerBound
cat@upperBound <- json_cat$upperBound
cat@quadPoints <- json_cat$quadPoints
cat@D <- json_cat$D
cat@difficulty <- lapply(json_cat$difficulty, unlist)
cat@X <- json_cat$X
cat@poly <- TRUE
nextItemEPVcpp(cat)

## Run this if you change code and need to resource the cpp file
sourceCpp("./catSurv/src/epv.cpp")

## Build a version of the package to share manually
build(current.code, path=getwd())
