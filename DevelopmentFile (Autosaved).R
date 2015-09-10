
## Load libraries and set working directory
library(devtools)
library(roxygen2)
setwd("~/Desktop/CATsurv") #This will need to be changed to match your directory


## This can be run many times as the code is updates
current.code <- as.package("./catSurv")
Rcpp.package.skeleton(example_code=FALSE, attributes="TRUE")
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
sourceCpp("./catSurv/src/epv.cpp")

#see what things may be wrong...
#check(current.code)

## Build a version of the package to share manually
build(current.code, path=getwd())


##
x<-c(1,2,3,4)

vmean <- Vectorize(mean)
mean(x)
vmean(x)

x <- 1:100
filter(x, rep(1, 3))
