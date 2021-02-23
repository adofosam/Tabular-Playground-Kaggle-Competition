## Loading in my libraries
```{r}
library(dplyr)
library(tidyverse)
library(randomForest)
```

## Reading in the Data
```
test <- read.csv("test.csv")
train <- read.csv("train.csv")

# Removing the id column from both datasets

train <- dplyr::select(train, -id)

dim(test)
dim(train)

```