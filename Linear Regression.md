## Loading in my libraries
```{r}
library(dplyr)
library(tidyverse)
library(randomForest)
```

## Reading in the Data
```{r}
test <- read.csv("test.csv")
train <- read.csv("train.csv")

# Removing the id column from both datasets

train <- dplyr::select(train, -id)

dim(test)
dim(train)
```

## Plotting a correlation matrix
```{r}
# Here are two different correlation plots that may seem more clear to you

ggcorrplot::ggcorrplot(cor(train))

corrplot::corrplot(cor(train))
```
The target variable is not correlated well with most variables, and the covariates are not well correlated with few exceptions.

## Splitting the train data into a test and a training set
```{r}
library(caret)

intrain <-  createDataPartition(y = train$target, p = .70, list = FALSE)
training <- train[intrain,]
testing <- train[-intrain,]
```
## Building the Linear Regression Model
```{r}
mod.1 <- lm(target ~., data = training)
summary(mod.1)
```
|Multiple R Squared = 0.0185| Adjusted R Squared = 0.01844|
|---|---|

This is a poor model. Let's check our model assumptions
```{r}
plot(mod.1$residuals, mod.1$fitted.values)
```
This seems random with an outlier so let's remove outliers/influential points.

```{r}
# Standardize the residuals
out <- rstandard(mod.1)
length(which(abs(out)>3))

#Compute Cook's Distances
cd <- cooks.distance(mod.1)

#Compare Cook's Distances to 50th percentile of F distribution with 2 and 28 DF
F_thresh <- qf(.50,ncol(training)-2,nrow(training)-((ncol(training) - 1)+1))  

#Find which cooks distances are greater than the F threshold. 
#The which() function returns the element, or position, of the observation(s) satisifying the logical condition

which(cd > F_thresh)
```
While there are ostensibly no influential points, let's remove the outliers and see how well the model performs
```{r}
train_no_out <- training[-which(abs(out)>3),]

mod.2 <- lm(target ~., data = train_no_out)
summary(mod.2)
```
|Multiple R Squared = 0.0189| Adjusted R Squared = 0.0188|
|---|---|

## Let's now check our model assumptions
```{r}
plot(mod.2$residuals, mod.2$fitted.values)

qqnorm(mod.2$residuals)
hist(mod.2$residuals)
```
The tests seem to hold up well.

## Let's compute a stepAIC to determine the best model
```{r}
library(MASS)
stepAIC(mod.2)
```

Step AIC suggests we could use a smaller model.

## Let's fit this model but keep our original in mind.
```{r}
mod.3 <- lm(formula = target ~ cont1 + cont2 + cont3 + cont4 + cont5 + 
    cont6 + cont7 + cont8 + cont9 + cont10 + cont11 + cont12 + 
    cont13, data = train_no_out)
summary(mod.3)
```
|Multiple R Squared = 0.0189| Adjusted R Squared = 0.0188|
|---|---|

Not much has changed in this respect. 

## Let's now test the Root Mean Squared Error (RMSE) 
The RMSE is the metric the competition is evaluating the model.
```{r}
pred <- predict(mod.2, newdata = testing)
RMSE = sqrt(sum((testing$target - pred)^2)/nrow(testing))
RMSE

pred.2 <- predict(mod.3, newdata = testing)
RMSE = sqrt(sum((testing$target - pred)^2)/nrow(testing))
RMSE
```


