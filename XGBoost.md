The data transformation was done in the Linear Regression folder so please refer to that when finding training and testing.

## Trying an XGBoost model
```{r}
library(xgboost)
training_y <- training[,"target"]
training_no_y <- dplyr::select(training, -target)

testing_y <- testing[,"target"]
testing_no_y <- dplyr::select(testing, -target)

mode(training_no_y)
class(training_no_y)

dtrain <- xgb.DMatrix(data = as.matrix(training_no_y), label= training_y)
dtest <- xgb.DMatrix(data = as.matrix(testing_no_y), label= testing_y)
```
## Training the model Using a tree booster
```{r}
mod <- xgboost(data = dtrain, # the data
               nround = 20, # max number of boosting iterations
               objective = "reg:linear")  # the objective function
```
|RMSE = 0.698|
|---|
This is a better RMSE than the linear regression model. 
I also noticed when I increase the number of rounds to 200, the RMSE still continues to decrease. I decided to cut it off after 200 because the change in RMSE is very minimal (It is perfectly plausible to cut it off well before that). 

Now, to test the model on the testing set
```{r}
pred <- predict(mod, newdata = dtest)
RMSE = sqrt(sum((testing$target - pred)^2)/nrow(testing))
RMSE
```
|RMSE = 0.706|
|---|

## Training the model using a linear booster
```{r}

params = list(booster = "gbtree",
              objective = "reg:linear",
              eta = 0.1,
              gamma = 0,
              max_depth = 6,
              min_child_weight = 6,
              subsample = 1,
              colsample_bytree = 1,
              lambda = 0,
              alpha = 1)


mod.lin <- xgboost(params = params,
                   data = dtrain, # the data
                   nround = 200 # max number of boosting iterations
                   )
```
|RMSE = 0.677|
|---|

## Starting to optimize some of the parameters.

Let's Start with eta
```{r}

eta_opt <- seq(0.1, 0.3, by = 0.01)
min_eta = 10000


for(i in eta_opt){
  
  params = list(booster = "gbtree",
              objective = "reg:linear",
              eta = i,
              gamma = 0,
              max_depth = 6,
              min_child_weight = 6,
              subsample = 1,
              colsample_bytree = 1,
              lambda = 0,
              alpha = 1)


mod.lin <- xgboost(params = params,
                   data = dtrain, # the data
                   nround = 100, # max number of boosting iterations
                   verbose = 0)

  min_eta.1 <- min(mod.lin$evaluation_log$train_rmse)
  iter <- which.min(mod.lin$evaluation_log$train_rmse)
  
  if(min_eta.1 < min_eta){
    
    min_eta <- min_eta.1
    opt_eta <- cbind(iter, min_eta.1, i)
  }


}
print(opt_eta)
```
Seems as though 100 rounds of eta 0.3 works the best.

Now, optimizing max depth
```{r}

depth_opt <- seq(0, 100, by = 5)
min_rmse = 10000


for(i in depth_opt){
  
  params = list(booster = "gbtree",
              objective = "reg:linear",
              eta = 0.3,
              gamma = 0,
              max_depth = i,
              min_child_weight = 6,
              subsample = 1,
              colsample_bytree = 1,
              lambda = 0,
              alpha = 1)


mod.lin <- xgboost(params = params,
                   data = dtrain, # the data
                   nround = 100, # max number of boosting iterations
                   verbose = 0)

  min_rmse.1 <- min(mod.lin$evaluation_log$train_rmse)
  iter <- which.min(mod.lin$evaluation_log$train_rmse)
  
  if(min_rmse.1 < min_rmse){
    
    min_rmse <- min_rmse.1
    opt_rmse <- cbind(iter, min_rmse, i)
  }


}
print(opt_rmse)
```

## Let's test on the testing set
```{r}

pred <- predict(mod.lin, newdata = dtest)
RMSE = sqrt(sum((testing$target - pred)^2)/nrow(testing))
RMSE
```

# Now, let's go ahead and test it with the Kaggle test set.
```{r}

params = list(booster = "gbtree",
              objective = "reg:linear",
              eta = 0.3,
              gamma = 0,
              max_depth = 15,
              min_child_weight = 257,
              subsample = 0.7,
              colsample_bytree = 0.5,
              lambda = 0.03,
              alpha = 0.016)

mod_train <- xgb.train(params = params, 
                       data = dtrain, 
                       nrounds = 100)

test_data <- xgb.DMatrix(data = as.matrix(test))

pred.1 <- predict(mod_train, newdata = test_data)
```

## Printing to a dataframe
```{r}

id <- test$id
target <- pred.1

submission <- cbind(id,target)
submission <- as.data.frame(submission)

write_csv(submission, "submission_5.csv")
  
```




