### --- Model Fitting with Log(SALE.PRICE) --- ###
######################################

library(recipes)  # for feature engineering
library(glmnet)   # for implementing regularized regression
library(caret)    # for automating the tuning process
library(vip)     # for variable importance
library(rsample)
library(ranger)   # a c++ implementation of random forest 
library(h2o)      # a java-based implementation of random forest
library(car)







#install.packages('h2o')

nycprop2=read.csv('nycprop_processed.csv')
dim(nycprop2)
names(nycprop2)
#nycprop2=subset(nycprop2,select=-c(missing_value))
View(nycprop2)

boxplot(nycprop2$SALE.PRICE)
outliers = boxplot(nycprop2$SALE.PRICE)$out
idx_out = which(nycprop2$SALE.PRICE%in% outliers)
length(outliers)
#nycprop2 <- nycprop2[-idx_out, ]
dim(nycprop2)

nycprop2=subset(nycprop2,SALE.PRICE>150000)
nycprop2=subset(nycprop2,SALE.PRICE<5000000)
nycprop2=subset(nycprop2,GROSS.SQUARE.FEET<150000)
nycprop2=subset(nycprop2,LAND.SQUARE.FEET<30000)
nycprop2=subset(nycprop2,TOTAL.UNITS<10)
nycprop2=subset(nycprop2,COMMERCIAL.UNITS<6)
nycprop2=subset(nycprop2,Property_Age<200)



hist(nycprop2$Property_Age)
boxplot(nycprop2$GROSS.SQUARE.FEET)
outliers = boxplot(nycprop2$GROSS.SQUARE.FEET)$out
length(outliers)
max(nycprop2$GROSS.SQUARE.FEET)
table(nycprop2$COMMERCIAL.UNITS)



### Log transform SALE.PRICE ###############
nycprop2$SALE.PRICE=log(nycprop2$SALE.PRICE)
############################################





###Stratified sampling to split the data
#With a continuous response variable, stratified sampling will segment Y into quantiles and randomly sample 
#from each. Consequently, this will help ensure a balanced representation of the response distribution in both
#the training and test sets.

library(rsample)
library(caret)

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
split_strat <- initial_split(nycprop2, prop = 0.8, strata = "SALE.PRICE")
train_nyc <- training(split_strat)
test_nyc <- testing(split_strat)

### Log transform SALE.PRICE ###############
#train_nyc$SALE.PRICE=log(train_nyc$SALE.PRICE)
#test_nyc$SALE.PRICE=log(test_nyc$SALE.PRICE)
############################################

# Create the model matrix and response vectors for the training set
train_x <- model.matrix(SALE.PRICE ~ ., data = train_nyc)[, -1] #[, -1] to discard the intercept
train_y <- train_nyc$SALE.PRICE

# Create the model matrix and response vectors for the testing set
test_x <- model.matrix(SALE.PRICE ~ ., data = test_nyc)[, -1]
test_y <- test_nyc$SALE.PRICE

# transform y with log transformation
#Y_log_train = log(train_y) #Y == Y_log_train
#Y_log_test = log(test_y)


### ---- Regularized Regression ---- ####

#Reference - https://bradleyboehmke.github.io/HOML/regularized-regression.html


##################################################################################
############################# --- Ridge regression ---- ##########################
##################################################################################

ridge <- glmnet(
  x = train_x,
  y = train_y,
  alpha = 0
)
plot(ridge, xvar = "lambda")

#Tuning
#To identify the optimal λ value we can use k-fold cross-validation (CV).

# Apply CV ridge regression
ridge_cv <- cv.glmnet(
  x = train_x,
  y = train_y,
  alpha = 0
)
plot(ridge_cv, main = "Ridge penalty\n\n")


min(ridge_cv$cvm)       # minimum MSE =0.5199817
bestlam.ridge = ridge_cv$lambda.min     # lambda for this min MSE =4.816309

ridge$cvm[ridge$lambda == ridge$lambda.1se]  # 1-SE rule
ridge$lambda.1se  # lambda for this MSE

ridge.pred=predict (ridge ,s=bestlam.ridge, newx=test_x)
test_mse_ridge =  mean((exp(ridge.pred) -exp(test_y))^2)
sqrt(test_mse_ridge) # Test MSE of ridge = 0.5256771   rmse of ridge = 422361.9

ridge.pred_train=predict (ridge ,s=bestlam.ridge, newx=train_x)
train_mse_ridge =  mean((exp(ridge.pred_train) -exp(train_y))^2)
sqrt(train_mse_ridge) # Train rmse of ridge = 428927.9


ridge_min <- glmnet(
  x = train_x,
  y = Y_log_train,
  alpha = 0
)
# plot ridge model
plot(ridge_min, xvar = "lambda", main = "Ridge penalty\n\n")
abline(v = log(ridge_cv$lambda.min), col = "red", lty = "dashed")
abline(v = log(ridge_cv$lambda.1se), col = "blue", lty = "dashed")

coef(ridge_min)


# Calculate the test MAE
test_mae <- mean(abs(exp(ridge.pred) - exp(test_y)))
# Calculate the test MAE % as a percentage of the mean of the actual test values
test_mae_perc <- 100 * test_mae / mean(exp(test_y))
MAPE(exp(ridge.pred), exp(test_y))






##################################################################################
############################# --- Lasso regression ---- ##########################
##################################################################################
lasso <- glmnet(
  x = train_x,
  y = train_y,
  alpha = 1
)
plot(lasso, xvar = "lambda")

#Tuning
#To identify the optimal λ value we can use k-fold cross-validation (CV). 

# Apply CV lasso regression\
lasso_cv <- cv.glmnet(
  x = train_x,
  y = train_y,
  alpha = 1
)
plot(lasso_cv, main = "Lasso penalty\n\n")


min(lasso_cv$cvm)      # minimum MSE = 0.4460874
bestlam.lasso = lasso_cv$lambda.min     # lambda for this min MSE = 0.03397811

lasso.pred=predict (lasso ,s=bestlam.lasso ,newx=test_x)
test_mse_lasso = mean((exp(lasso.pred) -exp(test_y))^2) # Test MSE of lasso = 0.4684231
sqrt(test_mse_lasso)  #rmse_lasso Test = 391075.9

lasso.pred_train=predict (lasso ,s=bestlam.lasso ,newx=train_x)
train_mse_lasso = mean((exp(lasso.pred_train) -exp(train_y))^2) 
sqrt(train_mse_lasso)  #rmse_lasso Train= 396884

lasso_cv$nzero[lasso_cv$lambda == lasso_cv$lambda.min] # No. of  | Min MSE
lasso_cv$cvm[lasso_cv$lambda == lasso_cv$lambda.1se]  # 1-SE rule
lasso_cv$lambda.1se  # lambda for this MSE
lasso_cv$nzero[lasso_cv$lambda == lasso_cv$lambda.1se] # No. of  | 1-SE MSE

lasso_min <- glmnet(
  x = train_x,
  y = Y_log_train,
  alpha = 1
)
# plot lasso model
plot(lasso_min, xvar = "lambda", main = "Lasso penalty\n\n")
abline(v = log(lasso_cv$lambda.min), col = "red", lty = "dashed")
abline(v = log(lasso_cv$lambda.1se), col = "blue", lty = "dashed")


coef(lasso_min, s=0.03397811)


plot(exp(lasso.pred),exp(Y_log_test))


library(ggplot2)

# create a data frame with the coefficients and their names
coef_df <- data.frame(coef = coef(lasso_min, s=0.03397811), 
                      name = names(coef(lasso_min, s=0.03397811)))

# plot the coefficients
ggplot(data = coef_df, aes(x = name, y = coef, fill = coef > 0)) + 
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  xlab("Variable") + 
  ylab("Coefficient") + 
  ggtitle("Lasso Coefficients")

# Calculate the test MAE
test_mae <- mean(abs(exp(lasso.pred) - exp(test_y)))
# Calculate the test MAE % as a percentage of the mean of the actual test values
test_mae_perc <- 100 * test_mae / mean(exp(test_y))













##################################################################################
############################# --- Elastic-net regression ---- ####################
##################################################################################

#The following performs a grid search over 10 values of the alpha parameter between 0–1
#and ten values of the lambda parameter from the lowest to highest lambda values identified by glmnet.

cv_glmnet <- train(
  x = train_x,
  y = train_y,
  method = "glmnet",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# model with lowest RMSE
cv_glmnet$bestTune
#  #   alpha     lambda
# 44   0.5      0.06795621


# results for model with lowest RMSE
cv_glmnet$results %>%
  filter(alpha == cv_glmnet$bestTune$alpha, lambda == cv_glmnet$bestTune$lambda)

# predict sales price on training data
pred <- predict(cv_glmnet, test_x)
# compute RMSE of transformed predicted
RMSE(exp(pred), exp(test_y))

# predict sales price on training data
pred_train <- predict(cv_glmnet, train_x)
# compute RMSE of transformed predicted
RMSE(exp(pred_train), exp(train_y))

vip(cv_glmnet, num_features = 20, geom = "point")



# Calculate the test MAE
test_mae <- mean(abs(exp(pred) - exp(test_y)))
# Calculate the test MAE % as a percentage of the mean of the actual test values
test_mae_perc <- 100 * test_mae / mean(exp(test_y))






##########################################################################################
############################# --- Random Forest regression ---- ##########################
##########################################################################################

# number of features
n_features <- length(setdiff(names(nycprop2), "SALE.PRICE"))

# train a default random forest model
nycprop2_rf1 <- ranger(
  SALE.PRICE ~ ., 
  data = train_nyc,
  mtry = floor(n_features / 3),
  respect.unordered.factors = "order",
  seed = 123
)


default_rmse <- sqrt(nycprop2_rf1$prediction.error) # default rmse = 293351.1

# predict on test set
pred_rf <- predict(nycprop2_rf1, data = test_x)$predictions
# calculate test RMSE
test_rmse_rf <- sqrt(mean((exp(test_y) - exp(pred_rf))^2))
test_rmse_rf

# predict on test set
pred_rf_train <- predict(nycprop2_rf1, data = train_x)$predictions
# calculate test RMSE
train_rmse_rf <- sqrt(mean((exp(train_y) - exp(pred_rf_train))^2))
train_rmse_rf


# Calculate the test MAE
test_mae <- mean(abs(exp(pred_rf) - exp(test_y)))
# Calculate the test MAE % as a percentage of the mean of the actual test values
test_mae_perc <- 100 * test_mae / mean(exp(test_y))
MAPE(exp(pred_rf),exp(test_y))

# -- Hyperparameter Tuning -- #

#the h2o package provides a random grid search that allows you to jump from one random combination 
#to another and it also provides early stopping rules that allow you to stop the grid search once a 
#certain condition is met


h2o.no_progress() #h2o.no_progress()
h2o.init(max_mem_size = "5g")

# convert training data to h2o object
train_h2o <- as.h2o(train_nyc)
test_h20 = as.h2o(test_x)
# set the response column to Sale_Price
response <- "SALE.PRICE"

# set the predictor names
predictors <- setdiff(colnames(train_nyc), response)

#The following fits a default random forest model with h2o
h2o_rf1 <- h2o.randomForest(   
  x = predictors, 
  y = response,
  training_frame = train_h2o, 
  ntrees = n_features * 10,
  seed = 123
)

h2o_rf1

## --- compare rmse of this default h2o rf

## ---



# hyperparameter grid
hyper_grid <- list(
  mtries = floor(n_features * c(.25, .333, .4)),
  min_rows = c( 3, 5, 10),
  max_depth = c( 20, 30),
  sample_rate = c( .632, .70, .80)
)

# random grid search strategy
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.001,   # stop if improvement is < 0.1%
  stopping_rounds = 10,         # over the last 10 models
  max_runtime_secs = 60*40      # or stop search after 15 min.
)


# perform grid search 
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_random_grid",
  x = predictors, 
  y = response, 
  training_frame = train_h2o,
  hyper_params = hyper_grid,
  ntrees = n_features * 10,
  seed = 123,
  stopping_metric = "RMSE",   
  stopping_rounds = 10,           # stop if last 10 trees added 
  stopping_tolerance = 0.005,     # don't improve RMSE by 0.5%
  search_criteria = search_criteria
)

random_grid_perf <- h2o.getGrid(
  grid_id = "rf_random_grid", 
  sort_by = "mse", 
  decreasing = FALSE
)
random_grid_perf

#H2O Grid Details
#Used hyper parameters: 
#-  max_depth 
#-  min_rows 
#-  mtries 
#-  sample_rate 
#Number of models: 38

#Hyper-Parameter Search Summary: ordered by increasing mse

#   max_depth min_rows   mtries sample_rate  model_ids                 mse
#1   20.00000  3.00000 11.00000     0.6320  rf_random_grid_model_3   0.17431

rmse_best_randomforest = sqrt(86116775690.07941) #rmse of best random forest = 293456.6


#h2o_rf1 <- h2o.randomForest(   
  #x = predictors, 
 # y = response,
  #training_frame = train_h2o, 
  #ntrees = n_features * 10,
  #seed = 123
#)

h2o_rf1


# train random forest model with best hyperparameters
nycprop2_rf2 <- ranger(
  SALE.PRICE ~ ., 
  data = train_nyc,
  num.trees = 450,
  mtry = 11,
  min.node.size = 3,
  max.depth = 20,
  sample.fraction = 0.632,
  replace = FALSE,
  importance = "permutation",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed = 123
)


# predict on test set
pred_rf2 <- predict(nycprop2_rf2, data = test_x)$predictions

# calculate test RMSE
test_rmse_rf2 <- sqrt(mean((exp(test_y) - exp(pred_rf2))^2))
test_rmse_rf2


# predict on train set
pred_rf2_train <- predict(nycprop2_rf2, data = train_x)$predictions
# calculate test RMSE
train_rmse_rf2 <- sqrt(mean((exp(train_y) - exp(pred_rf2_train))^2))
train_rmse_rf2

# Calculate the test MAE
test_mae <- mean(abs(exp(pred_rf2) - exp(test_y)))
# Calculate the test MAE % as a percentage of the mean of the actual test values
test_mae_perc <- 100 * test_mae / mean(exp(test_y))
MAPE(exp(pred_rf2),exp(test_y))

p1 <- vip::vip(nycprop2_rf2, num_features = 25, bar = FALSE)


#for h2o defualt tree
# predict on test set
pred_rf2 <- h2o.predict(h2o_rf1, newdata = test_h20)

# calculate test RMSE
test_rmse_rf2 <- sqrt(mean((exp(test_y) - exp(pred_rf2))^2))
test_rmse_rf2


# predict on train set
pred_rf2_train <- predict(nycprop2_rf2, data = train_x)$predictions
# calculate test RMSE
train_rmse_rf2 <- sqrt(mean((train_y - pred_rf2_train)^2))
train_rmse_rf2



# re-run model with impurity-based variable importance
#rf_impurity <- ranger(
#formula = SALE.PRICE ~ ., 
#data = train_nyc, 
#num.trees = n_features * 10,
#mtry = 11,
#min.node.size = 3,
#sample.fraction = .632,
#replace = FALSE,
#importance = "impurity",
#respect.unordered.factors = "order",

#seed  = 123
#)
#sqrt(86982489029)
#p1 <- vip::vip(rf_impurity, num_features = 25, bar = FALSE)

library(randomForest)


# --- using only important variables --- #
names(nycprop2)
nycprop2_rf3 <- ranger(
  SALE.PRICE ~ LAND.SQUARE.FEET+GROSS.SQUARE.FEET+BOROUGH_X1+BOROUGH_X3+BOROUGH_X4+TOTAL.UNITS+Property_Age+
    RESIDENTIAL.UNITS+missing_value+Neighborhood_Category_Lower.Manhattan+Building_class_cat2_class_R, 
  data = train_nyc,
  mtry = floor(12 / 3),
  respect.unordered.factors = "order",
  seed = 123
)

# predict on test set
pred_rf3 <- predict(nycprop2_rf3, data = test_x)$predictions

# calculate test RMSE
test_rmse_rf3 <- sqrt(mean((exp(test_y) - exp(pred_rf3))^2))
test_rmse_rf3


# predict on train set
pred_rf3_train <- predict(nycprop2_rf3, data = train_x)$predictions
# calculate test RMSE
train_rmse_rf3 <- sqrt(mean((exp(train_y) - exp(pred_rf3_train))^2))
train_rmse_rf3

# Calculate the test MAE
test_mae <- mean(abs(exp(pred_rf3) - exp(test_y)))
# Calculate the test MAE % as a percentage of the mean of the actual test values
test_mae_perc <- 100 * test_mae / mean(exp(test_y))
MAPE(exp(pred_rf3),exp(test_y))




# --- using only uncorrelated variables --- #
names(nycprop2)
nycprop2_rf4 <- ranger(
  SALE.PRICE ~ TOTAL.UNITS+BOROUGH_X3+BOROUGH_X4+TAX.CLASS.AT.TIME.OF.SALE_X2+Neighborhood_Category_central_brooklyn+
    Neighborhood_Category_Lower.Manhattan+Neighborhood_Category_mid_islands_staten+Neighborhood_Category_north_shore_staten+
    Neighborhood_Category_northeast_bronx+Neighborhood_Category_northern_brooklyn+Neighborhood_Category_northwest_brooklyn+
    Neighborhood_Category_south_shore_staten+Neighborhood_Category_southeast_bronx+Neighborhood_Category_southern_brooklyn+
    Neighborhood_Category_southwest_bronx+Neighborhood_Category_southwestern_queens+Neighborhood_Category_Upper.Manhattan+
    Building_class_cat2_class_C+Building_class_cat2_class_R+Building_class_cat2_class_other, 
  data = train_nyc,
  mtry = floor(12 / 3),
  respect.unordered.factors = "order",
  seed = 123
)

# predict on test set
pred_rf4 <- predict(nycprop2_rf4, data = test_x)$predictions

# calculate test RMSE
test_rmse_rf4 <- sqrt(mean((exp(test_y) - exp(pred_rf4))^2))
test_rmse_rf4


# predict on train set
pred_rf4_train <- predict(nycprop2_rf4, data = train_x)$predictions
# calculate test RMSE
train_rmse_rf4 <- sqrt(mean((exp(train_y) - exp(pred_rf4_train))^2))
train_rmse_rf4

# Calculate the test MAE
test_mae <- mean(abs(exp(pred_rf4) - exp(test_y)))
# Calculate the test MAE % as a percentage of the mean of the actual test values
test_mae_perc <- 100 * test_mae / mean(exp(test_y))
MAPE(exp(pred_rf4),exp(test_y))

##################################################################################
############################# --- XGBOOST ---- ###################################
##################################################################################

#from chatgpt

# load the xgboost library
library(xgboost)

# convert the data into DMatrix format that xgboost requires
dtrain <- xgb.DMatrix(data = as.matrix(train_x), label = train_y)
dtest <- xgb.DMatrix(data = as.matrix(test_x), label = test_y)

#dtrain <- xgb.DMatrix(data = as.matrix(train_x), label = train_y)
#dtest <- xgb.DMatrix(data = as.matrix(test_x), label = test_y)

# Set up watchlist
# Set up watchlist
watchlist <- list(train = dtrain, test = dtest)


#--- newpart


#---

# specify the hyperparameters for the xgboost model
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.7,
  colsample_bytree = 0.7
)

# train the xgboost model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,
  watchlist = watchlist,
  early_stopping_rounds = 20,
  verbose = 0
)

xgb_model

# use the trained xgboost model to predict on the test set
pred_xgb <- predict(xgb_model, newdata = as.matrix(test_x))
# calculate the root mean squared error (RMSE) of the xgboost model
rmse_xgb <- sqrt(mean((exp(pred_xgb) - exp(test_y))^2))
# print the RMSE of the xgboost model
cat("Test RMSE (xgboost):", rmse_xgb, "\n") #305759.2

# use the trained xgboost model to predict on the train set
pred_xgb_train <- predict(xgb_model, newdata = as.matrix(train_x))
# calculate the root mean squared error (RMSE) of the xgboost model
rmse_xgb_train <- sqrt(mean((exp(pred_xgb_train) - exp(train_y))^2))
# print the RMSE of the xgboost model
cat("Train RMSE (xgboost):", rmse_xgb_train, "\n") #305759.2


# Calculate the test MAE
test_mae <- mean(abs(exp(pred_xgb) - exp(test_y)))
# Calculate the test MAE % as a percentage of the mean of the actual test values
test_mae_perc <- 100 * test_mae / mean(exp(test_y))







# load the caret library
library(mlr)
#install.packages('mlr')

library(xgboost)

grid_tune = expand.grid(nrounds = c(200,300,500),#number of trees
                        max_depth = c(4,6,8),
                        eta =0.1, #c(0.025,0.05,0.1,0.3),#learning rates 0.3
                        gamma = 0.1 , #C(0,0.05,0.1,0.5,0.7,0.9,1.0),#0,#pruning should be tuned, i.e. 
                        colsample_bytree = 0.8, # c(0.4,0.6,0.8,1.0), #subsample-ratio of columns for tree , 1
                        min_child_weight = 1,#c(1,2,3), #the larger, the more conservative models #is: can be used as stop , 1
                        subsample =0.8)#c(0.5,0.75,1.0)) #c(0.5,0.75,1.0) # used for prevevent overfitting by sampling x% ,1

train_control = trainControl(method = "cv",
                             number = 5,#number of folds
                             verboseIter = TRUE,
                             allowParallel = TRUE)

xgb_tune = train(SALE.PRICE~.,data=dtrain,
                 trControl = train_control,
                 tuneGrid = grid_tune,
                 method = "xgbTree",
                 verbose = TRUE)

xgb_tune













# set up the hyperparameter search space
params <- makeParamSet(
  makeNumericParam("eta", lower = 0.01, upper = 0.2),
  makeIntegerParam("max_depth", lower = 2, upper = 10),
  makeNumericParam("subsample", lower = 0.5, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1)
)

# set up the tuning control object to limit the tuning time
ctrl <- makeTuneControlTimeLimit(time_limit = 60)

# perform the hyperparameter tuning using 5-fold cross-validation
tune_res <- tuneParams(
  learner = "regr.xgboost",
  task = makeRegrTask(data = as.data.frame(train_x), target = "SalePrice"),
  resampling = makeResampleDesc("CV", iters = 5),
  measures = list(rmse),
  par.set = params,
  control = ctrl
)

# get the best hyperparameters and train the xgboost model with them
best_params <- tune_res$x
xgb_model <- xgboost(data = dtrain, nrounds = 1000, objective = "reg:squarederror", eval_metric = "rmse", verbose = 0, params = best_params)

# use the trained xgboost model to predict on the test set
pred_xgb <- predict(xgb_model, newdata = as.matrix(test_x))

# calculate the root mean squared error (RMSE) of the xgboost model
rmse_xgb <- sqrt(mean((exp(pred_xgb) - test_y)^2))

# print the RMSE of the xgboost model
cat("RMSE (xgboost):", rmse_xgb, "\n")






















# set up the tuning grid
tune_grid <- expand.grid(
  eta = c(0.01, 0.1, 0.3),
  max_depth = c(3, 6, 9),
  subsample = c(0.6, 0.7, 0.8),
  colsample_bytree = c(0.6, 0.7, 0.8)
)

# set up the control parameters for tuning
ctrl <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE,
  savePredictions = "final"
)

# set the time limit for tuning to 20mins
timeout <- 60 * 20  # in seconds

# perform the hyperparameter tuning
xgb_tuned <- tuneParams(
  x = dtrain,
  y = Y_log_train,
  tuneGrid = tune_grid,
  method = "xgbTree",
  trControl = ctrl,
  tuneLength = 10,  # set the number of tuning iterations
  timeout = timeout,  # set the time limit in seconds
  verbose = TRUE
)
