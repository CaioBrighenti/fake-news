############## LOAD ############## 
## load in libs
library(tidyverse)
library(glmnet)
library(caret)

# load helpers
source("helpers/loaddata.R")
source("helpers/helpers.R")

# load in data
train_articles <- loadFNNTrain()
train <- loadFNNtxtfeat("train")
test <- loadFNNtxtfeat("test")
train_titles <- loadFNNtxtfeat("train_titles")
test_titles <- loadFNNtxtfeat("test_titles")

# individual parts
train_complexity <- loadFNNComplexity("train") %>% filter(!grepl("gossipcop",ID))
train_LIWC <- loadFNNLIWC("train") %>% filter(!grepl("gossipcop",ID))
train_POS <- loadFNNPOS("train") %>% filter(!grepl("gossipcop",ID))
train_NER <- loadFNNNER("train") %>% filter(!grepl("gossipcop",ID))

# LWIC groups
LWIC_groups <- loadLIWCGroups()

############## FIT LASSO ############## 
fitLasso <- function(train_data, seed = 13){
  # Dumy code categorical predictor variables
  x <- model.matrix(label ~ . - ID, train_data)[,-1]
  
  # Convert the outcome (class) to a numerical variable
  y <- train_data$label
  
  # Find the best lambda using cross-validation
  set.seed(seed) 
  print("Starting cross validation")
  cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
  
  #visualize
  plot(cv.lasso)
  cv.lasso$lambda.min
  cv.lasso$lambda.1se
  
  # Fit the final model on the training data
  print("Fitting final model")
  model <- glmnet(x, y, alpha = 1, family = "binomial",
                  lambda = cv.lasso$lambda.1se)
  
  # return final model
  return(model)
}

# fit model
mod.lasso <- fitLasso(train)

# Display regression coefficients
mod.coefs <- coef(mod.lasso)

# get predictions
pred <- getLassoProbs(mod.lasso, test)

# get accuracy
calcAccuracyLasso(mod.lasso, test, pred=pred)

# ROC
getROC(mod.lasso, test, pred)

# try again with data balancing
train2 <- caret::upSample(train, train$label) %>% as_tibble() %>% select(-Class)

# fit model
mod.lasso2 <- fitLasso(train2)

# Display regression coefficients
mod.coefs2 <- coef(mod.lasso2)

# get predictions
pred2 <- getLassoProbs(mod.lasso2, test)

# ROC
getROC(mod.lasso2, test, pred2)

# get accuracy
calcAccuracyLasso(mod.lasso2, test, pred=pred2, cutoff=0.537)

# variable importance
var.imp <- filterVarImp(train[,-c(1,2)], unlist(train[,2]))
plot(var.imp$X0, var.imp$X1)
