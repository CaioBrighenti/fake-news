############## LOAD ############## 
## load in libs
library(tidyverse)

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
# Dumy code categorical predictor variables
x <- model.matrix(label ~ . - ID, train)[,-1]

# Convert the outcome (class) to a numerical variable
y <- train$label


library(glmnet)
# Find the best lambda using cross-validation
set.seed(13) 
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")

#visualize
plot(cv.lasso)
cv.lasso$lambda.min
cv.lasso$lambda.1se

# Fit the final model on the training data
model <- glmnet(x, y, alpha = 1, family = "binomial",
                lambda = cv.lasso$lambda.1se)

# Make predictions on the test data
x.test <- model.matrix(label ~ . - ID, test)[,-1]
probabilities <- model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
# Model accuracy
observed.classes <- test$label
mean(predicted.classes == observed.classes)

# Display regression coefficients
mod.coefs <- coef(model)

as_tibble(data.frame(mod.coefs))
