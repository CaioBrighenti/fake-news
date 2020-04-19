############## LOAD ############## 
## load in libs
library(tidyverse)
library(glmnet)
library(caret)
library(xgboost)
library(ggrepel)
library(margins)
library(extrafont)
library(lubridate)

# load fonts
#font_import(paths = "C:/Users/caiob/Downloads/Roboto")
loadfonts(device = "win", quiet = TRUE)

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

# scale data
train[,-c(1,2)] <- scale(train[,-c(1,2)])
test[,-c(1,2)] <- scale(test[,-c(1,2)])

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
train2 <- caret::downSample(train, train$label) %>% as_tibble() %>% select(-Class)

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

# grab reduced variables
vars_kept <- names(mod.coefs2[,1][which(abs(mod.coefs2[,1]) > 0)])

### refit to get p-values
train3 <- dplyr::select(train2, ID, label, vars_kept)
test2 <- dplyr::select(test, ID, label, vars_kept)

# fit model
mod3 <- glm(label ~ ., data = train3[,-1], family="binomial")
summary(mod3)

# get pvals
mod.coefs3 <- coef(mod3)
mod.pvals3 <- coef(summary(mod3))[,4]

# get predictions
pred3 <- predict(mod3, test2, type="response")

# ROC
getROC(mod3, test2, pred3)

# get accuracy
calcAccuracyLasso(mod3, test2, pred=pred3, cutoff=0.5)


############## XGBOOST ############## 
bst <- xgboost(data = as.matrix(train2[,-c(1,2)]), 
               label = as.character(train2$label), 
               max.depth = 2, 
               eta = 1, 
               nthread = 2, 
               nrounds = 2,
               objective = "binary:logistic")

bst.pred <- predict(bst, as.matrix(test[,-c(1,2)]))

# get accuracy
calcAccuracyLasso(bst, test, pred=bst.pred)

# ROC
getROC(bst.pred, test, bst.pred)




############## GET IMPORTANT VARIABLES ############## 
# variable importance
var.imp <- filterVarImp(train3[,-c(1,2)], unlist(train3[,2]))
var.imp <- tibble(
  var = row.names(var.imp),
  varimp = var.imp[,1]
)

# get coefficients
coefs <- tibble(
  var = names(mod.coefs3),
  coef = mod.coefs3, 
  pval = mod.pvals3
) %>%
  mutate(
    pval_group = case_when(
      pval <= 0.001 ~ "< 0.001",
      pval <= 0.01 ~ "< 0.01",
      pval <= 0.05 ~ "< 0.05",
      pval <= 0.1 ~ "< 0.1",
      TRUE ~ "> 0.1"
    )
  )


# merge together
coefs_merge <- left_join(coefs,var.imp) %>%
  filter(var != "(Intercept)")

# plot
coefs_merge %>%
  filter(pval_group != "> 0.1") %>%
  ggplot(aes(x=coef,y=varimp)) +
  geom_point(aes(shape=pval_group)) +
  geom_vline(xintercept = 0, linetype="longdash", alpha=0.5) + 
  geom_text_repel(aes(label = var)) +
  theme_minimal() +
  labs(
    x = "Coefficient",
    y = "Variable Importance",
    title = "Most important and impactful variables for fake news classification",
    caption = "All variables rescaled to mean 0 and SD 1",
    shape = "p-value group"
  ) +
  theme(
    legend.position = "bottom",
    text = element_text(family = "Roboto")
  )


############## INTERACTIONS ############## 
mod4 <- glm(label ~ . ^2, data = train3[,-1], family = "binomial")
summary(mod4)

# get predictions
pred4 <- predict(mod4, test2, type="response")

# ROC
getROC(mod4, test2, pred4)

# get accuracy
calcAccuracyLasso(mod.lasso2, test, pred=pred2, cutoff=0.5)

mod.meffects4 <- margins(mod4)

meffects <- as_tibble(summary(mod.meffects4)) %>%
  rename(var = factor) %>%
  mutate(pval_group = case_when(
    p <= 0.001 ~ "< 0.001",
    p <= 0.05 ~ "< 0.05",
    p <= 0.01 ~ "< 0.01",
    TRUE ~ "> 0.01"
  ))

# meffects plot
plot(mod.meffects4)

#ggplot version
meffects %>%
  mutate(var = fct_reorder(var, -AME)) %>%
  ggplot(aes(x=AME,y=var)) +
  geom_vline(xintercept = 0, linetype="longdash", alpha=0.3) + 
  geom_point() + 
  geom_errorbar(aes(xmin=lower, xmax=upper)) +
  theme_minimal() +
  labs(
    x = "Average Marginal Effect",
    y = "Variable",
    title = "Average marginal effect by variable"
  ) +
  theme(
    panel.grid.major.y = element_blank()
  )


meffects_merge <- left_join(meffects, var.imp)

# plot
meffects_merge %>%
  ggplot(aes(x=AME,y=varimp)) +
  geom_point(aes(shape=pval_group)) +
  geom_vline(xintercept = 0, linetype="longdash", alpha=0.5) + 
  geom_text_repel(aes(label = var), family = "Roboto") +
  theme_minimal() +
  labs(
    x = "Average Marginal Effect",
    y = "Variable Importance",
    title = "Most important and impactful variables for fake news classification",
    shape = "p-value"
  ) +
  theme(
    legend.position = "bottom",
    text = element_text(family = "Roboto")
  )

