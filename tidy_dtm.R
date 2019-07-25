############## LOAD ############## 
## load in libs
library("tidyverse")
library("tidytext")
library("tm")
library("caret")
library("textstem")
library("glmnet")

# load in data
source("helpers.R")
source("loaddata.R")
LIAR_train <- loadLIARTrain()
LIAR_test <- loadLIARTest()
FNN_train <- loadFNNTrain()
FNN_test <- loadFNNTest()

## chose dataset
train <- FNN_train
test <- FNN_test
train$label <- as.factor(2-unclass(train$label))
test$label <- as.factor(2-unclass(test$label))
train <- caret::downSample(train, train$label) %>%
  as_tibble() %>%
  select(-Class)

############## PREP TRAIN DATA ############## 
# tidy train
train <- train %>% 
  mutate(text = as.character(text), ID = as.character(ID)) %>%
  filter(nchar(text) > 0) %>%
  dplyr::select(-title)

# unnest tokens
tidy_train <- train %>% 
  unnest_tokens(word, text)

# clean tokens
tidy_train <- tidy_train %>%
  anti_join(stop_words)

# get counts
train_counts <- tidy_train %>%
  rename(label.x=label) %>%
  group_by(ID) %>%
  count(ID, label.x, word, sort = FALSE) %>%
  group_by(word) %>%
  mutate(total = sum(n))

# prune vocab
train_counts <- train_counts %>%
  filter(total >= 1000)

# bind tf-idf
# train_counts %>%
#   bind_tf_idf(word, ID, n)

# make DTM
train_dtm <- train_counts %>%
  cast_dtm(ID, word, n   )#, weighting = tm::weightTfIdf)

#get labels
train_dtm_df <- train_dtm %>%
  as.matrix() %>%
  as_tibble(rownames = "ID")
train_labels <- train %>%
  dplyr::select(ID, `_label` = label)
train_dtm_df <- train_dtm_df %>%
  left_join(train_labels, by="ID") %>%
  dplyr::select(-ID)



############## PREP TEST DATA ############## 
# tidy train
test <- test %>% 
  mutate(text = as.character(text), ID = as.character(ID)) %>%
  filter(nchar(text) > 0) %>%
  dplyr::select(-title)

# unnest tokens
tidy_test <- test %>% 
  unnest_tokens(word, text)

# clean tokens
tidy_test <- tidy_test %>%
  intersect(tidy_train, word)

# get counts
test_counts <- tidy_test %>%
  rename(label.x=label) %>%
  group_by(ID) %>%
  count(ID, label.x, word, sort = FALSE) %>%
  group_by(word) %>%
  mutate(total = sum(n))

# make DTM
test_dtm <- test_counts %>%
  cast_dtm(ID, word, n   )#, weighting = tm::weightTfIdf)

#get labels
test_dtm_df <- test_dtm %>%
  as.matrix() %>%
  as_tibble(rownames = "ID")
test_labels <- test %>%
  dplyr::select(ID, `_label` = label)
test_dtm_df <- test_dtm_df %>%
  left_join(test_labels, by="ID") %>%
  dplyr::select(-ID)

############## FIT MODEL ############## 
# fit model
mod.logit<-glm(`_label` ~ . - ID, data=train_dtm_df,family="binomial")
summary(mod.logit)

# get coefficients
word_coef <- coef(mod.logit)
head(word_coef[order(-word_coef)], 20)
head(word_coef[order(word_coef)], 20)
  
# test accuracy
stats.logit_train<-calcAccuracyLR(mod.logit, train_dtm_df)
stats.logit_train<-calcAccuracyLR(mod.logit, test_dtm_df)

# compare lengths
train_lengths <- train %>%
  mutate(length = nchar(text)) %>%
  group_by(label) %>%
  summarise(length = mean(length))

# compare word count
tidy_wcount <- tidy_train %>%
  group_by(ID) %>%
  count(label,word) %>%
  mutate(wcount = sum(n))
tidy_wcount %>%
  group_by(label) %>%
  summarise(wcount = mean(wcount))

