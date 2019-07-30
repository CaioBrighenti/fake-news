############## LOAD ############## 
## load in libs
library("tidyverse")
library("tidytext")

# load in data
source("helpers.R")
source("loaddata.R")
LIAR_train <- loadLIARTrain()
LIAR_test <- loadLIARTest()
FNN_train <- loadFNNTrain()
FNN_test <- loadFNNTest()

# choose dataset
train <- FNN_train
test <- FNN_test

############## TIDY ############## 
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

############## COMPLEXITY ############## 


############## PSYCHOLOGY ############## 


############## STYLISTIC ############## 