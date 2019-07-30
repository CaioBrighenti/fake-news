############## LOAD ############## 
## load in libs
options( java.parameters = "-Xmx6g")
library("tidyverse")
library("tidytext")
library("cleanNLP")
library("rJava")

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

############## SETUP CORENLP ############## 
#cnlp_download_corenlp()
cnlp_init_corenlp()
anno <- cnlp_annotate(train[1,]$text)


############## COMPLEXITY ############## 


############## PSYCHOLOGY ############## 


############## STYLISTIC ############## 