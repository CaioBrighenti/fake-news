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
