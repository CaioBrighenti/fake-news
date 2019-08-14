############## LOAD ############## 
## load in libs
options( java.parameters = "-Xmx6g")
library("tidyverse")
library("tidytext")
library("cleanNLP")
library("rJava")
library("progress")

# load in data
source("helpers.R")
source("loaddata.R")
LIAR_train <- loadLIARTrain()
LIAR_train$text <- as.character(LIAR_train$text)
LIAR_test <- loadLIARTest()
FNN_train <- loadFNNTrain()
FNN_test <- loadFNNTest()

# choose dataset
train <- FNN_train
test <- FNN_test

############## TIDY ############## 
# tidy train
train <- train %>% 
  as_tibble() %>%
  mutate(text = as.character(text), ID = as.character(ID)) %>%
  filter(nchar(text) > 0 & nchar(text) < 100000) %>%
  dplyr::select(ID, label, text)

# tidy test
test <- test %>% 
  as_tibble() %>%
  mutate(text = as.character(text), ID = as.character(ID)) %>%
  filter(nchar(text) > 0 & nchar(text) < 100000) %>%
  dplyr::select(ID, label, text)

# unnest tokens
tidy_train <- train %>% 
  unnest_tokens(word, text)

# clean tokens
tidy_train <- tidy_train %>%
  anti_join(stop_words)

############## CORENLP SYNTAX TREES FROM PYTHON ############## 
## coreNLP server must be running at localhost:9000
## timed out observations return (-1,-1,-1)
library("reticulate")
use_python("C:/Users/Caio Brighenti/AppData/Local/Programs/Python/Python37", required = T)
py_config()
source_python("processCoreNLP.py")
## create empty dataframe
train_depths <- tibble(ID = train$ID,
                       mu_sentence = rep(0, nrow(train)),
                       mu_verb_phrase = rep(0, nrow(train)),
                       mu_noun_phrase = rep(0, nrow(train)), 
                       sd_sentence = rep(0, nrow(train)),
                       sd_verb_phrase = rep(0, nrow(train)),
                       sd_noun_phrase = rep(0, nrow(train)),
                       iqr_sentence = rep(0, nrow(train)),
                       iqr_verb_phrase = rep(0, nrow(train)),
                       iqr_noun_phrase = rep(0, nrow(train)),
                       num_verb_phrase = rep(0, nrow(train)))
## calculate tree depths for each document
pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = nrow(train))
for (idx in 1:nrow(train)) {
  pb$tick()
  if (train_depths[idx,]$mu_sentence == 0) {
    t_depths <- getConstTreeDepths(train[idx,]$text)
    train_depths[idx,] <- c(train_depths[idx,]$ID, t_depths)
  }
}


## check distribution
train_labels <- train %>%
  dplyr::select(ID, label)
train_depths <- train_depths %>%
  left_join(train_labels, by="ID")
train_depths %>%
  filter(mu_sentence != 0) %>%
  group_by(label) %>%
  summarise(mu_sentence = mean(mu_sentence), mu_verb_phrase = mean(mu_verb_phrase), mu_noun_phrase = mean(mu_noun_phrase))

# write to file
# write_tsv(train_depths, "coreNLP_annotations/fnn_train_trees.tsv")


############## CALCULATE COMPLEXITY ############## 
# syntax tree depths
train_depths <- read.csv(file="coreNLP_annotations/fnn_train_trees.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
  as_tibble() %>%
  mutate(ID = as.character(ID))
test_depths <- read.csv(file="coreNLP_annotations/fnn_test.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
  as_tibble() %>%
  mutate(ID = as.character(ID))

# readability
library("quanteda")
## FOG
FOG_scores <- quanteda::textstat_readability(train$text, measure = "FOG")
train_read <- train %>%
  mutate(FOG = FOG_scores$FOG)

## SMOG
SMOG_scores <- quanteda::textstat_readability(train$text, measure = "SMOG")
train_read <- train_read %>%
  mutate(SMOG = SMOG_scores$SMOG)

## Flesch-Kincaid
FK_scores <- quanteda::textstat_readability(train$text, measure = "Flesch.Kincaid")
train_read <- train_read %>%
  mutate(FK = FK_scores$Flesch.Kincaid)

## Coleman-Liau
CL_scores <- quanteda::textstat_readability(train$text, measure = "Coleman.Liau")
train_read <- train_read %>%
  mutate(CL = CL_scores$Coleman.Liau)

## ARI
ARI_scores <- quanteda::textstat_readability(train$text, measure = "ARI")
train_read <- train_read %>%
  mutate(ARI = ARI_scores$ARI)

## compare readability
train_read %>%
  group_by(label) %>%
  summarise(FOG = mean(FOG), SMOG = mean(SMOG), FK = mean(FK), CL = mean(CL), ARI = mean(ARI))

# basic stats
cnlp_init_tokenizers()
train_anno <- cnlp_annotate(train$text, as_strings = TRUE)

## mean sentence word counts
train_swc <- train_anno$token %>% 
  mutate(num_id = as.numeric(str_remove(id, "doc"))) %>%
  group_by(id) %>%
  count(sid, num_id) %>%
  summarise(swc = median(n), num_id = first(num_id)) %>%
  arrange(num_id) %>%
  mutate(ID = train$ID, label = train$label, text = train$text) %>%
  dplyr::select(ID, label, text, swc)

## mean word length
train_wl <- train_anno$token %>%
  mutate(num_id = as.numeric(str_remove(id, "doc"))) %>%
  mutate(wlen = nchar(word)) %>% 
  group_by(id) %>%
  summarise(wlen = mean(wlen), num_id = first(num_id)) %>%
  arrange(num_id) %>%
  mutate(ID = train$ID, label = train$label, text = train$text) %>%
  dplyr::select(ID, label, text, wlen)
  
## type-token ratio
train_totals <- train_anno$token %>%
  count(id)
train_ttr <- train_anno$token %>%
  mutate(num_id = as.numeric(str_remove(id, "doc"))) %>%
  left_join(train_totals) %>% 
  group_by(id) %>%
  summarise(types = n_distinct(word), tokens = first(n), num_id = first(num_id)) %>%
  arrange(num_id) %>%
  mutate(ID = train$ID, label = train$label, text = train$text, TTR = types / tokens) %>%
  dplyr::select(ID, label, text, types, tokens, TTR)

train_ttr %>%
  group_by(label) %>%
  summarise(TTR = mean(TTR))


## merge
train_complexity <- train %>%
  left_join(train_depths, by=c("ID", "label")) %>%
  left_join(train_swc, by=c("ID", "label", "text")) %>%
  left_join(train_wl, by=c("ID", "label", "text")) %>%
  left_join(train_ttr, by=c("ID", "label", "text"))

# write to file
#write_tsv(train_complexity, "text_features/fnn_train_complexity.tsv")


############## LOAD COMPLEXITY ############## 
# read from file
train_complexity <- read.csv(file="text_features/fnn_train_complexity.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
  as_tibble() %>%
  mutate(ID = as.character(ID))

## evaluate var imp and aov
complexity_ranks <- getVarRanks(train_complexity)

############## LOAD LIWC ############## 
train_LIWC<-read.csv(file="FakeNewsNet/dataset/LIWC2015_fnn_train.csv",header = TRUE, encoding="UTF-8") %>%
  as_tibble()
train_LIWC <- train_LIWC %>%
  mutate(ID = A, label = B, title = C, text = D) %>%
  select(ID, label, title, text, WC:OtherP)

## evaluate var imp and aov
LIWC_ranks <- train_LIWC %>%
  dplyr::select(-title) %>%
  getVarRanks()

############## PSYCHOLOGY ############## 


############## POS TAGS FROM CORENLP ############## 
library("reticulate")
use_python("C:/Users/Caio Brighenti/AppData/Local/Programs/Python/Python37", required = T)
py_config()
source_python("processCoreNLP.py")
## create empty dataframe
train_POS <- tibble(ID = train$ID, CC = rep(-1, nrow(train)), CD = rep(-1, nrow(train)), DT = rep(-1, nrow(train)),
  EX = rep(-1, nrow(train)), FW = rep(-1, nrow(train)),IN = rep(-1, nrow(train)),JJ = rep(-1, nrow(train)),
  JJR = rep(-1, nrow(train)),JJS = rep(-1, nrow(train)),LS = rep(-1, nrow(train)),MD = rep(-1, nrow(train)),
  NN = rep(-1, nrow(train)),NNS = rep(-1, nrow(train)),NNP = rep(-1, nrow(train)),NNPS = rep(-1, nrow(train)),
  PDT = rep(-1, nrow(train)),POS = rep(-1, nrow(train)),PRP = rep(-1, nrow(train)),`PRP$` = rep(-1, nrow(train)),
  RB = rep(-1, nrow(train)),RBR = rep(-1, nrow(train)),RBS = rep(-1, nrow(train)),RP = rep(-1, nrow(train)),
  SYM = rep(-1, nrow(train)),TO = rep(-1, nrow(train)),UH = rep(-1, nrow(train)),VB = rep(-1, nrow(train)),
  VBD = rep(-1, nrow(train)),VBG = rep(-1, nrow(train)),VBN = rep(-1, nrow(train)),VBP = rep(-1, nrow(train)),
  VBZ = rep(-1, nrow(train)),WDT = rep(-1, nrow(train)),WP = rep(-1, nrow(train)),`WP$` = rep(-1, nrow(train)),
  WRB = rep(-1, nrow(train)))
## get POS counts 
pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = nrow(train))
for (idx in 1:nrow(train)) {
  pb$tick()
  if (train_POS[idx,]$CC == -1) {
    t_POS <- getPOSCounts(train[idx,]$text)
    train_POS[idx,] <- c(train_POS[idx,]$ID, t_POS)
  }
}

## check distribution
train_labels <- train %>%
  dplyr::select(ID, label)
train_POS <- train_POS %>%
  left_join(train_labels, by="ID")
train_POS %>%
  filter(CC != -1) %>%
  group_by(label) %>%
  summarise_at(vars(CC:WRB), mean, na.rm = TRUE)


# write to file
write_tsv(train_POS, "coreNLP_annotations/fnn_train_POS.tsv")

############## STYLISTIC ############### 
# POS tags
train_POS <- read.csv(file="coreNLP_annotations/fnn_train_POS.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
  as_tibble() %>%
  mutate(ID = as.character(ID)) %>%
  select(ID, label, everything())

train_POS %>%
  dplyr::select(-ID) %>%
  filterVarImp(.,.$label) %>%
  mutate(var = row.names(.), mu = (fake + real) / 2) %>%
  arrange(desc(mu)) %>%
  select(var, mu, everything())

# LIWC features


