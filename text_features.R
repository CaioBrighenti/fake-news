############## LOAD ############## 
## load in libs
options( java.parameters = "-Xmx6g")
library("tidyverse")
library("tidytext")
library("cleanNLP")
library("rJava")
library("progress")
library("MASS")

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


## add label
train_labels <- train %>%
  dplyr::select(ID, label)
train_depths <- train_depths %>%
  left_join(train_labels, by="ID")

# write to file
# write_tsv(train_depths, "coreNLP_annotations/fnn_train_trees.tsv")



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
#write_tsv(train_POS, "coreNLP_annotations/fnn_train_POS.tsv")

############## NER TAGS FROM CORENLP ############## 
library("reticulate")
use_python("C:/Users/Caio Brighenti/AppData/Local/Programs/Python/Python37", required = T)
py_config()
source_python("processCoreNLP.py")
## create empty dataframe
test_NER <- tibble(ID = test$ID, NER = rep(-1, nrow(test)))
## get NER counts 
pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = nrow(test))
for (idx in 1:nrow(test)) {
  pb$tick()
  if (test_NER[idx,]$NER == -1) {
    t_NER <- getNERCounts(test[idx,]$text)
    test_NER[idx,] <- c(test_NER[idx,]$ID, t_NER)
  }
}

## check distribution
test_labels <- test %>%
  dplyr::select(ID, label)
test_NER <- test_NER %>%
  left_join(test_labels, by="ID") %>%
  mutate(NER = as.numeric(NER))
test_NER %>%
  filter(NER != -1) %>%
  group_by(label) %>%
  summarise_at(vars(NER), mean, na.rm = TRUE)


# write to file
write_tsv(test_NER, "coreNLP_annotations/fnn_test_NER.tsv")

############## CALCULATE COMPLEXITY ############## 
# syntax tree depths
train_depths <- read.csv(file="coreNLP_annotations/fnn_train_trees.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
  as_tibble() %>%
  mutate(ID = as.character(ID))
test_depths <- read.csv(file="coreNLP_annotations/fnn_test_trees.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
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
write_tsv(train_complexity, "text_features/fnn_train_complexity.tsv")


############## LOAD COMPLEXITY ############## 
# read from file
train_complexity <- loadFNNComplexity('train')
test_complexity <- loadFNNComplexity('test')


## evaluate var imp and aov
complexity_ranks <- getVarRanks(train_complexity)
test_complexity_ranks <- getVarRanks(test_complexity)

############## LIWC PREDICTOR GROUPS ############### 
LIWC_groups <- tibble(
  WC = "summary",
  Analytic = "summary",	
  Clout = "summary",	
  Authentic = "summary",	
  Tone = "summary",
  WPS = "summary",	
  Sixltr = "summary",	
  Dic = "summary",	
  function. = "function",
  pronoun = "function",	
  ppron = "function",	
  i = "function",	
  we = "function",	
  you = "function",	
  shehe = "function",	
  they = "function",	
  ipron = "function",	
  article = "function",	
  prep = "function",	
  auxverb = "function",	
  adverb = "function",	
  conj = "function",	
  negate = "function",	
  verb = "othergram",	
  adj = "othergram",	
  compare = "othergram",		
  interrog = "othergram",	
  number = "othergram",		
  quant = "othergram",	
  affect = "affect",
  posemo = "affect",
  negemo = "affect",
  anx = "affect",
  anger = "affect",	
  sad = "affect",	
  social = "social",
  family = "social",
  friend = "social",
  female = "social",	
  male = "social",
  cogproc	= "cogproc",
  insight	= "cogproc",
  cause	= "cogproc",
  discrep	= "cogproc",	
  tentat	= "cogproc",	
  certain	= "cogproc",	
  differ	= "cogproc",
  percept	= "percept",
  see	= "percept",	
  hear = "percept",
  feel = "percept",
  bio = "bio",
  body = "bio",
  health = "bio",
  sexual = "bio",	
  ingest = "bio",
  drives = "drives",
  affiliation = "drives",
  achieve = "drives",
  power = "drives",	
  reward = "drives",
  risk = "drives",	
  focuspast	= "timeorient",
  focuspresent	= "timeorient",	
  focusfuture	= "timeorient",	
  relativ = "relativ",	
  motion = "relativ",		
  space = "relativ",		
  time = "relativ",		
  work = "personc",
  leisure = "personc",
  home = "personc",	
  money = "personc",	
  relig = "personc",	
  death = "personc",	
  informal = "informal",	
  swear = "informal",		
  netspeak = "informal",		
  assent = "informal",		
  nonflu = "informal",		
  filler = "informal",		
  AllPunc = "punc",
  Period = "punc",	
  Comma = "punc",	
  Colon = "punc",	
  SemiC = "punc",	
  QMark = "punc",	
  Exclam = "punc",	
  Dash = "punc",	
  Quote = "punc",	
  Apostro = "punc",	
  Parenth = "punc",	
  OtherP = "punc"
) %>%
  gather(var, group)

############## LOAD LIWC ############## 
## train
train_LIWC<-read.csv(file="FakeNewsNet/dataset/LIWC2015_fnn_train.csv",header = TRUE, encoding="UTF-8") %>%
  as_tibble() %>%
  mutate(ID = as.character(A), label = B, title = as.character(C), text = as.character(D)) %>%
  dplyr::select(ID, label, -title, -text, WC:OtherP)
## test
test_LIWC<-read.csv(file="FakeNewsNet/dataset/LIWC2015_fnn_test.csv",header = TRUE, encoding="UTF-8") %>%
  as_tibble() %>%
  mutate(ID = as.character(A), label = B, title = as.character(C), text = as.character(D)) %>%
  dplyr::select(ID, label, -title, -text, WC:OtherP)

## evaluate var imp and aov
LIWC_ranks <- train_LIWC %>%
  getVarRanks()
## check by groups
LIWC_ranks %>%
  left_join(LIWC_groups, by = "var") %>%
  group_by(group) %>%
  summarise(varimp_rank = mean(varimp_rank), p_rank = mean(p_rank), avg_rank = mean(avg_rank)) %>%
  arrange(avg_rank)


############## STYLISTIC ############### 
# POS tags
train_POS <- read.csv(file="coreNLP_annotations/fnn_train_POS.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
  as_tibble() %>%
  mutate(ID = as.character(ID)) %>%
  dplyr::select(ID, label, everything())
test_POS <- read.csv(file="coreNLP_annotations/fnn_test_POS.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
  as_tibble() %>%
  mutate(ID = as.character(ID)) %>%
  dplyr::select(ID, label, everything())


## evaluate var imp and aov
POS_ranks <- train_POS %>%
  getVarRanks()

############## MERGE ############### 
## merge datasets
train_txtfeat <- train_complexity %>%
  left_join(train_LIWC, by = c("ID", "label")) %>%
  left_join(train_POS, by = c("ID", "label")) %>%
  mutate(label = as.factor(2 - unclass(label))) %>%
  dplyr::select(-ID)
test_txtfeat <- test_complexity %>%
  left_join(test_LIWC, by = c("ID", "label")) %>%
  left_join(test_POS, by = c("ID", "label")) %>%
  mutate(label = as.factor(2 - unclass(label))) %>%
  dplyr::select(-ID)

## merge ranks
txtfeat_ranks <- complexity_ranks %>%
  full_join(LIWC_ranks) %>%
  full_join(POS_ranks) %>%
  dplyr::select(var, max_varimp, p_val) %>%
  mutate(varimp_rank = rank(-max_varimp), p_rank = rank(p_val), avg_rank = (varimp_rank + p_rank) / 2) %>%
  arrange(avg_rank)

## upsample
txtfeat_fit <- upSample(train_txtfeat, train_txtfeat$label) %>%
  dplyr::select(-Class) %>%
  as_tibble()


############## MODEL ############### 
mod <-  glm(label ~ .,
             data=txtfeat_fit, family="binomial")
summary(mod)
getROC(mod, train_txtfeat)
calcAccuracyLR(mod, train_txtfeat, cutoff = 0.5)


