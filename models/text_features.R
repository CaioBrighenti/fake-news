############## LOAD ############## 
## load in libs
options( java.parameters = "-Xmx6g")
library("tidyverse")
library("tidytext")
library("cleanNLP")
library("rJava")
library("progress")
library("MASS")
library("car")
library("glmnet")
library("ggrepel")

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

# define colors
colgate_ter <- c("#64A50A", "#F0AA00","#0096C8", "#005F46","#FF6914","#004682")

############## TIDY ############## 
# tidy train
train <- train %>% 
  as_tibble() %>%
  mutate(text = as.character(title), ID = as.character(ID)) %>%
  filter(nchar(text) > 0 & nchar(text) < 100000) %>%
  dplyr::select(ID, label, text)

# tidy test
test <- test %>% 
  as_tibble() %>%
  mutate(text = as.character(text), ID = as.character(ID)) %>%
  filter(nchar(text) > 5 & nchar(text) < 100000) %>%
  dplyr::select(ID, label, text)

############## CORENLP SYNTAX TREES FROM PYTHON ############## 
## coreNLP server must be running at localhost:9000
## timed out observations return (-1,-1,-1)
library("reticulate")
use_python("C:/Users/Caio Brighenti/AppData/Local/Programs/Python/Python37", required = T)
py_config()
source_python("processCoreNLP.py")
## create empty dataframe
test_depths <- tibble(ID = test$ID,
                       mu_sentence = rep(0, nrow(test)),
                       mu_verb_phrase = rep(0, nrow(test)),
                       mu_noun_phrase = rep(0, nrow(test)), 
                       sd_sentence = rep(0, nrow(test)),
                       sd_verb_phrase = rep(0, nrow(test)),
                       sd_noun_phrase = rep(0, nrow(test)),
                       iqr_sentence = rep(0, nrow(test)),
                       iqr_verb_phrase = rep(0, nrow(test)),
                       iqr_noun_phrase = rep(0, nrow(test)),
                       num_verb_phrase = rep(0, nrow(test)))
## calculate tree depths for each document
pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = nrow(test))
for (idx in 1:nrow(test)) {
  pb$tick()
  if (test_depths[idx,]$mu_sentence == 0) {
    t_depths <- getConstTreeDepths(test[idx,]$text)
    test_depths[idx,] <- c(test_depths[idx,]$ID, t_depths)
  }
}


## add label
test_labels <- test %>%
  dplyr::select(ID, label)
test_depths <- test_depths %>%
  left_join(test_labels, by="ID")

# write to file
#write_tsv(test_depths, "annotations/coreNLP/fnn_test_titles_trees.tsv")



############## POS TAGS FROM CORENLP ############## 
library("reticulate")
use_python("C:/Users/Caio Brighenti/AppData/Local/Programs/Python/Python37", required = T)
py_config()
source_python("processCoreNLP.py")
## create empty dataframe
test_POS <- tibble(ID = test$ID, CC = rep(-1, nrow(test)), CD = rep(-1, nrow(test)), DT = rep(-1, nrow(test)),
                    EX = rep(-1, nrow(test)), FW = rep(-1, nrow(test)),IN = rep(-1, nrow(test)),JJ = rep(-1, nrow(test)),
                    JJR = rep(-1, nrow(test)),JJS = rep(-1, nrow(test)),LS = rep(-1, nrow(test)),MD = rep(-1, nrow(test)),
                    NN = rep(-1, nrow(test)),NNS = rep(-1, nrow(test)),NNP = rep(-1, nrow(test)),NNPS = rep(-1, nrow(test)),
                    PDT = rep(-1, nrow(test)),POS = rep(-1, nrow(test)),PRP = rep(-1, nrow(test)),`PRP$` = rep(-1, nrow(test)),
                    RB = rep(-1, nrow(test)),RBR = rep(-1, nrow(test)),RBS = rep(-1, nrow(test)),RP = rep(-1, nrow(test)),
                    SYM = rep(-1, nrow(test)),TO = rep(-1, nrow(test)),UH = rep(-1, nrow(test)),VB = rep(-1, nrow(test)),
                    VBD = rep(-1, nrow(test)),VBG = rep(-1, nrow(test)),VBN = rep(-1, nrow(test)),VBP = rep(-1, nrow(test)),
                    VBZ = rep(-1, nrow(test)),WDT = rep(-1, nrow(test)),WP = rep(-1, nrow(test)),`WP$` = rep(-1, nrow(test)),
                    WRB = rep(-1, nrow(test)))
## get POS counts 
pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = nrow(test))
for (idx in 1:nrow(test)) {
  pb$tick()
  if (test_POS[idx,]$CC == -1) {
    t_POS <- getPOSCounts(test[idx,]$text)
    test_POS[idx,] <- c(test_POS[idx,]$ID, t_POS)
  }
}

## check distribution
test_labels <- test %>%
  dplyr::select(ID, label)
test_POS <- test_POS %>%
  left_join(test_labels, by="ID")
test_POS %>%
  filter(CC != -1) %>%
  group_by(label) %>%
  summarise_at(vars(CC:WRB), mean, na.rm = TRUE)


# write to file
#write_tsv(test_POS, "annotations/coreNLP/fnn_test_titles_POS.tsv")

############## NER TAGS FROM CORENLP ############## 
library("reticulate")
use_python("C:/Users/Caio Brighenti/AppData/Local/Programs/Python/Python37", required = T)
py_config()
source_python("processCoreNLP.py")
## create empty dataframe
train_NER <- tibble(ID = train$ID, NER = rep(-1, nrow(train)))
## get NER counts 
pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = nrow(train))
for (idx in 1:nrow(train)) {
  pb$tick()
  if (train_NER[idx,]$NER == -1) {
    t_NER <- getNERCounts(train[idx,]$text)
    train_NER[idx,] <- c(train_NER[idx,]$ID, t_NER)
  }
}

## check distribution
train_labels <- train %>%
  dplyr::select(ID, label)
train_NER <- train_NER %>%
  left_join(train_labels, by="ID") %>%
  mutate(NER = as.numeric(NER))
train_NER %>%
  filter(NER != -1) %>%
  group_by(label) %>%
  summarise_at(vars(NER), mean, na.rm = TRUE)


# write to file
write_tsv(train_NER, "annotations/coreNLP/fnn_train_titles_NER.tsv")

############## CALCULATE COMPLEXITY ############## 
# syntax tree depths
train_depths <- read.csv(file="annotations/coreNLP/fnn_train_trees.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
  as_tibble() %>%
  mutate(ID = as.character(ID))
test_depths <- read.csv(file="annotations/coreNLP/fnn_test_titles_trees.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
  as_tibble() %>%
  mutate(ID = as.character(ID))

# readability
library("quanteda")
## FOG
FOG_scores <- quanteda::textstat_readability(test$text, measure = "FOG")
test_read <- test %>%
  mutate(FOG = FOG_scores$FOG)

## SMOG
SMOG_scores <- quanteda::textstat_readability(test$text, measure = "SMOG")
test_read <- test_read %>%
  mutate(SMOG = SMOG_scores$SMOG)

## Flesch-Kincaid
FK_scores <- quanteda::textstat_readability(test$text, measure = "Flesch.Kincaid")
test_read <- test_read %>%
  mutate(FK = FK_scores$Flesch.Kincaid)

## Coleman-Liau
CL_scores <- quanteda::textstat_readability(test$text, measure = "Coleman.Liau")
test_read <- test_read %>%
  mutate(CL = CL_scores$Coleman.Liau)

## ARI
ARI_scores <- quanteda::textstat_readability(test$text, measure = "ARI")
test_read <- test_read %>%
  mutate(ARI = ARI_scores$ARI)

## compare readability
test_read %>%
  group_by(label) %>%
  summarise(FOG = mean(FOG), SMOG = mean(SMOG), FK = mean(FK), CL = mean(CL), ARI = mean(ARI))

test_read <- test_read %>%
  dplyr::select(-text)

# basic stats
cnlp_init_tokenizers()
test_anno <- cnlp_annotate(test$text, as_strings = TRUE)

## mean sentence word counts
test_swc <- test_anno$token %>% 
  mutate(num_id = as.numeric(str_remove(id, "doc"))) %>%
  group_by(id) %>%
  count(sid, num_id) %>%
  summarise(swc = median(n), num_id = first(num_id)) %>%
  arrange(num_id) %>%
  mutate(ID = test$ID, label = test$label, text = test$text) %>%
  dplyr::select(ID, label, text, swc)

## mean word length
test_wl <- test_anno$token %>%
  mutate(num_id = as.numeric(str_remove(id, "doc"))) %>%
  mutate(wlen = nchar(word)) %>% 
  group_by(id) %>%
  summarise(wlen = mean(wlen), num_id = first(num_id)) %>%
  arrange(num_id) %>%
  mutate(ID = test$ID, label = test$label, text = test$text) %>%
  dplyr::select(ID, label, text, wlen)
  
## type-token ratio
test_totals <- test_anno$token %>%
  count(id)
test_ttr <- test_anno$token %>%
  mutate(num_id = as.numeric(str_remove(id, "doc"))) %>%
  left_join(test_totals) %>% 
  group_by(id) %>%
  summarise(types = n_distinct(word), tokens = first(n), num_id = first(num_id)) %>%
  arrange(num_id) %>%
  mutate(ID = test$ID, label = test$label, text = test$text, TTR = types / tokens) %>%
  dplyr::select(ID, label, text, types, tokens, TTR)

test_ttr %>%
  group_by(label) %>%
  summarise(TTR = mean(TTR))


## merge
test_complexity <- test %>%
  left_join(test_depths, by=c("ID", "label")) %>%
  left_join(test_swc, by=c("ID", "label", "text")) %>%
  left_join(test_wl, by=c("ID", "label", "text")) %>%
  left_join(test_ttr, by=c("ID", "label", "text")) %>%
  left_join(test_read, by=c("ID", "label"))

# write to file
write_tsv(test_complexity, "features/fnn_test_titles_complexity.tsv")


############## LOAD COMPLEXITY ############## 
# read from file
train_complexity <- loadFNNComplexity('train')
test_complexity <- loadFNNComplexity('test')
train_titles_complexity <- loadFNNComplexity('train_titles')
test_titles_complexity <- loadFNNComplexity('test_titles')

## evaluate var imp and aov
complexity_ranks <- getVarRanks(train_complexity)
titles_complexity_ranks <- getVarRanks(train_titles_complexity)

############## LOAD LIWC ############## 
## LIWC groups
LIWC_groups <- loadLIWCGroups()
## train
train_LIWC<-loadFNNLIWC('train')
train_titles_LIWC<-loadFNNLIWC('train_titles')
## test
test_LIWC<-loadFNNLIWC('test')
test_titles_LIWC<-loadFNNLIWC('test_titles')

## evaluate var imp and aov
LIWC_ranks <- train_LIWC %>%
  getVarRanks()
titles_LIWC_ranks <- train_titles_LIWC %>%
  getVarRanks()

## check by groups
LIWC_ranks %>%
  left_join(LIWC_groups, by = "var") %>%
  group_by(group) %>%
  summarise(varimp_rank = mean(varimp_rank), p_rank = mean(p_rank), avg_rank = mean(avg_rank)) %>%
  arrange(avg_rank)


############## POS ############### 
# POS tags
train_POS <- loadFNNPOS('train')
test_POS <- loadFNNPOS('test')
train_titles_POS <- loadFNNPOS('train_titles')
test_titles_POS <- loadFNNPOS('test_titles')

## evaluate var imp and aov
POS_ranks <- train_POS %>%
  getVarRanks()
titles_POS_ranks <- train_POS %>%
  getVarRanks()


############## NER ############### 
# NER tags
train_NER <- loadFNNNER('train')
test_NER <- loadFNNNER('test')
train_titles_NER <- loadFNNNER('train_titles')
test_titles_NER <- loadFNNNER('test_titles')

## evaluate var imp and aov
NER_ranks <- train_NER %>%
  getVarRanks()
titles_NER_ranks <- train_NER %>%
  getVarRanks()

############## MERGE ############### 
## merge datasets
train_txtfeat <- train_complexity %>%
  left_join(train_LIWC, by = c("ID", "label")) %>%
  left_join(train_POS, by = c("ID", "label")) %>%
  left_join(train_NER, by = c("ID", "label")) %>%
  mutate(label = as.factor(2 - unclass(label))) %>%
  distinct(ID, .keep_all= TRUE) %>%
  dplyr::select(-ID)
test_txtfeat <- test_complexity %>%
  left_join(test_LIWC, by = c("ID", "label")) %>%
  left_join(test_POS, by = c("ID", "label")) %>%
  left_join(test_NER, by = c("ID", "label")) %>%
  mutate(label = as.factor(2 - unclass(label))) %>%
  distinct(ID, .keep_all= TRUE) %>%
  dplyr::select(-ID)

## titles
train_titles_txtfeat <- train_titles_complexity %>%
  left_join(train_titles_LIWC, by = c("ID", "label")) %>%
  left_join(train_titles_POS, by = c("ID", "label")) %>%
  left_join(train_titles_NER, by = c("ID", "label")) %>%
  mutate(label = as.factor(2 - unclass(label))) %>%
  distinct(ID, .keep_all= TRUE) %>%
  dplyr::select(-ID)

test_titles_txtfeat <- test_titles_complexity %>%
  left_join(test_titles_LIWC, by = c("ID", "label")) %>%
  left_join(test_titles_POS, by = c("ID", "label")) %>%
  left_join(test_titles_NER, by = c("ID", "label")) %>%
  mutate(label = as.factor(2 - unclass(label))) %>%
  distinct(ID, .keep_all= TRUE) %>%
  dplyr::select(-ID)

## merge ranks
txtfeat_ranks <- complexity_ranks %>%
  full_join(LIWC_ranks) %>%
  full_join(POS_ranks) %>%
  full_join(NER_ranks) %>%
  dplyr::select(var, max_varimp, p_val) %>%
  mutate(varimp_rank = rank(-max_varimp), p_rank = rank(p_val), avg_rank = (varimp_rank + p_rank) / 2) %>%
  arrange(avg_rank)

txtfeat_titles_ranks <- titles_complexity_ranks %>%
  full_join(titles_LIWC_ranks) %>%
  full_join(titles_POS_ranks) %>%
  full_join(titles_NER_ranks) %>%
  dplyr::select(var, max_varimp, p_val) %>%
  mutate(varimp_rank = rank(-max_varimp), p_rank = rank(p_val), avg_rank = (varimp_rank + p_rank) / 2) %>%
  arrange(avg_rank)

## subset predictors
train_fit <- train_txtfeat %>%
  dplyr::select(-mu_sentence, -mu_noun_phrase, -sd_sentence, -sd_verb_phrase,-sd_noun_phrase, -num_verb_phrase,
                -tokens, -WC, -Analytic, -Clout, -WPS, -Sixltr, -Dic, -`function`, -pronoun, -ppron, -article,
                -prep, -verb, -adj,
                -compare, -posemo, -negemo, -anx, -anger, -sad, -family, -insight, -discrep, -tentat, -hear,
                -feel, -health, -sexual,
                -ingest, -power, -reward, -risk, -focuspast, -focusfuture, -relativ, -leisure, -money, -relig,
                -swear, -netspeak,
                -assent, -nonflu, -AllPunc, -Exclam, -Apostro, -OtherP, -CC, -CD, -DT, -EX, -IN, -JJ, -JJR,
                -JJS, -LS, -NN, -NNS, -NNPS,
                -PDT, -POS, -PRP, -`PRP$`, -RB, -RBR, -RP, -SYM, -TO, -UH, -VBD, -VBG, -VBN, -VBP, -VBZ, -WDT, -`WP$`, -WRB,
                -types, -NER, -VB,
                -filler) %>%
  filter(wlen < 7.5 & wlen > 2.5) %>%
  filter(TTR > .125) %>%
  filter(FK < 60) %>%
  mutate_if(is.numeric, funs(scale))

## titles
train_titles_fit <- train_txtfeat %>%
  dplyr::select(-mu_sentence, -mu_verb_phrase, -mu_noun_phrase, -sd_sentence, -sd_verb_phrase, -iqr_noun_phrase, -iqr_verb_phrase,
                -swc, wlen, types, tokens, FOG, FK, ARI, WC, Analytic, Clout, Authentic, WPS, Sixltr, Dic, `function`, pronoun,
                ppron, we, they, prep, adverb, conj, verb, adj, compare, affect, anger, sad, family, friend, cause, insight, 
                tentat, certain, hear, bio, sexual, ingest, drives, affiliation, power, risk, focuspast, motion, space, work, money,
                relig, informal, swear, assent, nonflu, filler, AllPunc, Period, Comma, Colon, SemiC, Quote, Parenth, CC, EX, FW,
                IN, JJ, JJR, LS, MD, NNS, NNP, PDT, PRP, RBR, RBS, RP, SYM, UH, VB, VBD, VBN, WP, `WP$`, WRB)

############## MODEL ############### 
mod <-  glm(label ~ .,
             data=train_fit, family="binomial")
summary(mod)
getROC(mod, train_fit)
calcAccuracyLR(mod, train_fit, cutoff = 0.25) 

## titles
mod.titles <-  glm(label ~ .,
            data=train_titles_fit, family="binomial")
summary(mod.titles)
getROC(mod.titles, train_titles_txtfeat)
calcAccuracyLR(mod.titles, train_titles_txtfeat, cutoff = 0.20) 

## interactions
# mod.int <- glm(label ~ . ^ 2,
#                data=train_fit, family="binomial")
# ints <- summary(mod)$coefficients %>%
#   data.frame() %>%
#   as_tibble(rownames = NA) %>%
#   rownames_to_column(var = "var") %>%
#   filter(`Pr...z..` < 0.01) %>%
#   arrange(desc(abs(Estimate)))

## get coefficients
coef <- coef(mod) %>%
  enframe() %>%
  rename(var = name, coef = value)
p_vals <- summary(mod)$coefficients[,4] %>%
  enframe() %>%
  rename(var = name, coef_pval = value)

coef_ranks <- coef %>%
  left_join(p_vals, by="var") %>%
  arrange(desc(abs(coef))) %>%
  mutate(coef_rank = rank(-abs(coef))) %>%
  dplyr::select(var, coef, coef_pval, coef_rank)

## compare with varImp and anova
total_ranks <- txtfeat_ranks %>%
  left_join(coef_ranks, by = "var") %>%
  filter(!is.na(coef)) %>%
  mutate(avg_rank = (varimp_rank + p_rank + coef_rank) / 3) %>%
  dplyr::select(var:p_val, coef, everything())



############## PREDICTOR SELECTION ############### 
# LASSO
# y <- as.matrix(train_titles_txtfeat$label)
# x <- as.matrix(train_titles_txtfeat[,-1])
# lasso <- cv.glmnet(x=x,y=y,alpha = 1, family="binomial")
# coef(lasso)
# 
# 
# # VIF
# vif(mod) %>%
#   enframe %>%
#   arrange(desc(value))
# 
# # PCA
# pr.out <- train_txtfeat %>%
#   dplyr::select(-label) %>%
#   prcomp(scale=TRUE)
# 
# pr.var <- pr.out$sdev^2
# pve <- pr.var/sum(pr.var)
# plot(pve)
# 


############## ANALYZE PREDICTORS ############### 
## get rid of ranks
pred_eval <- total_ranks %>%
  dplyr::select(var, max_varimp, p_val, coef, coef_pval) %>%
  rename(aov_pval = p_val)
    
## correlation matrix
pred_eval %>%
  filter(var != "TTR") %>%
  dplyr::select(-var) %>%
  mutate(coef = abs(coef)) %>%
  cor()

## get p_val groups
pred_eval <- pred_eval %>%
  mutate(c_pval = ifelse(coef_pval > .1, "",
                                  ifelse(coef_pval > 0.05, ".",
                                    ifelse(coef_pval > 0.01, "*",
                                      ifelse(coef_pval > 0.001, "**", "***")))))
# PLOT
pred_eval %>%
    mutate(coef_sign = ifelse(coef > 0, "+", "-")) %>%
    ggplot(aes(x = max_varimp, y = coef)) +
    geom_point(aes(shape = c_pval, color = coef_sign), size = 3) +
    geom_text_repel(mapping=aes(label=var, color=coef_sign),size=4, box.padding = unit(0.5, "lines"))
  

## invidual predictors
### FK 
train_fit %>%
  dplyr::select(label, FK) %>%
  ggplot(aes(x = label, color = label)) +
  geom_boxplot(aes(y=FK))
train_fit %>%
  dplyr::select(label, FK) %>%
  ggplot(aes(x = FK, color = label)) +
  geom_histogram(aes(y = stat(width*density)),fill="white") +
  facet_wrap(~label)
train_txtfeat %>%
  dplyr::select(label, FK) %>%
  ggplot(aes(x = seq(1,nrow(.)), y = FK, color = label)) +
  geom_point()

### filler
## REMOVE BECAUSE ONLY 440 NON-ZERO
train_fit %>%
  dplyr::select(label, wlen, wlen) %>%
  ggplot(aes(x = wlen, y=label, color = label)) +
  geom_point()


library("effects")
# plot(effect("auxverb",mod))

