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
library("reticulate")
use_python("C:/Users/Caio Brighenti/AppData/Local/Programs/Python/Python37", required = T)
py_config()
source_python("processCoreNLP.py")
## create empty dataframe
test_depths <- tibble(ID = test$ID, 
                       sentence = rep(0, nrow(test)),
                       verb_phrase = rep(0, nrow(test)),
                       noun_phrase = rep(0, nrow(test)))
## calculate tree depths for each document
pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = nrow(test))
for (idx in 1:nrow(test)) {
  pb$tick()
  if (test_depths[idx,]$sentence == 0) {
    t_depths <- getConstTreeDepths(test[idx,]$text)
    test_depths[idx,]$sentence <- t_depths$sentence
    test_depths[idx,]$verb_phrase <- t_depths$`verb-phrase`
    test_depths[idx,]$noun_phrase <- t_depths$`noun-phrase`
  }
}
  

## check distribution
test_labels <- test %>%
  dplyr::select(ID, label)
test_depths <- test_depths %>%
  left_join(test_labels, by="ID")
test_depths %>% 
  filter(sentence != 0) %>%
  group_by(label) %>%
  summarise(sentence = median(sentence), verb_phrase = median(verb_phrase), noun_phrase = median(noun_phrase))
plot(test_depths$verb_phrase, test_depths$noun_phrase)

# write to file
write_tsv(test_depths, "coreNLP_annotations/fnn_test.tsv")

############## SETUP CORENLP ############## 
# cnlp_download_corenlp()
cnlp_init_corenlp()
t1 <- Sys.time()
anno <- cnlp_annotate(train$text)
t2 <- Sys.time()
print(t2 - t1)



############## COMPLEXITY ############## 


############## PSYCHOLOGY ############## 


############## STYLISTIC ############## 