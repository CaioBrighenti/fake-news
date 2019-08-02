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
getSyntaxTreeDepths <- function(dat, dat_name){
  library("reticulate")
  use_python("C:/Users/Caio Brighenti/AppData/Local/Programs/Python/Python37", required = T)
  py_config()
  source_python("processCoreNLP.py")
  ## create empty dataframe
  dat_depths <- tibble(ID = dat$ID,
                         sentence = rep(0, nrow(dat)),
                         verb_phrase = rep(0, nrow(dat)),
                         noun_phrase = rep(0, nrow(dat)))
  ## calculate tree depths for each document
  pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = nrow(dat))
  for (idx in 1:nrow(dat)) {
    pb$tick()
    if (dat_depths[idx,]$sentence == 0) {
      t_depths <- getConstTreeDepths(dat[idx,]$text)
      dat_depths[idx,]$sentence <- t_depths$sentence
      dat_depths[idx,]$verb_phrase <- t_depths$`verb-phrase`
      dat_depths[idx,]$noun_phrase <- t_depths$`noun-phrase`
    }
  }
  
  
  ## check distribution
  dat_labels <- dat %>%
    dplyr::select(ID, label)
  dat_depths <- dat_depths %>%
    left_join(dat_labels, by="ID")
  dat_depths %>%
    filter(sentence != 0) %>%
    group_by(label) %>%
    summarise(sentence = median(sentence), verb_phrase = median(verb_phrase), noun_phrase = median(noun_phrase))
  plot(dat_depths$verb_phrase, dat_depths$noun_phrase)
  
  # write to file
  write_tsv(dat_depths, paste("coreNLP_annotations/",dat_name,".tsv", sep=""))
}

############## SETUP CORENLP ############## 
# cnlp_download_corenlp()
cnlp_init_corenlp()
t1 <- Sys.time()
anno <- cnlp_annotate(train$text)
t2 <- Sys.time()
print(t2 - t1)



############## COMPLEXITY ############## 
## syntax tree depths
train_depths <- read.csv(file="coreNLP_annotations/fnn_train.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
  as_tibble()
test_depths <- read.csv(file="coreNLP_annotations/fnn_test.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
  as_tibble()

## readability

############## PSYCHOLOGY ############## 


############## STYLISTIC ############## 