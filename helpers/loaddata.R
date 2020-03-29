library(readr)
library(dplyr)
library(data.table)

# load in FNN outliers
fnn_outliers <- as.character(read.csv("thesis/outliers.csv")$x)

##################################
############## LIAR ##############
##################################
loadLIARTrain <- function() {
  ## load train data
  train<-fread(file="../data/LIAR/dataset/train.TSV",sep = '\t', quote="", header = FALSE) %>%
    as_tibble()
  header<-c("ID","label","text","subject","speaker","speaker.title","state",
            "party","bt.count","f.count","ht.count","mt.count","pof.count","context")
  names(train)<-header

    ## reorder and number label
  train$label <- as.factor(train$label)
  labels<-c("pants-fire","false","barely-true","half-true","mostly-true","true")
  for (num in 6:1) {
    train$label <- relevel(train$label,labels[num])
  }
  return(train)
}

loadLIARTest <- function() {
  ## load train data
  test<-fread(file="data/LIAR/dataset/test.TSV",sep = '\t', quote="", header = FALSE) %>%
    as_tibble()
  header<-c("ID","label","text","subject","speaker","speaker.title","state",
            "party","bt.count","f.count","ht.count","mt.count","pof.count","context")
  names(test)<-header
  
  ## reorder and number label
  test$label <- as.factor(test$label)
  labels<-c("pants-fire","false","barely-true","half-true","mostly-true","true")
  for (num in 6:1) {
    test$label <- relevel(test$label,labels[num])
  }
  return(test)
}


##################################
########## FAKENEWSNET ###########
##################################
#splitDataFNN()
## load FakeNewsNet dataset
splitDataFNN <- function() {
  ffn<-fread(file="data/FakeNewsNet/dataset/fnn_data.TSV",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
    as_tibble()
  train_ind <- sample(seq_len(nrow(ffn)), size = floor(nrow(ffn) * .75))
  train <- ffn[train_ind,]
  test <- ffn[-train_ind,]
  write_tsv(data.frame(train), "FakeNewsNet/dataset/fnn_train.tsv")
  write_tsv(data.frame(test), "FakeNewsNet/dataset/fnn_test.tsv")
}

loadFNNTrain <- function() {
  train<-fread(file="data/FakeNewsNet/dataset/fnn_train.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
    as_tibble() %>%
    mutate(label = as.factor(label)) %>%
    filter(!(ID %in% fnn_outliers))
  return(train)
}
loadFNNTest <- function() {
  test<-fread(file="data/FakeNewsNet/dataset/fnn_test.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
    as_tibble() %>%
    mutate(label = as.factor(label))
  return(test)
}

loadFNNComplexity <- function(str){
  path = paste("features/fnn_",str,"_complexity.tsv", sep = '')
  dat_complexity <- fread(file=path,sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
    as_tibble() %>%
    mutate(ID = as.character(ID), label = as.factor(label)) %>%
    dplyr::select(-text) %>%
    filter(!(ID %in% fnn_outliers))
  return(dat_complexity)
}

loadLIWCGroups <- function(){
  LIWC_groups <- tibble(
    WC = "summary",
    Analytic = "summary",	
    Clout = "summary",	
    Authentic = "summary",	
    Tone = "summary",
    WPS = "summary",	
    Sixltr = "summary",	
    Dic = "summary",	
    `function` = "function",
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
  return(LIWC_groups)
}

loadFNNLIWC <- function(str){
  path = paste("annotations/LIWC/LIWC2015_fnn_",str,".csv", sep = '')
  dat_LIWC<-fread(file=path,header = TRUE, encoding="UTF-8") %>%
    as_tibble() %>%
    mutate(ID = as.character(A), label = as.factor(B), title = as.character(C), text = as.character(D)) %>%
    dplyr::select(ID, label, -title, -text, WC:OtherP) %>%
    filter(!(ID %in% fnn_outliers))
  return(dat_LIWC)
}

loadFNNPOS <- function(str){
  path = paste("annotations/coreNLP/fnn_",str,"_POS.tsv", sep = '')
  dat_POS <- fread(file=path,sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
    as_tibble() %>%
    mutate(ID = as.character(ID), label = as.factor(label)) %>%
    dplyr::select(ID, label, everything()) %>%
    filter(!(ID %in% fnn_outliers))
  return(dat_POS)
}

loadFNNNER <- function(str){
  path = paste("annotations/coreNLP/fnn_",str,"_NER.tsv", sep = '')
  dat_NER <- fread(file=path,sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
    as_tibble() %>%
    mutate(ID = as.character(ID), label=as.factor(label)) %>%
    dplyr::select(ID, label, everything()) %>%
    filter(!(ID %in% fnn_outliers))
  return(dat_NER)
}

loadFNNtxtfeat <- function(str){
  # load complexity
  temp_complexity <- loadFNNComplexity(str)
  
  # load LWIC
  temp_LIWC<-loadFNNLIWC(str)
  
  # load POS
  temp_POS <- loadFNNPOS(str)
  
  # load NER
  temp_NER <- loadFNNNER(str)
  
  # merge
  temp_txtfeat <- temp_complexity %>%
  left_join(temp_LIWC, by = c("ID", "label")) %>%
    left_join(temp_POS, by = c("ID", "label")) %>%
    left_join(temp_NER, by = c("ID", "label")) %>%
    mutate(label = as.factor(2 - unclass(label))) %>%
    distinct(ID, .keep_all= TRUE)
  
  return(temp_txtfeat)
}

## write LIAR to fastText format
# file.create("LIAR/train.txt")
# #out<-file("liar_dataset/train.txt")
# for (i in 1:nrow(train)) {
#   line<-paste("__label__", unclass(train$label[i]), " ",train$statement[i], sep="")
#   write(line,file="LIAR/train.txt",append=TRUE)
# }
#close(out)


## write LIAR to output
# file.create("LIAR/test.txt")
# #out<-file("liar_dataset/test.txt")
# for (i in 1:nrow(test)) {
#   line<-paste("__label__", unclass(test$label[i]), " ",test$statement[i], sep="")
#   write(line,file="LIAR/test.txt",append=TRUE)
# }
#close(out)


## sentiment analysis
#library("sentimentr")
#test <- get_sentences(as.character(train$statement))
# library("syuzhet")
# sentiments<-get_nrc_sentiment(as.character(train$statement))
# train<-cbind(train,sentiments)
# #logistic
# mod.logit<-glm(label~anger+anticipation+disgust+fear+joy+sadness+surprise+
#            trust+negative+positive,data=train,family = binomial(link = "logit"))
# 
# summary(mod.logit)
# library(pscl)
# pR2(mod.logit)
# #linear
# mod.lm<-lm(unclass(label)~anger+anticipation+disgust+fear+joy+sadness+surprise+
#              trust+negative+positive,data=train)
# summary(mod.lm)


