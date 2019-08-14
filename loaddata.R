library(readr)

##################################
############## LIAR ##############
##################################
loadLIARTrain <- function() {
  ## load train data
  train<-read.csv(file="LIAR/dataset/train.TSV",sep = '\t', quote="", header = FALSE)
  header<-c("ID","label","text","subject","speaker","speaker.title","state",
            "party","bt.count","f.count","ht.count","mt.count","pof.count","context")
  names(train)<-header

    ## reorder and number label
  labels<-c("pants-fire","false","barely-true","half-true","mostly-true","true")
  for (num in 6:1) {
    train$label <- relevel(train$label,labels[num])
  }
  return(train)
}

loadLIARTest <- function() {
  ## load train data
  test<-read.csv(file="LIAR/dataset/test.TSV",sep = '\t', quote="", header = FALSE)
  header<-c("ID","label","text","subject","speaker","speaker.title","state",
            "party","bt.count","f.count","ht.count","mt.count","pof.count","context")
  names(test)<-header
  
  ## reorder and number label
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
  ffn<-read.csv(file="FakeNewsNet/dataset/fnn_data.TSV",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
    as_tibble()
  train_ind <- sample(seq_len(nrow(ffn)), size = floor(nrow(ffn) * .75))
  train <- ffn[train_ind,]
  test <- ffn[-train_ind,]
  write_tsv(data.frame(train), "FakeNewsNet/dataset/fnn_train.tsv")
  write_tsv(data.frame(test), "FakeNewsNet/dataset/fnn_test.tsv")
}

loadFNNTrain <- function() {
  train<-read.csv(file="FakeNewsNet/dataset/fnn_train.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
    as_tibble()
  return(train)
}
loadFNNTest <- function() {
  test<-read.csv(file="FakeNewsNet/dataset/fnn_test.tsv",sep = '\t', quote="", header = TRUE, encoding="UTF-8") %>%
    as_tibble()
  return(test)
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


