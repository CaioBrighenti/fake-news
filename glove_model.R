# load in data
source("loaddata.R")
train <- loadLIARTrain()
test <- loadLIARTest()


## create word embeddings
library("text2vec")
# Create iterator over tokens
tokens <- space_tokenizer(train$statement)
# Create vocabulary. Terms will be unigrams (simple words).
it = itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it) 
vocab <- prune_vocabulary(vocab, term_count_min = 5L)
# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)
# use window of 5 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)
glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
word_vectors<-glove$fit_transform(tcm, n_iter = 20)

## create doc vectors for train data
train_dvec <- docVector(tokens, vocab, word_vectors)
mode(train_dvec) = "numeric"
dat_train<-data.frame(label=as.factor(unclass(train$label)),train_dvec)
head(dat_train)

## fit ordinal logistic model
#library(MASS)
mod.polr<-polr(as.factor(label)~.,data=dat_train,Hess=TRUE)
summary(mod.polr)

calcAccuracy(mod.polr, dat_train, dat_train$label)




## create doc vectors for test data
test_tokens <- space_tokenizer(test$statement)  
## create doc vectors for train data
test_dvec <- docVector(test_tokens, vocab, word_vectors)
mode(test_dvec) = "numeric"
dat_test<-data.frame(label=as.factor(unclass(test$label)),test_dvec)
head(dat_test)

## test accuracy
calcAccuracy(mod.polr, dat_test, dat_test$label)



docVector <- function(tokens,vocab,word_vectors){
  ## create doc vectors for train data
  doc_vectors<-matrix(0, nrow=length(tokens), ncol=50)
  ## loop through each document
  for (i in 1:length(tokens)) {
    # grab words in doc
    words<-unlist(tokens[i])
    # remove pruned words
    words<-words[which(words %in% vocab$term)]
    # calculate document mean
    if (length(words)>1){
      doc_vec <- colMeans(word_vectors[words, ])
    } else if (length(words) == 1){
      doc_vec <- word_vectors[words, ]
    } else {
      doc_vec <- rep(0,50)
    }
    # add document to matrix
    doc_vectors[i,]<-doc_vec
  }

  return(doc_vectors)
}

calcAccuracy <- function(mod,new_data,true_labels) {
  pred <- predict(mod, newdata = new_data, type="class")
  acc <- mean(pred == true_labels)
  return(acc)
}

