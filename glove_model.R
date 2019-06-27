############## LOAD ############## 
## load in libs
library("text2vec")
library("tidytext")
library("stopwords")
library("dplyr")
library("textstem")
# load in data
source("loaddata.R")
train <- loadLIARTrain()
test <- loadLIARTest()

############## CREATE WORD EMBEDDINGS ############## 
# Create iterator over tokens
tokens <- train$statement %>%
            lemmatize_strings %>%
            tolower %>%
            word_tokenizer

# Create vocabulary. Terms will be unigrams (simple words).
## remove stopwords
it = itoken(iterable = tokens,
            progressbar = FALSE)
vocab <- create_vocabulary(it,stopwords = stopwords('en')) 
## remove infrequent words and stem words
vocab <- prune_vocabulary(vocab, term_count_min = 5L)
#vocab$term <- lemmatize_strings(vocab$term)
# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)
# use window of 5 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)
glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
word_vectors<-glove$fit_transform(tcm, n_iter = 20)

############## CREATE DOC VECTORS############## 
## create doc vectors for train data
train_dvec <- docVector(tokens, vocab, word_vectors)
mode(train_dvec) = "numeric"
dat_train<-data.frame(label=as.factor(unclass(train$label)),train_dvec)
head(dat_train)

## create doc vectors for test data
test_tokens <- space_tokenizer(test$statement)  
test_dvec <- docVector(test_tokens, vocab, word_vectors)
mode(test_dvec) = "numeric"
dat_test<-data.frame(label=as.factor(unclass(test$label)),test_dvec)
head(dat_test)


############## FIT MODELS ############## 
## fit ordinal logistic model
library(MASS)
mod.polr<-polr(as.factor(label)~.,data=dat_train,Hess=TRUE)
summary(mod.polr)

calcAccuracy(mod.polr, dat_train, dat_train$label)
## test accuracy
calcAccuracy(mod.polr, dat_test, dat_test$label)



docVector <- function(tokens,vocab,word_vectors){
  ## create doc vectors for train data
  doc_vectors<-matrix(0, nrow=length(tokens), ncol=50)
  ## loop through each document
  for (i in 1:length(tokens)) {
    # grab words in doc
    words<-unlist(tokens[i])
    # remove pruned words and stopwords
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

