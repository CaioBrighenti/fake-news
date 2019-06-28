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

############## CREATE TRAIN DTM ############## 
# Create iterator over tokens
tokens <- train$statement %>%
  lemmatize_strings %>%
  tolower %>%
  word_tokenizer

# Create vocabulary. Terms will be unigrams (simple words).
## remove stopwords
it = itoken(iterable = tokens,
            progressbar = FALSE)
vocab <- create_vocabulary(it,stopwords = stopwords('en'),ngram = c(1L, 2L)) 
## remove infrequent words and stem words
vocab <- prune_vocabulary(vocab, term_count_min = 5L)
#vocab$term <- lemmatize_strings(vocab$term)
# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)
# create DTM
dtm_train = create_dtm(it, vectorizer)

############## CREATE TEST DTM ############## 
# Create iterator over tokens
test_tokens <- test$statement %>%
  lemmatize_strings %>%
  tolower %>%
  word_tokenizer

# Create vocabulary. Terms will be unigrams (simple words).
## remove stopwords
test_it = itoken(iterable = test_tokens,
            progressbar = FALSE)
# create DTM
dtm_test = create_dtm(test_it, vectorizer)

############## TFIDF TRANSFORM ##############


############## FIT MODELS ############## 
## fit ordinal logistic model
library(MASS)
dat_train<-data.frame(label=train$label,as.matrix(dtm_train))
mod.lm<-lm(label~.,data=dat_train,Hess=TRUE)
#summary(mod.polr)
calcAccuracy(mod.polr, dat_train)
calcAccuracy(mod.polr, dat_test)
