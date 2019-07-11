############## HYPERPARAMETERS ############## 
wvec_size <- 100


############## LOAD ############## 
## load in libs
library("text2vec")
library(devtools)
library("tidytext")
library("stopwords")
library("dplyr")
library("textstem")
library(progress)
library(MASS)
library(e1071)
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
glove = GlobalVectors$new(word_vectors_size = wvec_size, vocabulary = vocab, x_max = 10, learning_rate=0.01)
word_vectors<-glove$fit_transform(tcm, n_iter = 35)

############## GET TFIDF DTM ############## 
# dtm_train = create_dtm(it, vectorizer)
# tfidf = TfIdf$new()
# dtm_tfidf = tfidf$fit_transform(dtm_train)

############## CREATE DOC VECTORS############## 
## create doc vectors for train data
train_dvec <- docVector(tokens, word_vectors)
mode(train_dvec) = "numeric"
dat_train<-data.frame(label=as.factor(unclass(train$label)),train_dvec)

## create doc vectors for test data
test_tokens <- test$statement %>%
  lemmatize_strings %>%
  tolower %>%
  word_tokenizer
test_dvec <- docVector(test_tokens, word_vectors)
mode(test_dvec) = "numeric"
dat_test<-data.frame(label=as.factor(unclass(test$label)),test_dvec)


############## GOOGLE NEWS MODEL ############## 
## load in Google News vectors
install_github("bmschmidt/wordVectors")
library(wordVectors)
path = "/google_vecs/gnews.bin"
f <- file.choose()
g_news <- read.vectors(f)

## create document vectors
train_dvec <- docVector(tokens, g_news)
mode(train_dvec) = "numeric"
dat_train<-data.frame(label=as.factor(unclass(train$label)),train_dvec)
test_dvec <- docVector(test_tokens, g_news)
mode(test_dvec) = "numeric"
dat_test<-data.frame(label=as.factor(unclass(test$label)),test_dvec)


############## TRUTH DICTIONARY FEATURES ############## 
source("survey.R")
truth_dict <- loadTruthDict() %>%
  anti_join(stop_words)

# tidy train data
train_df <- as_tibble(train[,1:3,])
train_df$statement <- as.character(train_df$statement)
tidy_train <- train_df %>% 
  unnest_tokens(word, statement) %>%
  anti_join(stop_words)
## test data
test_df <- as_tibble(test[,1:3,])
test_df$statement <- as.character(test_df$statement)
tidy_test <- test_df %>% 
  unnest_tokens(word, statement) %>%
  anti_join(stop_words)

# get truth counts
train_rating <- tidy_train %>%
  inner_join(truth_dict) %>%
  group_by(ID) %>%
  summarise(label= unique(label), truth = sum(truth), untruth=sum(untruth), net=sum(net))
# get test counts
test_rating <- tidy_test %>%
  inner_join(truth_dict) %>%
  group_by(ID) %>%
  summarise(label= unique(label), truth = sum(truth), untruth=sum(untruth), net=sum(net))

## model fits
mod.polr<-polr(as.factor(label)~truth+untruth,data=train_rating,Hess=TRUE)
calcAccuracy(mod.polr, train_rating)
calcAccuracy(mod.polr, test_rating)

#tune.out <- tune(svm, as.factor(label)~truth+untruth, data = train_rating, kernel="polynomial",
#            ranges = list(power = seq(1:5), cost = 2^(2:4)),
#            tunecontrol = tune.control(sampling = "fix")
#)
#plot(tune.out)
mod.svm<-svm(as.factor(label)~truth+untruth,data=train_rating, kernel="radial", gamma=3, cost=2, scale=FALSE)
calcAccuracy(mod.svm, train_rating)
calcAccuracy(mod.svm, test_rating)

plotPredictions(list(mod.polr,mod.svm), test_rating)

############## FIT MODELS ############## 
## fit ordinal logistic model
mod.polr<-polr(as.factor(label)~.,data=dat_train,Hess=TRUE)
#summary(mod.polr)
calcAccuracy(mod.polr, dat_train)
calcAccuracy(mod.polr, dat_test)

## SVM
mod.svm<-svm(as.factor(label)~.,data=dat_train, kernel="linear", cost=1, scale=FALSE)
#tune.out<-tune(svm,as.factor(label)~.,data=dat_train,kernel="linear", ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
calcAccuracy(mod.svm, dat_train)
calcAccuracy(mod.svm, dat_test)

############## HELPER FUNCTIONS ############## 
plotPredictions <- function(mods,dat_test){
  par(mfrow=c(1,length(mods)+1))
  plot(test$label)
  for (mod in mods) {
    plot(predict(mod, newdata = dat_test))
  }
}

docVector <- function(tokens, word_vectors){
  ## create doc vectors for train data
  doc_vectors<-matrix(0, nrow=length(tokens), ncol=dim(word_vectors)[2])
  ## loop through each document
  pb <- progress_bar$new(total = length(tokens))
  for (i in 1:length(tokens)) {
    pb$tick()
    # grab words in doc
    words<-unlist(tokens[i])
    # remove pruned words and stopwords
    words<-words[which(words %in% rownames(word_vectors))]
    # calculate document mean
    if (length(words)>1){
      doc_vec <- colMeans(word_vectors[words, ])
    } else if (length(words) == 1){
      doc_vec <- word_vectors[words, ]
    } else {
      doc_vec <- rep(0,ncol=dim(word_vectors)[2])
    }
    # add document to matrix
    doc_vectors[i,]<-doc_vec
  }
  
  return(doc_vectors)
}

docVectorWeighted <- function(tokens,vocab,word_vectors,weights){
  ## create doc vectors for train data
  doc_vectors<-matrix(0, nrow=length(tokens), ncol=wvec_size)
  ## loop through each document
  for (i in 1:length(tokens)) {
    # grab words in doc
    words<-unlist(tokens[i])
    # remove pruned words and stopwords
    words<-words[which(words %in% vocab$term)]
    if (length(words)==0) {
      doc_vectors[i,]<-rep(0,wvec_size)
      next
    }
    # get word vectors to calc doc vec
    word_vecs<-word_vectors[words, ]
    # tfidf weighing
    doc_weights <- weights[i,][which(weights[i,]!=0)]
    ## should vectorize but for now loop
    if (length(words)>1){
      for (i in 1:length(words)) {
        word<-words[i]
        word_vecs[i,] <- word_vecs[i,] * doc_weights[word]
      }
    } else {
      word_vecs <- word_vecs * doc_weights
    }
    # get document vector
    if (length(words)>1){
      doc_vec <- colMeans(word_vecs)
    } else if (length(words) == 1){
      doc_vec <- word_vecs
    }
    # add document to matrix
    doc_vectors[i,]<-doc_vec
  }
  return(doc_vectors)
}

calcAccuracy <- function(mod,new_data,adj=0) {
  true_labels <- new_data$label
  pred <- predict(mod, newdata = new_data, type="class")
  dist <- abs(as.numeric(pred)-as.numeric(true_labels))
  acc <-  mean(dist <= adj)
  return(acc)
}

