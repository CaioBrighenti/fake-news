############## HYPERPARAMETERS ############## 
wvec_size <- 300


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
library(caret)
# load in data
source("loaddata.R")
source("helpers.R")
LIAR_train <- loadLIARTrain()
LIAR_test <- loadLIARTest()
FNN_train <- loadFNNTrain()
FNN_test <- loadFNNTest()

## chose dataset
train <- FNN_train
test <- FNN_test
#train <- LIAR_train

############## FIX IMBALANCED DATA ############## 
barplot(table(train$label))
train <- caret::upSample(train, train$label)
barplot(table(train$label))

############## CREATE WORD EMBEDDINGS ############## 
# Create iterator over tokens
tokens <- train$text %>%
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
word_vectors<-glove$fit_transform(tcm, n_iter = 20)

############## GET TFIDF DTM ############## 
# dtm_train = create_dtm(it, vectorizer)
# tfidf = TfIdf$new()
# dtm_tfidf = tfidf$fit_transform(dtm_train)

############## CREATE DOC VECTORS############## 
## create doc vectors for train data
train_dvec <- docVector(tokens, word_vectors)
mode(train_dvec) = "numeric"
#  0 = real, 1 = fake
dat_train<-data.frame(ID=train$ID, label=as.factor(2-unclass(train$label)),train_dvec)[,-1]
## create doc vectors for test data
test_tokens <- test$text %>%
  lemmatize_strings %>%
  tolower %>%
  word_tokenizer
test_dvec <- docVector(test_tokens, word_vectors)
mode(test_dvec) = "numeric"
dat_test<-data.frame(ID=test$ID, label=as.factor(2-unclass(test$label)),test_dvec)[,-1]


############## GOOGLE NEWS MODEL ############## 
## load in Google News vectors
#install_github("bmschmidt/wordVectors")
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
train$text <- as.character(train$text)
tidy_train <- as_tibble(train) %>% 
  dplyr::select(-title) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words)
## test data
test$text <- as.character(test$text)
tidy_test <- as_tibble(test) %>% 
  dplyr::select(-title) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words)

# get truth counts
train_rating <- tidy_train %>%
  inner_join(truth_dict) %>%
  group_by(ID) %>%
  summarise(label= first(label), truth = sum(truth), untruth=sum(untruth), net=sum(net))
# get test counts
test_rating <- tidy_test %>%
  inner_join(truth_dict) %>%
  group_by(ID) %>%
  summarise(label= first(label), truth = sum(truth), untruth=sum(untruth), net=sum(net))

## merge with document vectors
dat_train <- as_tibble(dat_train)
train_full <- dat_train %>%
  full_join(train_rating, by="ID") %>%
  mutate(truth = replace_na(truth, 0), untruth = replace_na(untruth, 0), net = replace_na(net, 0)) %>%
  mutate(label=as.factor(label.x)) %>%
  dplyr::select(-label.y, -label.x) %>%
  dplyr::select(label, truth, untruth, net, everything(), -ID,)
## test
dat_test <- as_tibble(dat_test)
test_full <- dat_test %>%
  full_join(test_rating, by="ID") %>%
  mutate(truth = replace_na(truth, 0), untruth = replace_na(untruth, 0), net = replace_na(net, 0)) %>%
  mutate(label=as.factor(label.x)) %>%
  dplyr::select(-label.y, -label.x) %>%
  dplyr::select(label, truth, untruth, net, everything(), -ID,)

############## FIT MODELS ############## 
## fit ordinal logistic model
mod.logit<-glm(label~.,data=dat_train,family="binomial")
summary(mod.logit)
stats.logit.train <- calcAccuracyLR(mod.logit, dat_train)
stats.logit.test <- calcAccuracyLR(mod.logit, dat_test)

## SVM
mod.svm<-svm(label~.,data=dat_train, kernel="linear", scale=FALSE)
# tune.out <- tune(svm, as.factor(label)~truth+untruth, data = train_rating, kernel="radial",
#                  ranges = list(gamma = seq(1:5), cost = 2^(2:4)),
#                  tunecontrol = tune.control(sampling = "fix")
# )
#mod.svm <- tune.out$best.model
stats.svm_train<-calcAccuracy(mod.svm, dat_train)
stats.svm_test<-calcAccuracy(mod.svm, dat_test)

#plotPredictions(list(mod.polr, mod.svm), test_full)

## accuracy plot
accs <- tibble(model=c("Ordered Logistic Regression","Ordered Logistic Regression",
                       "Ordered Logistic Regression","Ordered Logistic Regression",
                       "SVM","SVM","SVM","SVM"),
                    data=c("train","test","train","test","train","test","train","test"),
               type=c("acc","acc","adj","adj","acc","acc","adj","adj"),
               value=c(acc.polr.train,acc.polr.test,adj.polr.train,adj.polr.test,
                       acc.svm_train,acc.svm_test,adj.svm_train,adj.svm_test))
accs %>%
  ggplot(aes(x=model,y=value, fill=type)) +
  geom_bar(stat = "identity",position=position_dodge()) +
  facet_wrap(~data) +
  scale_fill_manual(values= colgate_ter, labels = c("Accuracy", "Adjacency")) +
  ylab("Value") +
  xlab("") +
  ggtitle("Preliminary modeling accuracy and adjacency for train/test data") +
  geom_text(aes(label=round(value, digits = 2)), vjust=1.6, color="white",
            position = position_dodge(0.9), size=4) +
  ylim(0,1)
  

############## HELPER FUNCTIONS ############## 
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

