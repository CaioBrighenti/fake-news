############## LOAD ############## 
## load in libs
library("text2vec")
library("tidytext")
library("stopwords")
library("dplyr")
library("textstem")
# load in data
source("helpers.R")
source("loaddata.R")
LIAR_train <- loadLIARTrain()
LIAR_test <- loadLIARTest()
FNN_train <- loadFNNTrain()
FNN_test <- loadFNNTest()

## chose dataset
train <- FNN_train
test <- FNN_test
train$label <- as.factor(2-unclass(train$label))
test$label <- as.factor(2-unclass(test$label))

############## FIX IMBALANCED DATA ############## 
barplot(table(train$label))
train <- caret::upSample(train, train$label)
barplot(table(train$label))

############## CREATE TRAIN DTM ############## 
# Create iterator over tokens
tokens <- train$text %>%
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
test_tokens <- test$text %>%
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
## fit logistic model
library(glmnet)
NFOLDS = 4
cv.out = cv.glmnet(x = dtm_train, y = train$label, 
                              family = 'binomial', 
                              # L1 penalty
                              alpha = 1,
                              # interested in the area under ROC curve
                              type.measure = "auc",
                              # 5-fold cross-validation
                              nfolds = NFOLDS,
                              # high value is less accurate, but has faster training
                              thresh = 1e-3,
                              # again lower number of iterations for faster training
                              maxit = 1e3)
plot(cv.out)

## get accuracy
stats.logit_train<-calcAccuracyLR(cv.out, dtm_train, 0, train$label)
stats.logit_test<-calcAccuracyLR(cv.out, dtm_test, 0, test$label)

# get coefficients
word_coef <- coef(cv.out)
head(word_coef[order(-word_coef)], 20)
head(word_coef[order(word_coef)], 20)
