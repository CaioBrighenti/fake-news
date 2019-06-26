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

## word similarity
x <- word_vectors["Trump", , drop = FALSE]
cos_sim = sim2(x = word_vectors, y = x, method = "cosine", norm = "l2")
head(sort(cos_sim[,1], decreasing = TRUE), 5)


## create doc vectors
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
    doc_vec <- words
  } else {
    doc_vec <- rep(0,50)
  }
  # add document to matrix
  doc_vectors[i,]<-doc_vec
}

dat<-data.frame(label=as.factor(unclass(train$label)),doc_vectors)
head(dat)

## make sure predictors are numberic
### shouldn't be necessary but I GUESS THAT'S OK
for (idx in 2:51) {
  dat[,idx]<-as.numeric(type.convert(dat[,idx]))
}


## fit ordinal logistic model
library(MASS)
#mod<-lm(as.numeric(label)~.,data=dat)
#summary(mod)
mod.polr<-polr(label~.,data=dat,Hess=TRUE)
summary(mod.polr)

calcAccuracy(mod.polr, dat, dat$label)

calcAccuracy <- function(mod,new_data,true_labels) {
  pred <- predict(mod, newdata = new_data, type="class")
  acc <- mean(pred == true_labels)
  return(acc)
}

