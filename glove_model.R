train<-read.csv(file="LIAR/dataset/train.TSV",sep = '\t', quote="", header = FALSE)
header<-c("ID","label","statement","subject","speaker","speaker.title","state","party","bt.count","f.count","ht.count","mt.count","pof.count","context")
names(train)<-header

## reorder and number label
labels<-c("pants-fire","false","barely-true","half-true","mostly-true","true")
for (num in 6:1) {
  train$label <- relevel(train$label,labels[num])
}
levels(train$label)

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
x <- word_vectors["dog", , drop = FALSE]
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
  doc_vecs[i,]<-doc_vec
}

dat<-data.frame(label=as.factor(unclass(train$label)),doc_vecs)
head(dat)

## fit ordinal logistic model
#library(MASS)
dat_sub<-sample(dat,size=10)
mod.polr<-polr(dat$label~dat$X1)
summary(mod.polr)