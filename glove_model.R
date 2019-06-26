train<-read.csv(file="LIAR/dataset/train.TSV",sep = '\t', quote="", header = FALSE)
header<-c("ID","label","statement","subject","speaker","speaker.title","state","party","bt.count","f.count","ht.count","mt.count","pof.count","context")
names(train)<-header

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
x <- word_vectors["Obama", , drop = FALSE]
cos_sim = sim2(x = word_vectors, y = x, method = "cosine", norm = "l2")
head(sort(cos_sim[,1], decreasing = TRUE), 5)
