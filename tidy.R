############## LOAD ############## 
## load in libs
library("tidytext")
library("dplyr")
library("syuzhet")
library(scales)
library(tidyr)
library("ggplot2")
library(ggraph)
library("sentimentr")
library("reshape2")
library("wordcloud")
library(igraph)
library("corrplot")
library("viridis")
# load in data
source("loaddata.R")
train <- loadLIARTrain()

# tidy data
names(train)
train_df <- as_tibble(train[,1:3,])
train_df$statement <- as.character(train_df$statement)
tidy_train <- train_df %>% 
  unnest_tokens(word, statement)

# pre-process
data(stop_words)
tidy_train <- tidy_train %>%
    anti_join(stop_words)

############## WORD FREQUENCY ############## 
# word frequency
tidy_train %>%
  count(word, sort = TRUE) 
## plot
tidy_train %>%
  count(word, sort = TRUE) %>%
  filter(n > 200) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip() +
  ggtitle("Most common words in train set")

# word frequency by group
frequency <- count(tidy_train,label,word,sort=TRUE) %>%
  group_by(label) %>%
  mutate(proportion = n / sum(n)) %>%
  dplyr::select(-n) %>%
  spread(label, proportion) %>%
  gather(label, proportion, 'pants-fire':'true') 
## facet plot
t_prop <- frequency[frequency$label=="true",]$proportion
ggplot(frequency, aes(x = proportion, y = rep(t_prop,6), color = proportion)) +
  geom_abline(color = "gray40", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  theme(legend.position="none") +
  scale_color_viridis(limits = c(0, 0.001),option="B") +
  facet_wrap(~label, ncol = 3) +
  labs(y = "true", x = NULL) +
  ggtitle("Plots of word frequency proportion in true group vs other groups")


# word frequency by group
prop.aov <- aov(proportion ~ label, data = frequency)
summary(prop.aov)
TukeyHSD(prop.aov)

frequency <- frequency %>%
  spread(label, proportion)
frequency[is.na(frequency)] <- 0
# correlation tests
cor <- cor(frequency[,2:7])
corrplot(cor, type = "upper", 
         tl.col = "black", tl.srt = 45)

############## SENTIMENT ANALYSIS ############## 
tidy_train <- tidy_train %>%
  group_by(label)

## check most positive words
bing_pos <- get_sentiments("bing") %>% 
  filter(sentiment == "positive")
tidy_train %>%
  filter(label == "true") %>%
  inner_join(bing_pos) %>%
  count(word, sort = TRUE)

## sentiment by label + doc
train_sentiment <- tidy_train %>%
  inner_join(get_sentiments("bing")) %>%
  count(label, index=ID, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)

ggplot(train_sentiment, aes(index, sentiment, fill = label)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~label, ncol = 2, scales = "free_x") +
  ggtitle("Claim net sentiment by label")

## sentiment by label
label_sentiment <- train_sentiment %>% 
  group_by(label) %>%
  summarise(negative=-sum(negative),positive = sum(positive), sentiment=sum(sentiment))

dfm <- melt(label_sentiment[,c('label','negative','positive', 'sentiment')],id.vars = 1)
ggplot(dfm,aes(x = label,y = value)) + 
  geom_bar(aes(fill = variable),stat = "identity",position = "stack") +
  geom_hline(yintercept=0) +
  ggtitle("Net sentiment by label")

## sentiment counts
bing_word_counts <- tidy_train %>%
  ungroup() %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE)

bing_word_counts %>%
  group_by(sentiment) %>%
  top_n(30) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to sentiment",
       x = NULL) +
  coord_flip()

# word clouds
##full data
tidy_train %>%
  ungroup() %>%
  anti_join(stop_words) %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100))

## comparison
tidy_train %>%
  count(word, sort = TRUE) %>%
  acast(word ~ label, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = viridis(6),
                   max.words = 300)


############## TFIDF ############## 
## BY LABEL
train_words <- train_df %>%
  unnest_tokens(word, statement) %>%
  count(label, word, sort = TRUE)

total_words <- train_words %>% 
  group_by(label) %>% 
  summarize(total = sum(n))

train_words <- left_join(train_words, total_words)
train_words

ggplot(train_words, aes(n/total, fill = label)) +
  geom_histogram(show.legend = FALSE) +
  xlim(NA, 0.0009) +
  facet_wrap(~label, ncol = 2, scales = "free_y")

freq_by_rank <- train_words %>% 
  group_by(label) %>% 
  mutate(rank = row_number(), 
         `term frequency` = n/total)

freq_by_rank

freq_by_rank %>% 
  ggplot(aes(rank, `term frequency`, color = label)) + 
  geom_line(size = 1.1, alpha = 0.8, show.legend = TRUE) + 
  scale_x_log10() +
  scale_y_log10()

train_words <- train_words %>%
  bind_tf_idf(word, label, n)
train_words

train_words %>%
  arrange(desc(tf_idf))

train_words %>%
  arrange(desc(tf_idf)) %>%
  top_n(100) %>% 
  ggplot(aes(word, tf_idf, fill = label)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~label, ncol = 2, scales = "free") +
  coord_flip()

## BY DOCUMENT
train_words <- tidy_train %>%
  count(label, index=ID, word, sort = TRUE)

total_words <- train_words %>% 
  group_by(index) %>% 
  summarize(total = sum(n))

train_words <- left_join(train_words,total_words)
train_words

train_words <- train_words %>%
  bind_tf_idf(word, index, n)
train_words

train_words %>%
  arrange(desc(tf_idf))

train_words %>%
  arrange(desc(tf_idf)) %>%
  top_n(15) %>% 
  ggplot(aes(word, tf_idf, fill = label)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~label, ncol = 2, scales = "free") +
  coord_flip()

############## N-GRAMS ############## 
train_bigrams <- train_df %>%
  unnest_tokens(bigram, statement, token = "ngrams", n = 2)

train_bigrams %>%
  count(bigram, sort = TRUE)

# remove stopwords
bigrams_separated <- train_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)

# new bigram counts:
bigram_counts <- bigrams_filtered %>% 
  count(word1, word2, sort = TRUE)
bigram_counts

# recombine bigrams
bigrams_united <- bigrams_filtered %>%
  unite(bigram, word1, word2, sep = " ")

bigrams_united

# trigrams
train_df %>%
  unnest_tokens(trigram, statement, token = "ngrams", n = 3) %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word3 %in% stop_words$word) %>%
  count(word1, word2, word3, sort = TRUE)

# common word pairs with input word
bigrams_filtered %>%
  filter(word1 == "donald") %>%
  count(label, word2, sort = TRUE)

# tfidf
bigram_tf_idf <- bigrams_united %>%
  count(label, bigram, sort = TRUE) %>%
  bind_tf_idf(bigram, label, n) %>%
  arrange(desc(tf_idf))

bigram_tf_idf

# plot bigrams
bigram_tf_idf %>%
  arrange(desc(tf_idf)) %>%
  top_n(150) %>% 
  ggplot(aes(bigram, tf_idf, fill = label)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~label, ncol = 2, scales = "free") +
  coord_flip()

# bigram sentiment analysis
bigrams_separated %>%
  filter(word1 == "not") %>%
  count(word1, word2, sort = TRUE)

AFINN <- get_sentiments("afinn")
not_words <- bigrams_separated %>%
  filter(word1 == "not") %>%
  inner_join(AFINN, by = c(word2 = "word")) %>%
  count(word2, value, sort = TRUE)
not_words

# plot not words
not_words %>%
  mutate(contribution = n * value) %>%
  arrange(desc(abs(contribution))) %>%
  head(20) %>%
  mutate(word2 = reorder(word2, contribution)) %>%
  ggplot(aes(word2, contribution, fill = contribution > 0)) +
  geom_col(show.legend = FALSE) +
  xlab("Words preceded by \"not\"") +
  ylab("Sentiment score * number of occurrences") +
  coord_flip()

# other negations
negation_words <- c("not", "no", "never", "without")

negated_words <- bigrams_separated %>%
  filter(word1 %in% negation_words) %>%
  inner_join(AFINN, by = c(word2 = "word")) %>%
  count(word1, word2, value, sort = TRUE)

# plot other negations
negated_words %>%
  mutate(contribution = n * value) %>%
  arrange(desc(abs(contribution))) %>%
  head(40) %>%
  mutate(word2 = reorder(word2, contribution)) %>%
  ggplot(aes(word2, contribution, fill = contribution > 0)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~word1, ncol = 3, scales = "free") +
  xlab("Words preceded by \"not\",\"no\",\"never\",\"without\"") +
  ylab("Sentiment score * number of occurrences") +
  coord_flip()

# original counts
bigram_counts

# filter for only relatively common combinations
bigram_graph <- bigram_counts %>%
  filter(n > 20) %>%
  graph_from_data_frame()

bigram_graph

# graph
set.seed(2016)
a <- grid::arrow(type = "closed", length = unit(.075, "inches"))
ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(show.legend = FALSE,
                 arrow = a, end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()


############## WORD ODDS BY LABEL ############## 
##
tidy_train
##
word_ratios <- tidy_train %>%
  count(word, label) %>%
  group_by(word) %>%
  filter(sum(n) >= 10) %>%
  ungroup() %>%
  spread(label, n, fill = 0) %>%
  mutate_if(is.numeric, funs((. + 1) / (sum(.) + 1))) %>%
  mutate(logratio = log(true / `pants-fire`)) %>%
  arrange(desc(logratio))
# equally likely words
word_ratios %>% 
  arrange(abs(logratio))

# plot most likely in pof vs true
word_ratios %>%
  group_by(logratio < 0) %>%
  top_n(15, abs(logratio)) %>%
  ungroup() %>%
  mutate(word = reorder(word, logratio)) %>%
  ggplot(aes(word, logratio, fill = logratio < 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  ylab("log odds ratio (true/pants-fire)") +
  scale_fill_discrete(name = "", labels = c("true", "pants-fire"))
