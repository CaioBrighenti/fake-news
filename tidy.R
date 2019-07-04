############## LOAD ############## 
## load in libs
library("tidytext")
library("syuzhet")
library(scales)
library(tidyr)
library("ggplot2")
library("reshape2")
library("wordcloud")
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
