############## LOAD ############## 
## load in libs
library("tidytext")
library(scales)
library(tidyr)
library("ggplot2")
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
  coord_flip()

# word frequency by group
frequency <- count(tidy_train,label,word,sort=TRUE) %>%
  group_by(label) %>%
  mutate(proportion = n / sum(n)) %>%
  dplyr::select(-n) %>%
  spread(label, proportion) #%>%
  #gather(label, proportion, 'pants-fire':'true')
## plot
pf_ht <- wordplot(frequency,frequency$`pants-fire`,frequency$`half-true`,"pants-fire","half-true")
pf_t <- wordplot(frequency,frequency$`pants-fire`,frequency$`true`,"pants-fire","true")
ht_t <- wordplot(frequency,frequency$`half-true`,frequency$`true`,"half-true","true")


wordplot <- function(frequency,label1,label2,x_lab,y_lab){
  p <- ggplot(frequency, aes(x = label1, y = label2, color = true)) +
  geom_abline(color = "gray40", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  theme(legend.position="none") +
  scale_color_viridis(limits = c(0, 0.001),option="B") +
  labs(y = y_lab, x = x_lab)
  return(p)
}
