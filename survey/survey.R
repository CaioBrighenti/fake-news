# libraries
library(dplyr)
library(tidyverse)

survey_claims<-read.csv(file="survey/survey_claims.csv",header = TRUE)
survey_claims <- as_tibble(survey_claims)

#colors
colgate_ter <- c("#64A50A", "#F0AA00","#0096C8", "#005F46","#FF6914","#004682")

## analysize truth ratings
survey_truth <- survey_claims %>% 
  dplyr::select(ID, label, `2`=FALSE., `3`=btrue, `4`=htrue, `5`=mtrue, `6`=TRUE.) %>%
  mutate(total = `2` + `3` + `4` + `5` + `6`)
correct <- c(14,13,22,13,15,15,3,18,19,29)
adjacent <- c(14+12,27+15,13+16,15,7,8+12,7,14,9+20,7+18)
survey_truth <- survey_truth %>% 
  mutate(corr = correct, adjacent = adjacent + corr, acc = corr / total, adj = adjacent / total) %>%
  mutate(mean_truth = (2*`2`+3*`3`+4*`4`+5*`5`+6*`6`)/total)
## mean accuracy
survey_truth %>% 
  summarise(mean_acc = mean(acc), mean_adj = mean(adj))
survey_truth %>% 
  group_by(label) %>%
  summarise(mean_acc = mean(acc), mean_adj = mean(adj)) %>%
  mutate(label = replace(label, c(1,2,3,4,5,6), c("pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"))) %>%
  gather(type, value, `mean_acc`:`mean_adj`) %>% 
  ggplot(aes(x=type, y=value, fill=type)) +
  geom_bar(stat="identity") +
  facet_wrap(~label) +
  scale_fill_manual(values= colgate_ter[1:2], labels=c("Exact match", "Off by one")) +
  geom_text(aes(label=round(value, digits = 2)), vjust=1.6, color="white", size=3.5) +
  ylab("Proportion") +
  xlab("") +
  ggtitle("Survey respondent claim classification performance by label") +
  ylim(0,1)
  


# plot mean accuracy
tibble(acc = c(0.260, 0.866), type = c("exact match", "off by one")) %>%
  ggplot(aes(x=type, y=acc)) +
  geom_bar(stat="identity", fill=c("#E10028", "#821019")) +
  geom_text(aes(label=acc), vjust=1.6, color="white", size=4)+
  theme_minimal() +
  ylim(0,1) + 
  xlab("") +
  ylab("proportion")


## analyze types
survey_types <- survey_claims %>%
  dplyr::select(ID, label, omits, deceptive, lie, projection, vague, satire) %>%
  mutate(mean.truth = survey_truth$mean_truth)


## analyze justifications
survey_just <- survey_claims %>%
  dplyr::select(ID, label, speaker, knowledge, words, grammar, intuition, other)

# means
survey_just %>%
  summarise(speaker=mean(speaker), knowledge=mean(knowledge), words=mean(words), 
            grammar=mean(grammar), intuition=mean(intuition), other=mean(other))

survey_just %>%
  group_by(label) %>%
  summarise(speaker=mean(speaker), knowledge=mean(knowledge), words=mean(words), 
            grammar=mean(grammar), intuition=mean(intuition), other=mean(other))

# sums
survey_just %>%
  dplyr::select(-other) %>%
  summarise(speaker=sum(speaker), knowledge=sum(knowledge), words=sum(words), 
            grammar=sum(grammar), intuition=sum(intuition)) %>%
  gather(just, count, `speaker`:`intuition`) %>%
  ggplot(aes(x=just, y=count, fill=just)) +
  geom_bar(stat="identity") +
  scale_fill_manual(values= colgate_ter[1:5],
                    labels=c("Sentence grammar", "Intuition", "Prior knowledge of topic","Speaker's identity", "Word choice")) +
  geom_text(aes(label=count), vjust=1.6, color="white", size=3.5) +
  ggtitle("Survey respondent justifications for chosen truthfulness ratings")
  

survey_just %>%
  group_by(label) %>%
  summarise(speaker=sum(speaker), knowledge=sum(knowledge), words=sum(words), 
            grammar=sum(grammar), intuition=sum(intuition), other=sum(other))


# load in truthfulness dictionaries
## untruth
loadTruthDict <- function(){
  untruth_dict <-read.csv(file="survey/untruth_dict.csv",header = TRUE)
  untruth_dict <- as_tibble(untruth_dict) %>%
    arrange(desc(word)) %>%
    mutate(word = as.character(word))
  ## truth
  truth_dict <-read.csv(file="survey/truth_dict.csv",header = TRUE)
  truth_dict <- as_tibble(truth_dict) %>%
    arrange(desc(word)) %>%
    mutate(word = as.character(word))
  
  dict <-  truth_dict %>%
    full_join(untruth_dict, by="word", fill=0) %>%
    mutate(untruth = replace_na(untruth, 0)) %>%
    mutate(net=truth-untruth) %>%
    filter(net != 0) %>%
    mutate(rating = ifelse(net > 0,"truthful","untruthful")) 
  return(dict)
}

