# libraries
library(dplyr)
library(tidyverse)

survey_claims<-read.csv(file="survey/survey_claims.csv",header = TRUE)
survey_claims <- as_tibble(survey_claims)


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
  summarise(mean_acc = mean(acc), mean_adj = mean(adj))


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

