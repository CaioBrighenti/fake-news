############## LOAD ############## 
## load in libs
library(tidyverse)
library(RVAideMemoire)
library(FSA)
library(boot)
library(progress)
library(ggstance)
library(ggforce)

# load in data
train <- loadFNNtxtfeat("train")
test <- loadFNNtxtfeat("test")
train_titles <- loadFNNtxtfeat("train_titles")
test_titles <- loadFNNtxtfeat("test_titles")

# individual parts
train_complexity <- loadFNNComplexity("train")
train_LIWC <- loadFNNLIWC("train")
train_POS <- loadFNNPOS("train")
train_NER <- loadFNNNER("train")


############## LOOK FOR OUTLIERS ############## 



# group differences
train_conf_fake <- tibble(
  var = names(dplyr::select(train,-label)),
  label = rep(1,ncol(train) - 1),
  median = rep(NA,ncol(train) - 1),
  lower_bound = rep(NA,ncol(train) - 1),
  upper_bound = rep(NA,ncol(train) - 1),
)

train_conf_true <- tibble(
  var = names(dplyr::select(train,-label)),
  label = rep(0,ncol(train) - 1),
  median = rep(NA,ncol(train) - 1),
  lower_bound = rep(NA,ncol(train) - 1),
  upper_bound = rep(NA,ncol(train) - 1),
)

# setup bootstrap function
R<-1000
boot.median<-function(data,indices){
  d<-data[indices]  # allows boot to select sample
  return(median(d))
}

# bootstrap each
pb <- progress_bar$new(total = ncol(train)-1)
for (idx in seq(2,ncol(train))) {
  pb$tick()
  # extract var
  var_fake <- pull(filter(train,label==1)[,idx])
  
  # boostrap CI
  var.boot<-boot(var_fake,R=R,statistic=boot.median)
  var.ci<-boot.ci(boot.out=var.boot,conf=0.95,type="perc")
  
  # pull out bounds
  train_conf_fake[idx-1,]$median <- var.boot$t0
  if (is.null(var.ci)){
    train_conf_fake[idx-1,]$lower_bound <- NA
    train_conf_fake[idx-1,]$upper_bound <- NA
  } else {
    train_conf_fake[idx-1,]$lower_bound <- var.ci$percent[4]
    train_conf_fake[idx-1,]$upper_bound <- var.ci$percent[5]
  }
  
  
  # repeat for real label
  # extract var
  var_true <- pull(filter(train,label==0)[,idx])
  
  # boostrap CI
  var.boot<-boot(var_true,R=R,statistic=boot.median)
  var.ci<-boot.ci(boot.out=var.boot,conf=0.95,type="perc")
  
  # pull out bounds
  train_conf_true[idx-1,]$median <- var.boot$t0
  if (is.null(var.ci)){
    train_conf_true[idx-1,]$lower_bound <- NA
    train_conf_true[idx-1,]$upper_bound <- NA
  } else {
    train_conf_true[idx-1,]$lower_bound <- var.ci$percent[4]
    train_conf_true[idx-1,]$upper_bound <- var.ci$percent[5]
  }
}

# merge
train_conf <- rbind(train_conf_true,train_conf_fake) %>%
  mutate(lower_bound = as.numeric(lower_bound),
         upper_bound = as.numeric(upper_bound),
         label = if_else(label == 1,"Fake","Real"))

# plots
names(train)

# first attempt, too noisy
train_conf %>% 
  filter(var %in% colnames(train_complexity)) %>%
  mutate(var = as.numeric(as.factor(var)),
         var = case_when(label == "Real" ~ var + 0.25,
                        label == "Fake" ~ var - 0.25),
         lower_bound = as.numeric(lower_bound),
         upper_bound = as.numeric(upper_bound)) %>%
  ggplot(aes(x=median,y=var)) +
  geom_point() +
  geom_errorbarh(aes(xmin=lower_bound,xmax=upper_bound)) 

# facet plot
### complexity
train_conf %>%
  filter(var %in% colnames(train_complexity)) %>%
  mutate(label = as.factor(label)) %>%
  ggplot(aes(y=label,x=median)) +
  geom_point() +
  geom_errorbarh(aes(xmin=lower_bound,xmax=upper_bound)) +
  facet_wrap(~var,scales="free_x") +
  theme_bw() +
  labs(title = "Confidence interval for medians of textual properties for fake and real articles",
       subtitle = "Variables capturing textual complexity",
       x = "Median",
       y = "")

### LIWC
for (i in seq(1,5)){
  p<-train_conf %>%
    filter(var %in% colnames(train_LIWC)) %>%
    mutate(label = as.factor(label)) %>%
    ggplot(aes(y=label,x=median)) +
    geom_point() +
    geom_errorbarh(aes(xmin=lower_bound,xmax=upper_bound)) +
    facet_wrap_paginate(~var,scales="free_x",nrow=4,ncol=5,page=i) +
    theme_bw() +
    labs(title = "Confidence interval for medians of textual properties for fake and real articles",
         subtitle = "Variables capturing properties from LWIC dictionary",
         x = "Median",
         y = "")
  print(p)
}

### POS
train_conf %>%
  filter(var %in% colnames(train_POS)) %>%
  mutate(label = as.factor(label)) %>%
  ggplot(aes(y=label,x=median)) +
  geom_point() +
  geom_errorbarh(aes(xmin=lower_bound,xmax=upper_bound)) +
  facet_wrap(~var,scales="free_x") +
  theme_bw() +
  labs(title = "Confidence interval for medians of textual properties for fake and real articles",
       subtitle = "Variables capturing parts-of-speech distribution",
       x = "Median",
       y = "")

### NER
train_conf %>%
  filter(var %in% colnames(train_NER)) %>%
  mutate(label = as.factor(label)) %>%
  ggplot(aes(y=label,x=median)) +
  geom_point() +
  geom_errorbarh(aes(xmin=lower_bound,xmax=upper_bound)) +
  facet_wrap(~var,scales="free_x") +
  theme_bw() +
  labs(title = "Confidence interval for medians of textual properties for fake and real articles",
       subtitle = "Variables capturing named-entity-recognition",
       x = "Median",
       y = "")


