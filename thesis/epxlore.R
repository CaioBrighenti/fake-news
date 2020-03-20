############## LOAD ############## 
## load in libs
library(tidyverse)
library(RVAideMemoire)
library(FSA)
library(boot)
library(progress)
library(ggstance)
library(ggforce)
library(xtable)

# load helpers
source("helpers/loaddata.R")
source("helpers/helpers.R")

# load in data
train <- loadFNNtxtfeat("train")
test <- loadFNNtxtfeat("test")
train_titles <- loadFNNtxtfeat("train_titles")
test_titles <- loadFNNtxtfeat("test_titles")

# individual parts
train_complexity <- loadFNNComplexity("train") %>% filter(!grepl("gossipcop",ID))
train_LIWC <- loadFNNLIWC("train") %>% filter(!grepl("gossipcop",ID))
train_POS <- loadFNNPOS("train") %>% filter(!grepl("gossipcop",ID))
train_NER <- loadFNNNER("train") %>% filter(!grepl("gossipcop",ID))

# LWIC groups
LWIC_groups <- loadLIWCGroups()

############## LOOK FOR OUTLIERS ############## 



############## EXPLORE GROUP DIFFERENCES ############## 
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
         label = if_else(label == 1,"Fake","Real"),
         group = case_when(
    var %in% names(train_complexity) ~ "complexity",
    var %in% names(train_POS) ~ "POS",
    var %in% names(train_NER) ~ "NER"
  )) %>%
  left_join(LWIC_groups, by=c("var")) %>%
  mutate(group.x = if_else(is.na(group.x),group.y,group.x)) %>%
  rename(group = group.x) %>%
  dplyr::select(-group.y)

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

### all groups
for (var_group in unique(train_conf$group)){
  p<-train_conf %>%
    filter(group == var_group) %>%
    mutate(label = as.factor(label)) %>%
    ggplot(aes(y=label,x=median)) +
    geom_point() +
    geom_errorbarh(aes(xmin=lower_bound,xmax=upper_bound)) +
    facet_wrap(~var,scales="free_x") +
    theme_bw() +
    labs(title = "Confidence interval for medians of textual properties for fake and real articles",
         subtitle = paste("Variables capturing",var_group),
         x = "Median",
         y = "")
  print(p)
}

# tables
pvals <- tibble(
  var = names(train[,-1]),
  pval = rep(NA,ncol(train) - 1)
)

for (idx in seq(2,ncol(train))) {
  test_temp<-mood.medtest(unlist(train[,idx],use.names = FALSE) ~ train$label,
                  exact = FALSE)
  pvals[idx-1,]$pval <- test_temp$p.value
}

# add group names
pval_table <- pvals %>%
  left_join(
    dplyr::select(train_conf,var,group),
    by=c("var")
  ) %>%
  left_join(
    dplyr::select(train_conf,var,label,median),
    by=c("var")
  ) %>%
  distinct() %>%
  spread(label,median) %>%
  dplyr::select(var,group,Fake,Real,pval)

for (var_group in unique(pval_table$group)) {
  vars_temp <- filter(pval_table, group == var_group) %>%
    dplyr::select(-group)
  
  x<-print.xtable(xtable(vars_temp))
  
  write(x,file="thesis/tables.txt",append=TRUE)
  write("\n\n",file="thesis/tables.txt",append=TRUE)
}




