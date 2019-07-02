############## LOAD ############## 
## load in libs
library("text2vec")
library("tidytext")
library("ggplot2")
library("viridis")
# load in data
source("loaddata.R")
train <- loadLIARTrain()
test <- loadLIARTest()

## label frequency plots
ggplot(train, aes(x=factor(label)))+
  geom_bar(aes(y = (..count..)/sum(..count..)*100), width=0.7, fill="steelblue")+
  ggtitle("Label frequency in train set") +
  xlab("Truth Label") + ylab("Frequency") +
  theme_minimal()
ggplot(test, aes(x=factor(label)))+
  geom_bar(aes(y = (..count..)/sum(..count..)*100), width=0.7, fill="steelblue")+
  ggtitle("Label frequency in test set") +
  xlab("Truth Label") + ylab("Frequency") +
  theme_minimal()

## label frequency by speaker credit
names(train)
dim(train)
train <- train[,c(1:8,13,10,9,11,12,14)]
### split into groups
t_train<-train[which(train$label=="true"),]
t_creds<-colSums(t_train[,9:13])
mt_train<-train[which(train$label=="mostly-true"),]
mt_creds<-colSums(mt_train[,9:13])
ht_train<-train[which(train$label=="half-true"),]
ht_creds<-colSums(ht_train[,9:13])
bt_train<-train[which(train$label=="barely-true"),]
bt_creds<-colSums(bt_train[,9:13])
f_train<-train[which(train$label=="false"),]
f_creds<-colSums(f_train[,9:13])
pf_train<-train[which(train$label=="pants-fire"),]
pf_creds<-colSums(pf_train[,9:13])
## make bar plots
par(mfrow=c(2,3))
colors <- viridis::inferno(6)
barplot(t_creds, main="True statements", col=colors)
barplot(mt_creds, main="Mostly-true statements", col=colors)
barplot(ht_creds, main="Half-true statements", col=colors)
barplot(bt_creds, main="Barely-true statements", col=colors)
barplot(f_creds, main="False statements", col=colors)
barplot(pf_creds, main="Pants-on-fire statements", col=colors)
## normalize counts
train_cred<-train[,c(1,2,5,9:13)]
train_cred$sum.count <- rowSums(train_cred[,4:8])
### hack to avoid div by 0

train_cred[,4:8] <- train_cred[,4:8] / train_cred$sum.count

