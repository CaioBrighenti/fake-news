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
fnn <- loadFNN()

############## LABEL FREQUENCY PLOTS ############## 
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
ggplot(fnn, aes(x=factor(label)))+
  geom_bar(aes(y = (..count..)/sum(..count..)*100), width=0.7, fill="steelblue")+
  ggtitle("Label frequency in test set") +
  xlab("Truth Label") + ylab("Frequency") +
  theme_minimal()

############## LABEL FREQ BY SPEAKER CREDIT ############## 
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


############## NORMALIZED CREDIT ############## 
train_cred<-train[,c(1,2,5,9:13)]
train_cred$sum.count <- rowSums(train_cred[,4:8])
### hack to avoid div by 0
train_cred$sum.count[which(train_cred$sum.count==0)] <- -1
train_cred[,4:8] <- train_cred[,4:8] / train_cred$sum.count

## speaker plots
par(mfrow=c(2,2))
speakerCredPlot(train_cred,"barack-obama")
speakerCredPlot(train_cred,"hillary-clinton")
speakerCredPlot(train_cred,"donald-trump")
speakerCredPlot(train_cred,"mitt-romney")

### split into groups
t_train_cred<-train_cred[which(train_cred$label=="true"),]
t_creds_mean<-colMeans(t_train_cred[,4:8])
mt_train_cred<-train_cred[which(train_cred$label=="mostly-true"),]
mt_creds_mean<-colMeans(mt_train_cred[,4:8])
ht_train_cred<-train_cred[which(train_cred$label=="half-true"),]
ht_creds_mean<-colMeans(ht_train_cred[,4:8])
bt_train_cred<-train_cred[which(train_cred$label=="barely-true"),]
bt_creds_mean<-colMeans(bt_train_cred[,4:8])
f_train_cred<-train_cred[which(train_cred$label=="false"),]
f_creds_mean<-colMeans(f_train_cred[,4:8])
pf_train_cred<-train_cred[which(train_cred$label=="pants-fire"),]
pf_creds_mean<-colMeans(pf_train_cred[,4:8])
## make bar plots
par(mfrow=c(2,3))
colors <- viridis::inferno(6)
barplot(t_creds_mean, main="True statements", col=colors)
barplot(mt_creds_mean, main="Mostly-true statements", col=colors)
barplot(ht_creds_mean, main="Half-true statements", col=colors)
barplot(bt_creds_mean, main="Barely-true statements", col=colors)
barplot(f_creds_mean, main="False statements", col=colors)
barplot(pf_creds_mean, main="Pants-on-fire statements", col=colors)


############## HELPER FUNCTIONS ############## 
speakerCredPlot <- function(cred_dat,speaker,norm=FALSE){
  speaker_cred <- cred_dat[which(cred_dat$speaker == speaker),][1,]
  if (norm == FALSE){
    speaker_cred[4:8] <- speaker_cred[4:8] * speaker_cred$sum.count
  }
  barplot(as.matrix(speaker_cred[4:8]),main=paste(speaker,"credibility history"),col=colors,beside=TRUE)
}
