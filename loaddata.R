##################################
############## LIAR ##############
##################################
############ TRAIN ###############
train<-read.csv(file="LIAR/dataset/train.TSV",sep = '\t', quote="", header = FALSE)
header<-c("ID","label","statement","subject","speaker","speaker.title","state","party","bt.count","f.count","ht.count","mt.count","pof.count","context")
names(train)<-header
head(train)
# grab 10 random observations
survey_LIAR<-train[sample(nrow(train), 10), ]

## reorder and number label
labels<-c("pants-fire","false","barely-true","half-true","mostly-true","true")
for (num in 6:1) {
  train$label <- relevel(train$label,labels[num])
}
levels(train$label)

## write LIAR to fastText format
# file.create("LIAR/train.txt")
# #out<-file("liar_dataset/train.txt")
# for (i in 1:nrow(train)) {
#   line<-paste("__label__", unclass(train$label[i]), " ",train$statement[i], sep="")
#   write(line,file="LIAR/train.txt",append=TRUE)
# }
#close(out)

############ TEST ###############
test<-read.csv(file="LIAR/dataset/test.TSV",sep = '\t', quote="", header = FALSE)
header<-c("ID","label","statement","subject","speaker","speaker.title","state","party","bt.count","f.count","ht.count","mt.count","pof.count","context")
names(test)<-header
head(test)

## reorder and number label
labels<-c("pants-fire","false","barely-true","half-true","mostly-true","true")
for (num in 6:1) {
  test$label <- relevel(test$label,labels[num])
}
levels(test$label)


## write LIAR to output
# file.create("LIAR/test.txt")
# #out<-file("liar_dataset/test.txt")
# for (i in 1:nrow(test)) {
#   line<-paste("__label__", unclass(test$label[i]), " ",test$statement[i], sep="")
#   write(line,file="LIAR/test.txt",append=TRUE)
# }
#close(out)


## sentiment analysis
#library("sentimentr")
#test <- get_sentences(as.character(train$statement))
library("syuzhet")
sentiments<-get_nrc_sentiment(as.character(train$statement))
train<-cbind(train,sentiments)
#logistic
mod.logit<-glm(label~anger+anticipation+disgust+fear+joy+sadness+surprise+
           trust+negative+positive,data=train,family = binomial(link = "logit"))

summary(mod.logit)
library(pscl)
pR2(mod.logit)
#linear
mod.lm<-lm(unclass(label)~anger+anticipation+disgust+fear+joy+sadness+surprise+
             trust+negative+positive,data=train)
summary(mod.lm)

##################################
########## FAKENEWSNET ###########
##################################
## load FakeNewsNet dataset
ffn_fake<-read.csv(file="FakeNewsNet/dataset/politifact_fake.CSV",header = FALSE)
ffn_true<-read.csv(file="FakeNewsNet/dataset/politifact_real.CSV",header = FALSE)
gossip_fake<-read.csv(file="FakeNewsNet/dataset/gossipcop_fake.CSV",header = FALSE)
gossip_true<-read.csv(file="FakeNewsNet/dataset/gossipcop_real.CSV",header = FALSE)
head(ffn_fake)

