##################################
############## LIAR ##############
##################################
############ TRAIN ###############
train<-read.csv(file="liar_dataset/train.TSV",sep = '\t', header = FALSE)
header<-c("ID","label","statement","subject","speaker","speaker.title","state","party","bt.count","f.count","ht.count","mt.count","pof.count","context")
names(train)<-header
head(train)
# grab 10 random observations
survey_LIAR<-train[sample(nrow(train), 10), ]
# remove garbage observations
train<-train[-which(as.character(train$label)=="brian-kemp"),]
train<-train[-which(as.character(train$label)=="barack-obama"),]
train<-train[-which(as.character(train$label)=="bill-mccollum"),]
train<-train[-which(as.character(train$label)=="doug-macginnitie"),]
# hack to get rid of bad levels
train$label<-as.character(train$label)
train$label<-as.factor(train$label)
# reorder levels
labels<-c("pants-fire","false","barely-true","half-true","mostly-true","true")
for (num in 6:1) {
  train$label <- relevel(train$label,labels[num])
}
levels(train$label)
train$label<-unclass(train$label)
# hack to get reset levels
train$label<-as.numeric(train$label)
train$label<-as.factor(train$label)

## write LIAR to output
file.create("liar_dataset/train.txt")
#out<-file("liar_dataset/train.txt")
for (i in 1:nrow(train)) {
  line<-paste("__label__", as.character(train$label[i]), " ",train$statement[i], sep="")
  if (grepl("json", line)==TRUE){
    write(line,file="liar_dataset/train.txt",append=TRUE)
  }
}
#close(out)

############ TEST ###############
test<-read.csv(file="liar_dataset/test.TSV",sep = '\t', header = FALSE)
header<-c("ID","label","statement","subject","speaker","speaker.title","state","party","bt.count","f.count","ht.count","mt.count","pof.count","context")
names(test)<-header
head(test)
# reorder levels
labels<-c("pants-fire","false","barely-true","half-true","mostly-true","true")
for (num in 6:1) {
  test$label <- relevel(test$label,labels[num])
}
levels(test$label)
test$label<-unclass(test$label)
# hack to get reset levels
test$label<-as.numeric(test$label)
test$label<-as.factor(test$label)

## write LIAR to output
file.create("liar_dataset/test.txt")
#out<-file("liar_dataset/test.txt")
for (i in 1:nrow(test)) {
  line<-paste("__label__", as.character(test$label[i]), " ",test$statement[i], sep="")
  write(line,file="liar_dataset/test.txt",append=TRUE)
}
#close(out)


## load FakeNewsNet dataset
ffn_fake<-read.csv(file="FakeNewsNet/dataset/politifact_fake.CSV",header = FALSE)
ffn_true<-read.csv(file="FakeNewsNet/dataset/politifact_real.CSV",header = FALSE)
gossip_fake<-read.csv(file="FakeNewsNet/dataset/gossipcop_fake.CSV",header = FALSE)
gossip_true<-read.csv(file="FakeNewsNet/dataset/gossipcop_real.CSV",header = FALSE)
head(ffn_fake)
survey_ffn<-ffn_fake[sample(nrow(ffn_fake), 10), ]
survey_gossip<-gossip_fake[sample(nrow(gossip_fake), 10), ]
  

