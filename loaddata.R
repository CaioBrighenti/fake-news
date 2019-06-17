## load LIAR dataset
train<-read.csv(file="liar_dataset/train.TSV",sep = '\t', header = FALSE)
header<-c("ID","label","statement","subject","speaker","speaker.title","state","party","bt.count","f.count","ht.count","mt.count","pof.count","context")
names(train)<-header
head(train)
# grab 10 random observations
survey_LIAR<-train[sample(nrow(train), 10), ]




## load FakeNewsNet dataset
ffn_fake<-read.csv(file="FakeNewsNet/dataset/politifact_fake.CSV",header = FALSE)
head(ffn_fake)
survey_ffn<-ffn_fake[sample(nrow(ffn_fake), 10), ]
