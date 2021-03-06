############## LOAD ############## 
## load in libs
library(tidyverse)
library(glmnet)
library(caret)
library(ggrepel)
library(margins)
library(extrafont)
library(lubridate)
library(knitr)
library(kableExtra)
library(RVAideMemoire)
library(ggforce)
library(tidytext)


# load fonts
#font_import(paths = "C:/Users/caiob/Downloads/Roboto")
loadfonts(device = "win", quiet = TRUE)

# load helpers
source("helpers/loaddata.R")
source("helpers/helpers.R")

# load in articles
fnn <- loadFNN() %>% filter(!grepl("gossipcop",ID))
train_articles <- loadFNNTrain() %>% filter(!grepl("gossipcop",ID))
test_articles <- loadFNNTest() %>% filter(!grepl("gossipcop",ID))

# load in text features
train <- loadFNNtxtfeat("train") %>% filter(!grepl("gossipcop",ID)) %>%
  rename(function. = `function`)
test <- loadFNNtxtfeat("test") %>% filter(!grepl("gossipcop",ID)) %>%
  rename(function. = `function`)
train_titles <- loadFNNtxtfeat("train_titles") %>% filter(!grepl("gossipcop",ID)) %>%
  rename(function. = `function`)
test_titles <- loadFNNtxtfeat("test_titles") %>% filter(!grepl("gossipcop",ID)) %>%
  rename(function. = `function`)

# individual parts
train_complexity <- loadFNNComplexity("train") %>% filter(!grepl("gossipcop",ID))
train_LIWC <- loadFNNLIWC("train") %>% filter(!grepl("gossipcop",ID))
train_POS <- loadFNNPOS("train") %>% filter(!grepl("gossipcop",ID))
train_NER <- loadFNNNER("train") %>% filter(!grepl("gossipcop",ID))

# LWIC groups
LWIC_groups <- loadLIWCGroups()

# get lengths
fnn$body_length <- str_length(fnn$text)
fnn$title_length <- str_length(fnn$title)

# get stopword count
text_stopwords <- fnn %>% 
  dplyr::select(-title) %>%
  unnest_tokens(word, text) %>%
  filter(word %in% stop_words$word) %>%
  group_by(ID) %>%
  count() %>%
  rename(stopwords = n)

title_stopwords <-  fnn %>% 
  dplyr::select(-text) %>%
  unnest_tokens(word, title) %>%
  filter(word %in% stop_words$word) %>%
  group_by(ID) %>%
  count() %>%
  rename(stopwords = n)

train <- left_join(train, text_stopwords) %>% mutate(stopwords = ifelse(is.na(stopwords),0,stopwords))
test <- left_join(test, text_stopwords) %>% mutate(stopwords = ifelse(is.na(stopwords),0,stopwords))
train_titles <- left_join(train_titles, title_stopwords) %>% mutate(stopwords = ifelse(is.na(stopwords),0,stopwords))
test_titles <- left_join(test_titles, title_stopwords) %>% mutate(stopwords = ifelse(is.na(stopwords),0,stopwords))

# add ALL CAPS
text_allcaps <- mutate(fnn, all_caps = str_count(text, "[A-Z]")) %>% dplyr::select(ID, all_caps)
title_allcaps <- mutate(fnn, all_caps = str_count(title, "[A-Z]")) %>% dplyr::select(ID, all_caps)

train <- left_join(train, text_allcaps)
test <- left_join(test, text_allcaps)
train_titles <- left_join(train_titles, title_allcaps)
test_titles <- left_join(test_titles, title_allcaps)

# reorder variables
colnames(train)
colnames(train_titles)
train <- train %>% dplyr::select(ID:ARI,all_caps,stopwords,everything())
test <- test %>% dplyr::select(ID:ARI,all_caps,stopwords,everything())
train_titles <- train_titles %>% dplyr::select(ID:ARI,all_caps,stopwords,everything())
test_titles <- test_titles %>% dplyr::select(ID:ARI,all_caps,stopwords,everything())

############## PREPROCCESS ##############
# NORMALIZE BY WORD COUNT
train[,c(23,seq(118,154))] <- train[,c(23,seq(118,154))] / train$WC * 100
test[,c(23,seq(118,154))] <- test[,c(23,seq(118,154))] / test$WC * 100
train_titles[,c(23,seq(118,154))] <- train_titles[,c(23,seq(118,154))] / train_titles$WC * 100
test_titles[,c(23,seq(118,154))] <- test_titles[,c(23,seq(118,154))] / test_titles$WC * 100

# normalize all caps
train <- left_join(train, dplyr::select(fnn,ID,body_length))
test <- left_join(test, dplyr::select(fnn,ID,body_length))
train$all_caps <- train$all_caps / train$body_length * 100
test$all_caps <- test$all_caps / test$body_length * 100

train_titles <- left_join(train_titles, dplyr::select(fnn,ID,title_length))
test_titles <- left_join(test_titles, dplyr::select(fnn,ID,title_length))
train_titles$all_caps <- train_titles$all_caps / train_titles$title_length * 100
test_titles$all_caps <- test_titles$all_caps / test_titles$title_length * 100

train <- dplyr::select(train, -body_length)
test <- dplyr::select(test, -body_length)
train_titles <- dplyr::select(train_titles, -title_length)
test_titles <- dplyr::select(test_titles, -title_length)

# remove tree depth for title 
train_titles <- train_titles %>% dplyr::select(-("mu_sentence":"num_verb_phrase"))
test_titles <- test_titles %>% dplyr::select(-("mu_sentence":"num_verb_phrase"))


# remove constant variance columns
const_cols_train <- which(apply(train_titles, 2, var)==0)
const_cols_test <- which(apply(test_titles, 2, var)==0) 
const_cols <- c(const_cols_train, const_cols_test)
train_titles <- train_titles[,-const_cols]
test_titles <- test_titles[,-const_cols]

# created merged df
fnn_titles <- rbind(train_titles, test_titles)
fnn_text <- rbind(train, test)


############## PCA ############## 
library(factoextra)
res.pca <- prcomp(fnn_titles[,-c(1,2)], scale = TRUE)

fviz_eig(res.pca)


fviz_pca_var(res.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

# Results for Variables
res.var <- get_pca_var(res.pca)
coord <- as_tibble(res.var$coord) %>% mutate(var = row.names(res.var$coord)) %>% dplyr::select(var,everything())
contrib <- as_tibble(res.var$contrib) %>% mutate(var = row.names(res.var$contrib)) %>% dplyr::select(var,everything())

coord %>%
  left_join(contrib, by = c("var"), suffix = c(".coord",".contrib")) %>%
  dplyr::select(var, Dim.1.contrib, Dim.2.contrib, Dim.1.coord, Dim.2.coord) %>% 
  mutate(contrib = pmax(Dim.1.contrib,Dim.2.contrib)) %>%
  filter(contrib >= mean(contrib) + (sd(contrib) * 1)) %>%
  ggplot(aes(x=Dim.1.coord,y=Dim.2.coord,color=contrib)) +
  geom_text_repel(aes(label=var)) +
  geom_segment(aes(x=0,y=0,xend = Dim.1.coord, yend = Dim.2.coord),
               arrow = arrow(length = unit(0.01, "npc"))) + 
  geom_hline(yintercept = 0, linetype = "longdash") + 
  geom_vline(xintercept = 0, linetype = "longdash") + 
  geom_circle(aes(x0=0,y0=0,r=1), inherit.aes = FALSE) + 
  scale_x_continuous(limits = c(-1,1)) + 
  scale_y_continuous(limits = c(-1,1)) + 
  theme_minimal() +
  theme(
    aspect.ratio=1,
    legend.position = "bottom",
    legend.key.width = unit(1, "cm")
  ) +
  labs(title = "Highest contributing variables for 1st and 2nd dimensions of PCA",
       x = "Dim 1 Coordinate",
       y= "Dim 2 Coordinate",
       color = "Contribution"
       )
ggsave("thesis/plots/pca1.png", dpi=600)

coord %>%
  left_join(contrib, by = c("var"), suffix = c(".coord",".contrib")) %>%
  dplyr::select(var, Dim.3.contrib, Dim.4.contrib, Dim.3.coord, Dim.4.coord) %>% 
  mutate(contrib = pmax(Dim.3.contrib,Dim.4.contrib)) %>%
  filter(contrib >= mean(contrib) + (sd(contrib) * 1)) %>%
  ggplot(aes(x=Dim.3.coord,y=Dim.4.coord,color=contrib)) +
  geom_text_repel(aes(label=var)) +
  geom_segment(aes(x=0,y=0,xend = Dim.3.coord, yend = Dim.4.coord),
               arrow = arrow(length = unit(0.01, "npc"))) + 
  geom_hline(yintercept = 0, linetype = "longdash") + 
  geom_vline(xintercept = 0, linetype = "longdash") + 
  geom_circle(aes(x0=0,y0=0,r=1), inherit.aes = FALSE) + 
  scale_x_continuous(limits = c(-1,1)) + 
  scale_y_continuous(limits = c(-1,1)) + 
  theme_minimal() +
  theme(
    aspect.ratio=1,
    legend.position = "bottom",
    legend.key.width = unit(1, "cm")
  ) +
  labs(title = "Highest contributing variables for 1st and 2nd dimensions of PCA",
       x = "Dim 1 Coordinate",
       y= "Dim 2 Coordinate",
       color = "Contribution"
  )
ggsave("thesis/plots/pca2.png", dpi=600)


# Results for observations
res.obs <- get_pca_ind(res.pca)
coord.ind <- as_tibble(res.obs$coord) %>% mutate(ID = fnn_titles$ID) %>% dplyr::select(ID,everything())
contrib.ind <- as_tibble(res.obs$contrib) %>% mutate(ID = fnn_titles$ID) %>% dplyr::select(ID,everything())

# grab fake and real
fake_coords <- left_join(coord.ind, train_articles) %>% filter(label == "fake") %>% dplyr::select(ID:Dim.4)
real_coords <- left_join(coord.ind, train_articles) %>% filter(label == "real") %>% dplyr::select(ID:Dim.4)

# get closest neighbor
## fake news
fake_knn <- get.knn(fake_coords[,-1],k=1)
dists <- fake_knn$nn.dist
dists[dists == 0] <- 100
fake1 <- which.min(dists)
fake2 <- fake_knn$nn.index[fake1]
## reals news
real_knn <- get.knn(real_coords[,-1],k=1)
dists <- real_knn$nn.dist
dists[dists == 0] <- 100
real1 <- which.min(dists)
real2 <- real_knn$nn.index[real1]


## DIM 1
fake_examples <- fake_coords[c(fake1,fake2),c(1,2,3)] %>%
  left_join(fnn, by = c("ID"))
real_examples <- coord.ind[c(fake1,fake2),c(1,2,3)] %>%
  left_join(fnn, by = c("ID"))


exs <- rbind(fake_examples, real_examples)

exs %>%
  ggplot(aes(x=Dim.1,y=Dim.2)) +
  geom_point(aes(color=label)) +
  geom_text_repel(aes(label=title, color=label)) +
  geom_hline(yintercept = 0, linetype = "longdash") +
  geom_vline(xintercept = 0, linetype = "longdash") +
  scale_x_continuous(limits = c(-7,7)) +
  scale_y_continuous(limits = c(-7,7)) + 
  theme_minimal() +
  theme(
    aspect.ratio=1
  )


### DIM 2
fake_examples <- coord.ind[c(308,17),c(1,2,3,4,5)] %>%
  left_join(fnn, by = c("ID"))
real_examples <- coord.ind[c(484,346),c(1,2,3,4,5)] %>%
  left_join(fnn, by = c("ID"))

plot(fake_coords$Dim.1, fake_coords$Dim.2)

exs <- rbind(fake_examples, real_examples)

exs %>%
  ggplot(aes(x=Dim.1,y=Dim.2)) +
  geom_point(aes(color=label)) +
  geom_text_repel(aes(label=title, color=label)) +
  geom_hline(yintercept = 0, linetype = "longdash") +
  geom_vline(xintercept = 0, linetype = "longdash") +
  scale_x_continuous(limits = c(-7,7)) +
  scale_y_continuous(limits = c(-7,7)) + 
  theme_minimal() +
  theme(
    aspect.ratio=1
  )



############## SCALE ##############
# scale data
train[,-c(1,2)] <- scale(train[,-c(1,2)])
test[,-c(1,2)] <- scale(test[,-c(1,2)])
train_titles[,-c(1,2)] <- scale(train_titles[,-c(1,2)])
test_titles[,-c(1,2)] <- scale(test_titles[,-c(1,2)])



############## FIT LASSO ############## 
fitLasso <- function(train_data, seed = 13){
  # Dumy code categorical predictor variables
  x <- model.matrix(label ~ . - ID, train_data)[,-1]
  
  # Convert the outcome (class) to a numerical variable
  y <- train_data$label
  
  # Find the best lambda using cross-validation
  set.seed(seed) 
  print("Starting cross validation")
  cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
  
  #visualize
  plot(cv.lasso)
  cv.lasso$lambda.min
  cv.lasso$lambda.1se
  
  # Fit the final model on the training data
  print("Fitting final model")
  model <- glmnet(x, y, alpha = 1, family = "binomial",
                  lambda = cv.lasso$lambda.1se)
  
  # return final model
  return(model)
}



# fit model
mod.lasso <- fitLasso(train)

# Display regression coefficients
mod.coefs <- coef(mod.lasso)

# get predictions
pred <- getLassoProbs(mod.lasso, test)

# get accuracy
calcAccuracyLasso(mod.lasso, test, pred=pred)

# ROC
getROC(mod.lasso, test, pred)

# try again with data balancing
train2 <- caret::upSample(train, train$label) %>% as_tibble() %>% select(-Class)

# fit model
mod.lasso2 <- fitLasso(train2)

# Display regression coefficients
mod.coefs2 <- coef(mod.lasso2)

# get predictions
pred2 <- getLassoProbs(mod.lasso2, test)

# ROC
getROC(mod.lasso2, test, pred2)

# get accuracy
calcAccuracyLasso(mod.lasso2, test, pred=pred2, cutoff=0.5)

# grab reduced variables
vars_kept <- names(mod.coefs2[,1][which(abs(mod.coefs2[,1]) > 0)])[-1]

### refit to get p-values
train3 <- dplyr::select(train2, ID, label, vars_kept)
test2 <- dplyr::select(test, ID, label, vars_kept)

# fit model
mod3 <- glm(label ~ ., data = train3[,-1], family="binomial")
summary(mod3)

# get pvals
mod.coefs3 <- coef(mod3)
mod.pvals3 <- coef(summary(mod3))[,4]

# get predictions
pred3 <- predict(mod3, test2, type="response")

# ROC
getROC(mod3, test2, pred3)

# get accuracy
calcAccuracyLasso(mod3, test2, pred=pred3, cutoff=0.5)


############## GET IMPORTANT VARIABLES ############## 
# variable importance
var.imp <- filterVarImp(train3[,-c(1,2)], unlist(train3[,2]))
var.imp <- tibble(
  var = row.names(var.imp),
  varimp = var.imp[,1]
)

# get coefficients
coefs <- tibble(
  var = names(mod.coefs3),
  coef = mod.coefs3, 
  pval = mod.pvals3
) %>%
  mutate(
    pval_group = case_when(
      pval <= 0.001 ~ "< 0.001",
      pval <= 0.01 ~ "< 0.01",
      pval <= 0.05 ~ "< 0.05",
      pval <= 0.1 ~ "< 0.1",
      TRUE ~ "> 0.1"
    )
  )


# merge together
coefs_merge <- left_join(coefs,var.imp) %>%
  filter(var != "(Intercept)")

# plot
coefs_merge %>%
  filter(pval_group != "> 0.1") %>%
  ggplot(aes(x=coef,y=varimp)) +
  geom_point(aes(shape=pval_group)) +
  geom_vline(xintercept = 0, linetype="longdash", alpha=0.5) + 
  geom_text_repel(aes(label = var)) +
  theme_minimal() +
  labs(
    x = "Coefficient",
    y = "Variable Importance",
    title = "Most important and impactful variables for fake news classification",
    caption = "All variables rescaled to mean 0 and SD 1",
    shape = "p-value group"
  ) +
  theme(
    legend.position = "bottom",
    text = element_text(family = "Roboto")
  )
ggsave("thesis/final_draft/figures/body.png",dpi=600)

############## TITLE MODELS ############## 
fitLassoTitles <- function(train_data, seed = 13){
  # Dumy code categorical predictor variables
  x <- model.matrix(train_data[,-1])
  
  # Convert the outcome (class) to a numerical variable
  y <- train_data$label
  
  # Find the best lambda using cross-validation
  set.seed(seed) 
  print("Starting cross validation")
  cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
  
  #visualize
  plot(cv.lasso)
  cv.lasso$lambda.min
  cv.lasso$lambda.1se
  
  # Fit the final model on the training data
  print("Fitting final model")
  model <- glmnet(x, y, alpha = 1, family = "binomial",
                  lambda = cv.lasso$lambda.1se)
  
  # return final model
  return(model)
}

# drop ID
train_titles2 <- train_titles %>% drop_na()
test_titles2 <- test_titles %>% drop_na()


# try again with data balancing
train_titles3 <- caret::upSample(train_titles2, train_titles2$label) %>% as_tibble() %>% select(-Class)

# fit model
mod.lasso.titles <- fitLasso(train_titles3)

# Display regression coefficients
mod.coefs.titles <- coef(mod.lasso.titles)

# get predictions
pred.titles <- getLassoProbs(mod.lasso.titles, test_titles2)

# ROC
getROC(mod.lasso.titles, test_titles2, pred.titles)

# get accuracy
calcAccuracyLasso(mod.lasso.titles, test_titles2, pred=pred.titles, cutoff=0.5)

# grab reduced variables
vars_kept <- names(mod.coefs.titles[,1][which(abs(mod.coefs.titles[,1]) > 0)])[-1]
vars_kept[39] <- "PRP$"

### refit to get p-values
train_titles4 <- dplyr::select(train_titles3, label, vars_kept)
test_titles3 <- dplyr::select(test_titles2, label, vars_kept)

# fit model
mod.titles2 <- glm(label ~ ., data = train_titles4, family="binomial")
summary(mod.titles2)

# get pvals
mod.coefs.titles2 <- coef(mod.titles2)
mod.pvals.titles2 <- coef(summary(mod.titles2))[,4]

# get predictions
pred.titles2 <- predict(mod.titles2, test_titles3, type="response")

# ROC
getROC(mod.titles2, test_titles3, pred.titles2)

# get accuracy
calcAccuracyLasso(mod.titles2, test_titles3, pred=pred.titles2, cutoff=0.3)

############## GET IMPORTANT VARIABLES ############## 
# variable importance
var.imp.titles <- filterVarImp(train_titles4[,-1], unlist(train_titles4[,1]))
var.imp.titles <- tibble(
  var = row.names(var.imp.titles),
  varimp = var.imp.titles[,1]
)

# get coefficients
coefs.titles <- tibble(
  var = names(mod.coefs.titles2),
  coef = mod.coefs.titles2, 
  pval = mod.pvals.titles2
) %>%
  mutate(
    pval_group = case_when(
      pval <= 0.001 ~ "< 0.001",
      pval <= 0.01 ~ "< 0.01",
      pval <= 0.05 ~ "< 0.05",
      pval <= 0.1 ~ "< 0.1",
      TRUE ~ "> 0.1"
    )
  )


# merge together
coefs_merge.titles <- left_join(coefs.titles,var.imp.titles) %>%
  filter(var != "(Intercept)")

# plot
coefs_merge.titles %>%
  filter(pval_group != "> 0.1") %>%
  ggplot(aes(x=coef,y=varimp)) +
  geom_point(aes(shape=pval_group)) +
  geom_vline(xintercept = 0, linetype="longdash", alpha=0.5) + 
  geom_text_repel(aes(label = var)) +
  theme_minimal() +
  labs(
    x = "Coefficient",
    y = "Variable Importance",
    title = "Most important and impactful variables for fake news classification",
    subtitle = "Features describing titles of real and fake articles",
    caption = "All variables rescaled to mean 0 and SD 1",
    shape = "p-value group"
  ) +
  theme(
    legend.position = "bottom",
    text = element_text(family = "Roboto")
  )
ggsave("thesis/final_draft/figures/titles.png")

########### SETUP TABLES #############
complexity_table <- tibble(
  var = c("mu_sentence", "mu_verb_phrase", "mu_noun_phrase", "sd_sentence", "sd_verb_phrase", "sd_noun_phrase", "iqr_sentence", "iqr_verb_phrase",
          "iqr_noun_phrase","num_verb_phrase","swc","wlen","types","tokens","TTR","FOG","SMOG","FK","CL","ARI"),
  description = c("Mean number of sentences","Mean depth of verb-phrase trees",
                  "Mean depth of noun-phrase trees",
                  "Standard deviation of number of sentences",
                  "Standard deviation of depth of verb-phrase trees",
                  "Standard deviation of depth of noun-phrase trees",
                  "Interquantile range of number of sentences",
                  "Interquantile range of depth of verb-phrase trees",
                  "Interquantile range of depth of verb-phrase trees",
                  "Number of verb-phrase trees",
                  "Mean sentence word count",
                  "Mean word length",
                  "Number of unique words",
                  "number of total words",
                  "Type-token ration",
                  "Gunning's Fog Index",
                  "Simple Measure of Gobbledygook",
                  "Flesch-Kincaid Readability Score",
                  "Coleman-Liau Index",
                  "Automated Readability Index")
  
)

names(train_LIWC)

liwc_table <- tibble(
  var = c("WC","Analytic","Clout","Authentic","Tone","WPS","Sixltr","Dic","function","pronoun","ppron","i","we","you","shehe","they",
          "ipron","article","prep","auxverb","adverb","conj","negate","verb","adj","compare","interrog","number","quant","affect","posemo","negemo",
          "anx","anger","sad","social","family","friend","female","male","cogproc","insight","cause","discrep","tentat","certain","differ","percept",
          "see","hear","feel","bio","body","health","sexual","ingest","drives","affiliation","achieve","power","reward","risk","focuspast",
          "focuspresent","focusfuture","relativ","motion","space","time","work","leisure","home","money","relig","death","informal","swear","netspeak",
          "assent","nonflu","filler","AllPunc","Period","Comma","Colon","SemiC","QMark","Exclam","Dash","Quote","Apostro","Parenth","OtherP"),
  description = c("Word count","Words reflecting formal, logical, and hierarchical thinking",
                  "Words suggesting author is speaking from a position of authority",
                  "Words associated with a more honest, personal, and disclosing text",
                  "Words associated with positive, upbeat style",
                  "Words per sentence",
                  "Number of six+ letter words",
                  "unsure",
                  "Function words",
                  "Pronouns",
                  "Personal pronouns",
                  "1st person singular",
                  "1st person plural",
                  "2nd person",
                  "3rd person singular",
                  "3rd person plural",
                  "Impersonal pronoun",
                  "Articles",
                  "Prepositions",
                  "Auxiliary verbs",
                  "Common adverbs",
                  "Conjuctions",
                  "Negations",
                  "Regular verbs",
                  "Adjectives",
                  "Comparatives",
                  "Interrogatives",
                  "Numbers",
                  "Quantifiers",
                  "Affect words",
                  "Positive emotions",
                  "Negative emotions",
                  "Anxiety",
                  "Anger",
                  "Sad",
                  "Social words",
                  "Family",
                  "Friends",
                  "Female referents",
                  "Male referents",
                  "Cognitive processes",
                  "Insight",
                  "Cause",
                  "Discrepancies",
                  "Tentativeness",
                  "Certainty",
                  "Differentiation",
                  "Perceptual processes",
                  "Seeing",
                  "Hearing",
                  "Feeling",
                  "Biological processes",
                  "Body",
                  "Health/illness",
                  "Sexuality",
                  "Ingesting",
                  "Core drives",
                  "Affiliation",
                  "Achievement",
                  "Power",
                  "Reward focus",
                  "Risk/prevention focus",
                  "Past focus",
                  "Present focus",
                  "Future focus",
                  "Relativity",
                  "Motion",
                  "Space",
                  "Time",
                  "Work",
                  "Leisure",
                  "Home",
                  "Money",
                  "Religion",
                  "Death",
                  "Informal speech",
                  "Swear words",
                  "Netspeak",
                  "Assent",
                  "Nonfluencies",
                  "Fillers",
                  "All punctuation",
                  "Periods",
                  "Commas",
                  "Colons",
                  "Semicolons",
                  "Question marks",
                  "Exclamation marks",
                  "Dashes",
                  "Quotes",
                  "Apostrophes",
                  "Parentheses (pairs)",
                  "Other punctuation"))

POS_table <- tibble(
  var = c("CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS",
          "MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS",
          "RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"),
  description = c("Coordinating conjunctions",
                  "Cardinal numeral",
                  "Determiner",
                  "Existential",
                  "Foreign word",
                  "Preposition or subordinating conjunction",
                  "Ordinal number",
                  "Comparative adjective",
                  "Superlative adjective",
                  "List item marker",
                  "Model verb",
                  "Noun, singular or mass",
                  "Plural noun",
                  "Singular proper noun",
                  "Plural proper noun",
                  "Predeterminer",
                  "Possessive ending",
                  "Personal pronoun",
                  "Possessive pronoun",
                  "Adverb",
                  "Comparative adverb",
                  "Superlative adverb",
                  "Particle",
                  "Symbol",
                  "To",
                  "Exclamation/interjection",
                  "Verb, base form",
                  "Past tense verb",
                  "Present participle",
                  "Past participle",
                  "Present tense verb, other than 3rd person singular",
                  "Present tense verb, 3rd person singular",
                  "Wh-determiner",
                  "Wh-pronoun",
                  "Possessive wh-pronoun",
                  "Wh-adverb"))


################# MEDIAN TEST TABLES
test_table <- tibble(
  var = names(train[,-c(1,2)]),
  med_false = rep(NA,ncol(train) - 2),
  med_true = rep(NA,ncol(train) - 2),
  pval = rep(NA,ncol(train) - 2)
)

for (idx in seq(3,ncol(train))) {
  test_temp<-mood.medtest(unlist(train[,idx],use.names = FALSE) ~ train$label,
                          exact = FALSE)
  meds <- train[,c(2,idx)] %>%
    group_by(label) %>%
    summarize_all(median)
  
  test_table[idx-2,]$med_false <- pull(filter(meds,label==0)[,2])
  test_table[idx-2,]$med_true <- pull(filter(meds,label==1)[,2])
  
  test_table[idx-2,]$pval <- test_temp$p.value
}



tests_sig <- filter(test_table, pval <= 0.05, med_false != med_true) %>%
  mutate(Result = if_else(med_false > med_true, "Fake > Real", "Real > Fake")) %>%
  mutate(
    pval_group = case_when(
      pval <= 0.001 ~ "< 0.001",
      pval <= 0.01 ~ "< 0.01",
      pval <= 0.05 ~ "< 0.05",
      pval <= 0.1 ~ "< 0.1",
      TRUE ~ "> 0.1"
    )
  )



tests_sig <- tests_sig %>%
  mutate(`Gruppi et al.` = "-",
         `Horne et al.` = "-")
kable(dplyr::select(tests_sig,-med_false,-med_true,-pval), "latex", longtable = T, booktabs = T, caption = "Longtable")

################# MEDIAN TEST TABLES - TITLES
test_table <- tibble(
  var = names(train_titles[,-c(1,2)]),
  med_false = rep(NA,ncol(train_titles) - 2),
  med_true = rep(NA,ncol(train_titles) - 2),
  pval = rep(NA,ncol(train_titles) - 2)
)


for (idx in seq(3,ncol(train_titles))) {
  test_temp<-mood.medtest(unlist(train_titles[,idx],use.names = FALSE) ~ train_titles$label,
                          exact = FALSE)
  meds <- train_titles[,c(2,idx)] %>%
    group_by(label) %>%
    summarize_all(median)
  
  test_table[idx-2,]$med_false <- pull(filter(meds,label==0)[,2])
  test_table[idx-2,]$med_true <- pull(filter(meds,label==1)[,2])
  
  test_table[idx-2,]$pval <- test_temp$p.value
}



tests_sig <- filter(test_table, pval <= 0.05, med_false != med_true) %>%
  mutate(Result = if_else(med_false > med_true, "Fake > Real", "Real > Fake")) %>%
  mutate(
    pval_group = case_when(
      pval <= 0.001 ~ "< 0.001",
      TRUE ~ as.character(round(pval,4))
    )
  )

tests_sig <- tests_sig %>%
  mutate(`Gruppi et al.` = "-",
         `Horne et al.` = "-")
kable(dplyr::select(tests_sig,-med_false,-med_true,-pval), "latex", longtable = T, booktabs = T, caption = "Longtable")



