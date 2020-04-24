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


# load fonts
#font_import(paths = "C:/Users/caiob/Downloads/Roboto")
loadfonts(device = "win", quiet = TRUE)

# load helpers
source("helpers/loaddata.R")
source("helpers/helpers.R")

# load in articles
train_articles <- loadFNNTrain()
test_articles <- loadFNNTest()

# get lengths
train_articles$title_length <- sapply(train_articles$title, str_length)
train_articles$body_length <- sapply(train_articles$text, str_length)
test_articles$title_length <- sapply(test_articles$title, str_length)
test_articles$body_length <- sapply(test_articles$text, str_length)

# load in text features
train <- loadFNNtxtfeat("train") %>% filter(!grepl("gossipcop",ID)) %>%
  rename(function. = `function`) %>%
  left_join(
    dplyr::select(train_articles, ID, body_length),
    by = c("ID")
  ) %>%
  rename(len  = body_length)
test <- loadFNNtxtfeat("test") %>% filter(!grepl("gossipcop",ID)) %>%
  rename(function. = `function`) %>%
  left_join(
    dplyr::select(test_articles, ID, body_length),
    by = c("ID")
  ) %>%
  rename(len  = body_length)
train_titles <- loadFNNtxtfeat("train_titles") %>% filter(!grepl("gossipcop",ID)) %>%
  rename(function. = `function`) %>%
  left_join(
    dplyr::select(train_articles, ID, title_length),
    by = c("ID")
  ) %>%
  rename(len  = title_length)
test_titles <- loadFNNtxtfeat("test_titles") %>% filter(!grepl("gossipcop",ID)) %>%
  rename(function. = `function`) %>%
  left_join(
    dplyr::select(test_articles, ID, title_length),
    by = c("ID")
  ) %>%
  rename(len  = title_length)

# individual parts
train_complexity <- loadFNNComplexity("train") %>% filter(!grepl("gossipcop",ID))
train_LIWC <- loadFNNLIWC("train") %>% filter(!grepl("gossipcop",ID))
train_POS <- loadFNNPOS("train") %>% filter(!grepl("gossipcop",ID))
train_NER <- loadFNNNER("train") %>% filter(!grepl("gossipcop",ID))

# LWIC groups
LWIC_groups <- loadLIWCGroups()

############## PREPROCCESS ##############
train[,c(117,152)] <- train[,c(117,152)] / train$len * 100
test[,c(117,152)] <- test[,c(117,152)] / test$len * 100
train_titles[,c(117,152)] <- train_titles[,c(117,152)] / train_titles$len * 100
test_titles[,c(117,152)] <- test_titles[,c(117,152)] / test_titles$len * 100

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
calcAccuracyLasso(mod3, test2, pred=pred3, cutoff=0.2)


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


############## TITLE MODELS ############## 
fitLassoTitles <- function(train_data, seed = 13){
  # Dumy code categorical predictor variables
  x <- as.matrix(train_data[,-1])
  
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
train_titles2 <- train_titles %>% dplyr::select(-ID, -LS, -PDT, -swear, -assent, -nonflu, -filler, -LS, -RBS, -UH, -`WP$`) %>% drop_na()
test_titles2 <- test_titles %>% dplyr::select(-ID, -LS, -PDT, -swear, -assent, -nonflu, -filler, -LS, -RBS, -UH, -`WP$`) %>% drop_na()


# try again with data balancing
train_titles3 <- caret::upSample(train_titles2, train_titles2$label) %>% as_tibble() %>% select(-Class)

# fit model
mod.lasso.titles <- fitLassoTitles(train_titles3)

# Display regression coefficients
mod.coefs.titles <- coef(mod.lasso.titles)

# get predictions
pred.titles <- getLassoProbsTitles(mod.lasso.titles, test_titles2)

# ROC
getROC(mod.lasso.titles, test_titles2, pred.titles)

# get accuracy
calcAccuracyLasso(mod.lasso.titles, test_titles2, pred=pred.titles, cutoff=0.5)

# grab reduced variables
vars_kept <- names(mod.coefs.titles[,1][which(abs(mod.coefs.titles[,1]) > 0)])[-1]

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
calcAccuracyLasso(mod.titles2, test_titles3, pred=pred.titles2, cutoff=0.365)

############## GET IMPORTANT VARIABLES ############## 
# variable importance
var.imp.titles <- filterVarImp(train_titles4[,-1], unlist(train_titles4[,1]))
var.imp.titles <- tibble(
  var = row.names(var.imp.titles),
  varimp = var.imp.titles[,1]
)

# get coefficients
coefs.titles <- tibble(
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
coefs_merge.titles <- left_join(coefs.titles,var.imp.titles) %>%
  filter(var != "(Intercept)")

# plot
coefs_merge.titles %>%
  filter(pval_group != "> 0.1") %>%
  ggplot(aes(x=coef,y=varimp)) +
  geom_point(aes(shape=pval_group)) +
  xlim(-1.5,1.5) + 
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


gruppi_table <- tibble(
  var = c("SMOG","FK","WC","WPS","shehe","auxverb"),
  `Gruppi et al.` = c("Agree","Disagree","Disagree","Agree","Agree","Agree")
)

obrien_table <- tibble(
  var = c("wc","Quote","FK","NN","DT","shehe","TTR","mu_sentence"),
  `O'Brien et al.` = c("Disagree","Agree","Agree","Disagree","Disagree","Disagree","Disagree","Disagree")
)

tests_sig <- tests_sig %>%
  left_join(gruppi_table) %>%
  left_join(obrien_table) %>%
  mutate(
    `Gruppi et al.` = ifelse(is.na(`Gruppi et al.`),"-",`Gruppi et al.`),
    `O'Brien et al.` = ifelse(is.na(`O'Brien et al.`),"-",`O'Brien et al.`)
  )

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
      pval <= 0.01 ~ "< 0.01",
      pval <= 0.05 ~ "< 0.05",
      pval <= 0.1 ~ "< 0.1",
      TRUE ~ "> 0.1"
    )
  )


gruppi_table <- tibble(
  var = c("SMOG","FK","WC","WPS","shehe","auxverb"),
  `Gruppi et al.` = c("Agree","Disagree","Disagree","Agree","Agree","Agree")
)

obrien_table <- tibble(
  var = c("wc","Quote","FK","NN","DT","shehe","TTR","mu_sentence")
)


