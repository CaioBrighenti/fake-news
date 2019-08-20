library(caret)
library(progress)
############## HELPER FUNCTIONS ############## 
calcAccuracyLR <- function(mod,new_data,adj=0, true_labels = NULL, cutoff = 0.5) {
  if (is.null(true_labels)){
    if('_label' %in% names(new_data)) {
      true_labels <- new_data$`_label`
    } else {
      true_labels <- new_data$label
    }
  }
  levels(true_labels) <- c(0,1)
  pred <- predict(mod, new_data, type = "response")
  class <- factor(as.numeric(pred > cutoff), levels=c(0,1))
  acc <-  mean(class == true_labels)
  ## [1] = sensitivity/recall, [2] = specificity, [5] = precision, [7] = F1
  cm <- confusionMatrix(table(class, true_labels), positive="1")
  # calculate F1
  stats <- tibble(accuracy = round(acc,3), sensitivity = round(cm$byClass[1],3),
                  specificity = round(cm$byClass[2],3), precision = round(cm$byClass[5],3),
                  F1 = round(cm$byClass[7],3), c_matrix = cm$table)
  stats[2,-c(6,7)] <- '-'
  #print(stats)
  return(stats)
}

calcAccuracy <- function(mod,new_data,adj=0) {
  true_labels <- new_data$label
  pred <- predict(mod, newdata = new_data, type="class")
  dist <- abs(as.numeric(pred)-(as.numeric(true_labels)))
  acc <-  mean(dist <= adj)
  ## [1] = sensitivity/recall, [2] = specificity, [5] = precision, [7] = F1
  cm <- confusionMatrix(as.factor(pred), true_labels, positive="1")
  # calculate F1
  stats <- tibble(accuracy = acc, sensitivity = cm$byClass[1],
                  specificity = cm$byClass[2], precision = cm$byClass[5],
                  F1 = cm$byClass[7], c_matrix = cm$table)
  print(stats)
  return(stats)
}

plotPredictions <- function(mods,dat_test){
  par(mfrow=c(1,length(mods)+1))
  plot(test$label)
  for (mod in mods) {
    plot(predict(mod, newdata = dat_test))
  }
}

getVarRanks <- function(data) {
  ## get rid of unnecessary columns
  if('title' %in% names(data)){data <- data %>% dplyr::select(-title)}
  if('text' %in% names(data)){data <- data %>% dplyr::select(-text)}
  
  aovs <- data %>%
    dplyr::select(-ID, -label) %>%
    map(~ as.numeric(unlist(summary(aov(.x ~ data$label)))['Pr(>F)1'])) %>%
    as_tibble() %>%
    gather(var, p_val) %>%
    arrange(p_val)
  
  ranks <- data %>%
    dplyr::select(-ID, -label) %>%
    filterVarImp(.,data$label) %>%
    mutate(var = row.names(.), max_varimp = pmax(fake,real), varimp_rank = rank(-max_varimp)) %>%
    dplyr::select(var, max_varimp, varimp_rank) %>%
    left_join(aovs, by = 'var') %>%
    mutate(p_rank = rank(p_val), avg_rank = (varimp_rank + p_rank) / 2) %>%
    arrange(avg_rank)
  
  return(ranks)
}

getROC <- function(mod, data){
  colgate_ter <- c("#64A50A", "#F0AA00","#0096C8", "#005F46","#FF6914","#004682")
  roc_tib <- tibble(
    cutoff = seq(0,1,by=0.01),
    sensitivity = rep(0,length(cutoff)),
    specificity = rep(0,length(cutoff)),
    accuracy = rep(0,length(cutoff))
  )
  pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = nrow(roc_tib))
  for (idx in 1:nrow(roc_tib)) {
    pb$tick()
    cutoff <- roc_tib[idx,]$cutoff
    acc_table <- calcAccuracyLR(mod, data, cutoff = cutoff)
    roc_tib[idx,]$sensitivity <- as.numeric(acc_table$sensitivity[1])
    roc_tib[idx,]$specificity <- as.numeric(acc_table$specificity[1])
    roc_tib[idx,]$accuracy <- as.numeric(acc_table$accuracy[1])
  }
  p<-roc_tib %>%
    ggplot(aes(x=1-specificity, y=sensitivity, color=cutoff, size = accuracy)) +
    geom_point() +
    scale_color_gradient(low=colgate_ter[2],high=colgate_ter[3])
  print(p)
  
  best_cut <- roc_tib %>%
    mutate(crit1 = sensitivity + specificity, crit2 = abs(specificity - sensitivity)) %>%
    mutate(acc_rank = rank(-accuracy,ties.method="min"),
           crit1 = rank(-crit1,ties.method="min"),
           crit2 = rank(crit2,ties.method="min")) %>%
    filter(acc_rank == 1 | crit1 == 1 | crit2 == 1)
  
  return(best_cut)
}    
