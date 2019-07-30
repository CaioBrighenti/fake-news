library(caret)
############## HELPER FUNCTIONS ############## 
calcAccuracyLR <- function(mod,new_data,adj=0, true_labels = NULL) {
  if (is.null(true_labels)){
    true_labels <- new_data$`_label`
    if (is.null(true_labels)){true_labels <- new_data$label}
  }
  pred <- predict(mod, new_data)
  class <- as.numeric(pred > 0.5)
  dist <- abs(as.numeric(class)-(as.numeric(true_labels)-1))
  acc <-  mean(dist <= adj)
  ## [1] = sensitivity/recall, [2] = specificity, [5] = precision, [7] = F1
  cm <- confusionMatrix(as.factor(class), true_labels, positive="1")
  # calculate F1
  stats <- tibble(accuracy = acc, sensitivity = cm$byClass[1],
                  specificity = cm$byClass[2], precision = cm$byClass[5],
                  F1 = cm$byClass[7], c_matrix = cm$table)
  print(stats)
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