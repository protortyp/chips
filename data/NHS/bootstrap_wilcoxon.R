# bootstrap_wilcoxon.R
#
# Calculate a bootstrap wilcoxon signed rank test for two AUC values
#
# 2024.05.13 Eliana Marostica

library(tidyverse)
library(ROCR)

setwd("~/Library/CloudStorage/GoogleDrive-elianama@mit.edu/My Drive/Research/YuLab/Projects/SambaNova/HMS_artifacts_folder/NHS")

#-----------------
# First, load in predictions and true labels

# Load in comma-separated file for 512 ("standard") results
standard <- read_delim("512_run/test_log_NHS_patch_label_per_patient_changed_512_all_9_patches_with_reassigned_pat_id/logs__rank_0.txt",
           delim=",",
           col_names = c("mode", "step", "pid", "img_path", "true", "predicted"),
           skip=4,
           skip_empty_rows = T) %>%
  separate(pid, ": ", into=c(NA, "pid"), remove=T) %>%
  separate(img_path, ": ", into=c(NA, "img_path"), remove=T) %>%
  separate(true, ": ", into=c(NA, "true"), remove=T) %>%
  separate(predicted, ": ", into=c(NA, "predicted"), remove=T) %>%
  mutate(true = as.numeric(true),
         predicted = as.numeric(predicted)) %>%
  select(pid, true, predicted) %>%
  group_by(pid) %>%
  summarize(true = round(median(true)),
            predicted = round(median(predicted)))

# Load in semicolon-delimited file for chips results
chips <- read_delim("1440_runs/test_log_nhs_1440_reassigned_pat_id/logs__rank_0.txt",
                       delim=";",
                       col_names = c("mode", "step", "pid", "img_path", "true", "predicted"),
                       skip=4,
                       skip_empty_rows = T) %>%
  separate(pid, ": ", into=c(NA, "pid"), remove=T) %>%
  separate(img_path, ": ", into=c(NA, "img_path"), remove=T) %>%
  separate(true, ": ", into=c(NA, "true"), remove=T) %>%
  separate(predicted, ": ", into=c(NA, "predicted"), remove=T) %>%
  mutate(true = as.numeric(true),
         predicted = as.numeric(predicted)) %>%
  select(pid, true, predicted) %>%
  group_by(pid) %>%
  summarize(true = round(median(true)),
            predicted = round(median(predicted)))
  
head(standard)
head(chips)

# Check that the results are in the same order
sum(standard$pid != chips$pid) == 0 


#-----------------
# Try DeLong test

library(pROC)
standard.pred <- standard$predicted
standard.true <- standard$true
chips.pred <- chips$predicted
chips.true <- chips$true

response<- standard.true
modela <- standard.pred
modelb <- chips.pred
roca <- roc(response,modela)
rocb<-roc(response,modelb)

roc.test(roca,rocb,method=c("delong"))




#-----------------
# Generate a single bootstrap sample for standard and chips

n_bootstraps = 1000

# vectors to accumulate aucs
standard.aucs <- c()
chips.aucs <- c()

# reformat vectors of predicted and true values for standard and chips
standard.pred <- standard$predicted
standard.true <- standard$true
chips.pred <- chips$predicted
chips.true <- chips$true

for(i in 1:n_bootstraps){
  # sample with replacement from empirical distribution of predictions and corresponding true values
  sample.inds <- sample(1:length(standard.pred), length(standard.pred), replace = T)

  standard.pred.sample <- standard.pred[sample.inds]
  standard.true.sample <- standard.true[sample.inds]
  chips.pred.sample <- chips.pred[sample.inds]
  chips.true.sample <- chips.true[sample.inds]
  
  for (method in c("standard", "chips")) {
    if(method == "standard"){
      pred.sample <- standard.pred.sample
      true.sample <- standard.true.sample
    } else{
      pred.sample <- chips.pred.sample
      true.sample <- chips.true.sample
    }
    
    # accumulate AUROCs for chips
    tryCatch(
      { # calculate AUROC of sample
        pred.obj <- prediction(pred.sample, true.sample)
        tmp.auc <- max(performance(pred.obj, measure="auc")@y.values[[1]])
        if(method == "standard"){
          standard.aucs <- c(standard.aucs, tmp.auc)
        } else{
          chips.aucs <- c(chips.aucs, tmp.auc)

        }
      },
      error=function(cond){
        message(paste("AUROC calculation for bootstrap", i, "failed."))
        message("Original error message:")
        message(cond)
        message("Continuing bootstrap calculation.")
      }
    )
    
  }
}

standard.aucs
chips.aucs

library("car")
qqPlot(standard.aucs)
qqPlot(chips.aucs)

mean(standard.aucs)
mean(chips.aucs)

t.test(x=standard.aucs, y=chips.aucs, mu=0, paired=T)


#-----------------
# Bootstrap P-value Approach to Comparing Two AUROCs
# Based on the following video: https://www.youtube.com/watch?v=9STZ7MxkNVg&ab_channel=MarinStatsLectures-RProgramming%26Statistics


library(pROC)
n_bootstraps = 1000

test_statistics <- c()

# reformat vectors of predicted and true values for standard and chips
standard.pred <- standard$predicted
standard.true <- standard$true
chips.pred <- chips$predicted
chips.true <- chips$true

null <- tibble(ground_truth = standard.true,
       pred = standard.pred,
       chips = 0) %>%
  add_row(ground_truth = chips.true,
          pred = chips.pred,
          chips = 1) 

pred.obj.s <- prediction(null$pred[null$chips==0], null$ground_truth[null$chips==0])
pred.obj.c <- prediction(null$pred[null$chips==1], null$ground_truth[null$chips==1])
auc.s <- max(performance(pred.obj.s, measure="auc")@y.values[[1]])
auc.c <- max(performance(pred.obj.c, measure="auc")@y.values[[1]])
empirical.difference <- abs(auc.c-auc.s)

for(i in 1:n_bootstraps){
  sample.inds <- sample(1:nrow(null), nrow(null), replace = T)
  bootstrap.sample <- null[sample.inds,] %>%
    mutate(chips = sample(chips))
  bootstrap.sample.standard <- bootstrap.sample[bootstrap.sample$chips==0,]
  bootstrap.sample.chips <- bootstrap.sample[bootstrap.sample$chips==1,]
  
  # calculate AUROCs of sample
  pred.obj.standard <- prediction(bootstrap.sample.standard$pred, bootstrap.sample.standard$ground_truth)
  auc.bootstrap.standard <- max(performance(pred.obj.standard, measure="auc")@y.values[[1]])
  pred.obj.chips <- prediction(bootstrap.sample.chips$pred, bootstrap.sample.chips$ground_truth)
  auc.bootstrap.chips <-  max(performance(pred.obj.chips, measure="auc")@y.values[[1]])
  
  test_stat <- abs(auc.bootstrap.chips-auc.bootstrap.standard)
  test_statistics <- c(test_statistics, test_stat)
}


p.value <- sum(test_statistics >= empirical.difference) / n_bootstraps
p.value

#-----------------
# Calculate Wilcoxon rank sum test

# R program to illustrate
# Paired Samples Wilcoxon Test

# The data set
before <- standard.aucs
after <- chips.aucs

# Create a data frame
myData <- data.frame(
  group = rep(c("before", "after"), each = length(before)),
  weight = c(before, after)
)

# Print all data
print(myData)

# Paired Samples Wilcoxon Test
result = wilcox.test(before, after, paired = TRUE, exact=TRUE, correct=TRUE)

# Printing the results
print(result)
