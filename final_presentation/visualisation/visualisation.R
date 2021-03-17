library(tidyverse)
library(ggplot2)
library(gtsummary)

setwd("projects/HPI/GNN-Effectants/final_presentation/visualisation/")
data <- read_csv("GNN_Statistics.csv")


summarised_data <- data %>%
  select(-Run) %>%
  group_by(Algorithm, `Side-Effect`) %>%
  summarise(min_loss = min(Loss, na.rm=T),
            mean_loss = mean(Loss, na.rm=T),
            median_loss = median(Loss, na.rm=T),
            max_loss = max(Loss, na.rm=T),
            min_acc = min(acc, na.rm=T),
            mean_acc = mean(acc, na.rm=T),
            median_acc = median(acc, na.rm=T),
            max_acc = max(acc, na.rm=T),
            min_roc_auc = min(roc_auc, na.rm=T),
            mean_roc_auc = mean(roc_auc, na.rm=T),
            median_roc_auc = median(roc_auc, na.rm=T),
            max_roc_auc = max(roc_auc, na.rm=T),
            min_pr_auc = min(pr_auc, na.rm=T),
            mean_pr_auc = mean(pr_auc, na.rm=T),
            median_pr_auc = median(pr_auc, na.rm=T),
            max_pr_auc = max(pr_auc, na.rm=T),
            min_timec = min(`time in s`, na.rm=T),
            mean_time = mean(`time in s`, na.rm=T),
            median_time = median(`time in s`, na.rm=T),
            max_pr_time = max(`time in s`, na.rm=T)
  )

data_mumps <- summarised_data %>%
  filter(`Side-Effect` == "Mumps")

data_mumps_nodecagon <- summarised_data %>%
  filter(`Side-Effect` == "Mumps") %>%
  filter(Algorithm != "Decagon")

ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_acc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="Accuracy", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_roc_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="ROC-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_mumps, aes(x=Algorithm, y=mean_pr_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="PR-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_time)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="Time (s)")


data_bleeding <- summarised_data %>%
  filter(`Side-Effect` == "Bleeding")

data_bleeding_nodecagon <- summarised_data %>%
  filter(`Side-Effect` == "Bleeding") %>%
  filter(Algorithm != "Decagon")


ggplot(data_bleeding_nodecagon, aes(x=Algorithm, y=mean_acc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="Accuracy", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_bleeding_nodecagon, aes(x=Algorithm, y=mean_roc_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="ROC-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_bleeding, aes(x=Algorithm, y=mean_pr_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="PR-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_bleeding_nodecagon, aes(x=Algorithm, y=mean_time)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="Time (s)")
