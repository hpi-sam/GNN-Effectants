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

ggplot(summarised_data, aes(fill=Algorithm, x=`Side-Effect`, y=mean_pr_auc)) + 
  geom_bar(position="dodge", stat = "identity") +
  scale_y_continuous(name="PR-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1), expand = c(0,0)) + 
  theme_bw() + 
  scale_fill_brewer(palette="Paired")


t <- summarised_data %>%
  ungroup() %>%
  select(mean_time) %>%
  mutate(time_t = mean_time * 10) 

sum(t$time_t, na.rm = T)

data_mumps <- summarised_data %>%
  filter(`Side-Effect` == "Mumps")

data_mumps_nodecagon <- summarised_data %>%
  filter(`Side-Effect` == "Mumps") %>%
  filter(Algorithm != "Decagon")

#ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_acc)) + 
 # geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Accuracy", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_roc_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="ROC-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_mumps, aes(x=Algorithm, y=mean_pr_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="PR-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

#ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_time)) + 
#  geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Time (s)")


data_Carbuncle <- summarised_data %>%
  filter(`Side-Effect` == "Carbuncle")

data_Carbuncle_nodecagon <- summarised_data %>%
  filter(`Side-Effect` == "Carbuncle") %>%
  filter(Algorithm != "Decagon")


#ggplot(data_bleeding_nodecagon, aes(x=Algorithm, y=mean_acc)) + 
# geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Accuracy", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_Carbuncle_nodecagon, aes(x=Algorithm, y=mean_roc_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="ROC-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_Carbuncle, aes(x=Algorithm, y=mean_pr_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="PR-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

#ggplot(data_bleeding_nodecagon, aes(x=Algorithm, y=mean_time)) + 
# geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Time (s)")


data_Coccydynia <- summarised_data %>%
  filter(`Side-Effect` == "Coccydynia")

data_Coccydynia_nodecagon <- summarised_data %>%
  filter(`Side-Effect` == "Coccydynia") %>%
  filter(Algorithm != "Decagon")

#ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_acc)) + 
# geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Accuracy", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_Coccydynia_nodecagon, aes(x=Algorithm, y=mean_roc_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="ROC-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_Coccydynia, aes(x=Algorithm, y=mean_pr_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="PR-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

#ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_time)) + 
#  geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Time (s)")

data_Emesis <- summarised_data %>%
  filter(`Side-Effect` == "Emesis")

data_Emesis_nodecagon <- summarised_data %>%
  filter(`Side-Effect` == "Emesis") %>%
  filter(Algorithm != "Decagon")

#ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_acc)) + 
# geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Accuracy", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_Emesis_nodecagon, aes(x=Algorithm, y=mean_roc_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="ROC-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_Emesis, aes(x=Algorithm, y=mean_pr_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="PR-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

#ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_time)) + 
#  geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Time (s)")

data_temp <- summarised_data %>%
  filter(`Side-Effect` == "Increased body temp.")

data_temp_nodecagon <- summarised_data %>%
  filter(`Side-Effect` == "Increased body temp.") %>%
  filter(Algorithm != "Decagon")

#ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_acc)) + 
# geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Accuracy", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_temp_nodecagon, aes(x=Algorithm, y=mean_roc_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="ROC-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_temp, aes(x=Algorithm, y=mean_pr_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="PR-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

#ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_time)) + 
#  geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Time (s)")

data_Bleeding <- summarised_data %>%
  filter(`Side-Effect` == "Bleeding")

data_Bleeding_nodecagon <- summarised_data %>%
  filter(`Side-Effect` == "Bleeding") %>%
  filter(Algorithm != "Decagon")

#ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_acc)) + 
# geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Accuracy", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_Bleeding_nodecagon, aes(x=Algorithm, y=mean_roc_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="ROC-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

ggplot(data_Bleeding, aes(x=Algorithm, y=mean_pr_auc)) + 
  geom_bar(stat = "identity", width=0.5) +
  scale_y_continuous(name="PR-AUC", breaks=seq(0, 1, 0.05), limits = c(0, 1))

#ggplot(data_mumps_nodecagon, aes(x=Algorithm, y=mean_time)) + 
#  geom_bar(stat = "identity", width=0.5) +
# scale_y_continuous(name="Time (s)")





test <- read_csv("../pre/data/bio-decagon-combo.csv")

length(unique(c(test$`STITCH 2`, test$`STITCH 1`)))

length(test$`STITCH 1`)

networks <- read_csv("../pre/data/dataframe_top3.csv")

unique(networks$label)

single <- networks %>%
  filter(label == "Mumps")
nrow(single)
length(unique(c(single$source, single$target)))

test2 <- test %>%
  select(`STITCH 1`, `STITCH 2`) %>%
  plyr::count()

length(unique(test$`Side Effect Name`))

targets <- read_csv("../pre/data/bio-decagon-targets-all.csv")

length(unique(targets$STITCH))

testmumps <- read_csv("../SEAL/Python/data/Mumps_train copy.txt")
length(unique(c(testmumps$a, testmumps$b)))
1-4/75
