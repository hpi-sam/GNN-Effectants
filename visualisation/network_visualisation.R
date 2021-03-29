library(tidyverse)
library(ggplot2)
library(gtsummary)

setwd("projects/HPI/GNN-Effectants/final_presentation/visualisation/")



#Average neighborhood
mumps <- read_csv("Values_Metrics/AVERAGE_NEIGHBOR_DEGREE/Mumps_average_neighbor_degree.csv", col_names = F)
carbuncle <- read_csv("Values_Metrics/AVERAGE_NEIGHBOR_DEGREE/carbuncle_average_neighbor_degree.csv", col_names = F)
coccydynia <- read_csv("Values_Metrics/AVERAGE_NEIGHBOR_DEGREE/coccydynia_average_neighbor_degree.csv", col_names = F)
Bleeding <- read_csv("Values_Metrics/AVERAGE_NEIGHBOR_DEGREE/Bleeding_average_neighbor_degree.csv", col_names = F)
temperature <- read_csv("Values_Metrics/AVERAGE_NEIGHBOR_DEGREE/body temperature increased_average_neighbor_degree.csv", col_names = F)
emesis <- read_csv("Values_Metrics/AVERAGE_NEIGHBOR_DEGREE/emesis_average_neighbor_degree.csv", col_names = F)

mean(carbuncle$X1)

ggplot(data=Bleeding, aes(X1)) + 
  geom_histogram(bins = 50)  +
  scale_y_continuous(breaks=seq(5, 55, 5), limits = c(0, 55.5),expand = c(0,0)) +
  theme_bw() + 
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=22,face="bold"),
        legend.text = element_text(size = 20),
        legend.title = element_text(size=22,face="bold"),
        legend.position="top") + 
  labs(x = "Neighbors")


#Eigenvector centrality
mumps <- read_csv("Values_Metrics/EIGENVECTOR_CENTRALITY/Mumps_eigenvector_centrality.csv", col_names = F)
carbuncle <- read_csv("Values_Metrics/EIGENVECTOR_CENTRALITY/carbuncle_eigenvector_centrality.csv", col_names = F)
coccydynia <- read_csv("Values_Metrics/EIGENVECTOR_CENTRALITY/coccydynia_eigenvector_centrality.csv", col_names = F)
Bleeding <- read_csv("Values_Metrics/EIGENVECTOR_CENTRALITY/Bleeding_eigenvector_centrality.csv", col_names = F)
temperature <- read_csv("Values_Metrics/EIGENVECTOR_CENTRALITY/body temperature increased_eigenvector_centrality.csv", col_names = F)
emesis <- read_csv("Values_Metrics/EIGENVECTOR_CENTRALITY/emesis_eigenvector_centrality.csv", col_names = F)

ggplot(data=Bleeding, aes(X1)) + 
  geom_histogram(bins = 50)  +
  scale_y_continuous(breaks=seq(5, 50, 5), limits = c(0, 50.5),expand = c(0,0)) +
  theme_bw() + 
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=22,face="bold"),
        legend.text = element_text(size = 20),
        legend.title = element_text(size=22,face="bold"),
        legend.position="top") + 
  labs(x = "Eigenvector Centrality")
