library(tidyverse)
library(ggplot2)
library(gtsummary)

setwd("projects/HPI/GNN-Effectants/final_presentation/visualisation/")


data <- read_csv("GNN_Statistics.csv")

data  %>% tbl_summary(by = Algorithm)

# Generate summary for different algorithms

