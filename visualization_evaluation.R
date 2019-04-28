# Visualization of evaluation results of Ancient Greek semantic spaces
# Author: Barbara McGillivray
# Date: 15/4/2019

# Load libraries:

library(ggplot2)
library(readxl)
library(GGally)

# Load data:

path =  file.path("/Users", "bmcgillivray", "Documents", "OneDrive", "The Alan Turing Institute", 
                  "Martina Astrid Rodda - MAR dphil project", "Evaluation", fsep = "/")
filename = "Summary_evaluation.xlsx"
ev_summ = data.frame(read_excel(paste(path, filename, sep = "/"), skip = 1, 
                     col_types = c("text", "numeric", "numeric", "text", 
                                   "numeric", "numeric", "numeric", "numeric",
                                   "numeric", "numeric"), na = c("NA")))

# Clean data:

dim(ev_summ)
head(ev_summ)
str(ev_summ)
ev_summ$lexicon = as.factor(ev_summ$lexicon)
ev_summ = ev_summ[,c(2,3,4,7,9,10)]
summary(ev_summ)
colnames(ev_summ)[4] = "Sp.corr"
ev_summ = ev_summ[complete.cases(ev_summ), ]
summary(round(ev_summ$Mean.of.precision, 2))
summary(round(ev_summ$Mean.of.recall, 2))
summary(round(ev_summ$Sp.corr, 2))

# Visualize data:

boxplot(Mean.of.precision ~ lexicon, data = ev_summ, main = "Mean precision")
boxplot(Mean.of.recall ~ lexicon, data = ev_summ, main = "Mean recall")

png(paste(path, "plots", "evaluation_visualization.png", sep = "/"))
ggplot(ev_summ, aes(Mean.of.precision, Mean.of.recall)) + 
  geom_point(aes(size = Sp.corr, shape = as.factor(window), colour = freq.threshold), alpha = 0.7) + 
  facet_wrap( ~ lexicon )
dev.off()

png(paste(path, "plots", "evaluation_visualization_variables.png", sep = "/"))
ggpairs(ev_summ)
dev.off()

