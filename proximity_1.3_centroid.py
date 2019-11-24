#author: Martina Astrid Rodda

# Import modules:

import os
import csv
from collections import defaultdict
import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr
from scipy import ndimage
import pandas as pd

# Default parameters:

window_default = 5
freq_threshold_default = 50

# Parameters:

window = input("What is the window size? Leave empty for default (" + str(window_default) + ").")  # 5 or 10 # size of context window for co-occurrence matrix
freq_threshold = input("What is the frequency threshold for vocabulary lemmas? Leave empty for default (" + str(freq_threshold_default) + ").")  # 2 or 50 # frequency threshold for lemmas in vocabulary
targetsname = input("What is the target file name? (format: target_$, enter $)")

if window == "":
    window = window_default

if freq_threshold == "":
    freq_threshold = freq_threshold_default

# Directory and file names:

directory = os.path.join("C:\\Users", "mrodda", "OneDrive - The Alan Turing Institute",
                         "MAR dphil project")
dir_ss = os.path.join(directory, "semantic_space", "sem_space_output")
dir_ss_rows = os.path.join(dir_ss, "ppmi_svd300", "w" + str(window), "w" + str(window) + "_spaces")
dir_out = os.path.join(directory, "outfiles")
dir_targets = os.path.join(directory, "digiclassics", "digiclassics homeric formulae")

# create output directory if it doesn't exist:
if not os.path.exists(os.path.join(directory, "outfiles")):
    os.makedirs(os.path.join(directory, "outfiles"))

# Input files:
ss_file_name = "CORE_SS.matrix_w" + str(window) + "_t" + str(freq_threshold) + ".ppmi.svd_300.dm"
targets_file_name = "targets_" + str(targetsname) + ".txt"

# Output files:
file_out_cos_distance = str(targetsname) + "_cos_distance" + str(window) + "_t" + str(freq_threshold) + "_centroid.csv"

# Initialize objects:
targets = list() #list of target lemmas
dissect_target2coordinates = dict()  # indexes a lemma to the array of its coordinates in the DISSECT semantic space

#creates list of targets from file
targets_file = open(os.path.join(dir_targets, targets_file_name), 'r', encoding="UTF-8")
count_n = 0
for line in targets_file:
    line = line.rstrip('\n')
    targets = line.split('\t')

#print check
print("Targets:", targets)

#first snippet of BmG code (modified) -- extracts stripped coordinates for each target lemma
ss_file = open(os.path.join(dir_ss_rows, ss_file_name), 'r', encoding="UTF-8")
print("Reading file:", ss_file_name)
count_n = 0
for line in ss_file:
    count_n += 1
    if count_n % 1000 == 0:
        print("Reading DISSECT coordinates, line", str(count_n))
    line = line.rstrip('\n')
    coordinates = line.split('\t')
    lemma = coordinates[0]
    if lemma in targets:
        coordinates.pop(0)
        dissect_target2coordinates[lemma] = np.asarray(list(np.float_(coordinates)))

#print check
# print(dissect_target2coordinates)

#print check
#print(dissect_target2coordinates)
print("Computing for " + str(len(dissect_target2coordinates)) + " targets")

# create array of coordinates from dictionary:
#initialise empty list:
centroid_list = list()
#populate list with dictionary entries:
for lemma in dissect_target2coordinates:
    centroid_list.append(dissect_target2coordinates[lemma])
#print check
# print(centroid_list)
#convert list to array:
centroid_array = np.asarray(centroid_list)
#print check
# print(centroid_array)

#compute centroid of target lemmas
centroid_targets = np.sum(centroid_array,axis=0)/len(dissect_target2coordinates)
#print check
# print(centroid_targets)

#compute cosine distance
with open(os.path.join(dir_out, file_out_cos_distance), 'w', encoding = 'UTF-8') as output:
    for target in dissect_target2coordinates:
        cos_distance_target2centroid = distance.cosine(dissect_target2coordinates[target], centroid_targets)
        output.write(str(cos_distance_target2centroid)+'\t')
