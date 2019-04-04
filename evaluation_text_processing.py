# # -*- coding: utf-8 -*- Author: Barbara McGillivray Date: 11/03/2019 Python version: 3 Script version: 1.0 Script
# for processing the data from Ancient Greek WordNet and prepare the data for the evaluation of the semantic space.


# ----------------------------
# Initialization
# ----------------------------


# Import modules:

import os
import csv
from collections import defaultdict
import numpy as np

from scipy.spatial import distance
from numpy import dot
from numpy.linalg import norm
import math

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import datetime

vm = input("Are you using the VM? Reply yes or no.")
if vm != "yes":
    import locale
    locale.setlocale(locale.LC_ALL, 'en_GB')

# Default parameters:

window_default = 5
freq_threshold_default = 50
istest_default = "yes"
lexicon_default = "AGWN"
first_evaluation_approach_yes_default = "no"
second_evaluation_approach_yes_default = "no"
third_evaluation_approach_yes_default = "no"
parameters_default = [window_default, freq_threshold_default, istest_default, lexicon_default,
                      first_evaluation_approach_yes_default,
                      second_evaluation_approach_yes_default, third_evaluation_approach_yes_default]

# Parameters:

window = input("What is the window size? Leave empty for default (" + str(
    window_default) + ").")  # 5 or 10 # size of context window for co-occurrence matrix
freq_threshold = input("What is the frequency threshold for vocabulary lemmas? Leave empty for default (" +
                       str(freq_threshold_default) + ").")  # 2 or 50 # frequency threshold for lemmas in vocabulary
istest = input("Is this a test? Leave empty for default (" + str(istest_default) + ").")
lexicon = input("Which lexicon do you want to consider? Leave empty for default (" + str(lexicon_default) +
                "). Another option is SCHMIDT.")
first_evaluation_approach_yes = input("Do you want to follow the first evaluation approach? Leave empty for default (" +
                                      first_evaluation_approach_yes_default + ").")
second_evaluation_approach_yes = input(
    "Do you want to follow the second evaluation approach? Leave empty for default (" +
    second_evaluation_approach_yes_default + ").")
third_evaluation_approach_yes = input("Do you want to follow the third evaluation approach? Leave empty for default (" +
                                      third_evaluation_approach_yes_default + ").")

parameters = [window, freq_threshold, istest, lexicon, first_evaluation_approach_yes, second_evaluation_approach_yes,
              third_evaluation_approach_yes]

for i_parameter in range(0, len(parameters)):
    if parameters[i_parameter] == "":
        parameters[i_parameter] = parameters_default[i_parameter]
[window, freq_threshold, istest, lexicon, first_evaluation_approach_yes, second_evaluation_approach_yes,
 third_evaluation_approach_yes] = parameters

lines_read_testing = 15000  # lines read in test case

now = datetime.datetime.now()

# Directory and file names:

if vm != "yes":
    directory = os.path.join("/Users", "bmcgillivray", "Documents", "OneDrive", "The Alan Turing Institute",
                         "Martina Astrid Rodda - MAR dphil project")
else:
    directory = os.path.join("/home", "bmcgillivray",
            "_Users_bmcgillivray_Documents_OneDrive_The_Alan_Turing_Institute_Martina_Astrid_Rodda_-_MAR_dphil_project")
dir_ss = os.path.join(directory, "semantic_space", "sem_space_output")
dir_ss_rows = os.path.join(dir_ss, "ppmi_svd300", "w" + str(window), "w" + str(window) + "_spaces")
dir_ss_neighbours = os.path.join(dir_ss, "ppmi_svd300", "w" + str(window), "w" + str(window) + "_nns")
dir_out = os.path.join(directory, "Evaluation", "output", "semantic-space-w"
                       + str(window) + "_t" + str(freq_threshold), "Lexicon_" + lexicon)
#dir_ss_neighbours = dir_ss
dir_lexicon = ""
if lexicon == "AGWN":
    dir_lexicon = os.path.join(directory, "TAL paper", "wordnet", "Open Ancient Greek WordNet 0.5")
else:
    dir_lexicon = os.path.join(directory, "TAL paper", "modern lexicography")

# create output directory if it doesn't exist:
if not os.path.exists(os.path.join(directory, "Evaluation", "output", "semantic-space-w"
        + str(window) + "_t" + str(freq_threshold), "Lexicon_" + lexicon)):
    os.makedirs(os.path.join(directory, "Evaluation", "output", "semantic-space-w"
                             + str(window) + "_t" + str(freq_threshold), "Lexicon_" + lexicon))

# Input files:

lexicon_file_name = ""
if lexicon == "AGWN":
    lexicon_file_name = "wn-data-grc.tab"
else:
    if lexicon == "SCHMIDT":
        lexicon_file_name = "schmidt.csv"

ss_file_name = "CORE_SS.matrix_w" + str(window) + "_t" + str(freq_threshold) + ".ppmi.svd_300.dm"
ssrows_file_name = "CORE_SS.matrix_w" + str(window) + "_t" + str(freq_threshold) + ".ppmi.svd_300.rows"
neighbours_file_name = "NEIGHBOURS.matrix_w" + str(window) + "_t" + str(freq_threshold) + ".rows.CORE_SS.matrix_w" \
                       + str(window) + "_t" + str(freq_threshold) + ".ppmi.svd_300.cos"

# Output files:

file_out_lexicon_cooccurrence_name = "Lexicon_" + lexicon + "co-occurrences.csv"  # matrix with 0s and 1s depending on whether two lemmas
# co-occur in the same Lexicon synset
# file_out_lexicon_distances_name = "Lexicon_distances.txt"  # matrix with cosine distances between pairs of Lexicon lemmas
file_out_dissect_5neighbour_name = "semantic-space_w" + str(window) + "_t" + str(
    freq_threshold) + "5neighbours_lemmas.txt"
file_out_dissect_5neighbour_distances_name = "semantic-space_w" + str(window) + "_t" + str(
    freq_threshold) + "5neighbours_distances.txt"
file_out_shared_lemmas_name = "Lexicon_" + lexicon + "-semantic-space_w" + str(window) + "_t" + str(
    freq_threshold) + "_shared-lemmas.txt"
file_out_lexicon_vocabulary_name = "Lexicon_" + lexicon + "_lemmas.txt"
# file_out_dissect_distances_name = "semantic_space_w" + str(window) + "_t" + str(freq_threshold) + "distances.csv"
hist_file_name = "cos-distances-5neighbours_semantic-space_w" + str(window) + "_t" + str(freq_threshold) + "hist.png"
summary_stats_dissect_5neighbours_file_name = "summary_statistics_distance_Lexicon_" + lexicon + \
                                              "-semantic-space_w" + str(window) + "_t" + \
                                              str(freq_threshold) + "_5neighbours" + ".txt"
summary_dissect_lexicon_distances_file_name = "summary_comparison_distances_Lexicon_" + lexicon + \
                                              "_semantic-space_w" + str(window) + "_t" + \
                                              str(freq_threshold) + "_5neighbours" + ".txt"
dissect_lexicon_distances_synonyms_file_name = "distances_synonyms_Lexicon_" + lexicon + "_semantic-space_w" + str(
    window) + "_t" + \
                                               str(freq_threshold) + "_5neighbours" + ".txt"
dissect_lexicon_distances_neighbours_file_name = "distances_neighbours_Lexicon_" + lexicon + "_semantic-space_w" + str(
    window) + "_t" + \
                                                 str(freq_threshold) + "_5neighbours" + ".txt"
summary_dissect_lexicon_overlap_file_name = "summary_overlap_Lexicon_" + lexicon + "_semantic-space_w" + str(
    window) + "_t" + \
                                            str(freq_threshold) + "_5neighbours" + ".txt"
log_file_name = "log_file_Lexicon_" + lexicon + "semantic-space_w" + str(window) + "_t" + str(
    freq_threshold) + "_5neighbours" + \
                str(now.strftime("%Y-%m-%d %H-%M")) + ".txt"
log_file_name = log_file_name.replace("/", "")
# str(now.strftime("%Y-%m-%d")) + ".txt"


if istest == "yes":
    file_out_lexicon_cooccurrence_name = file_out_lexicon_cooccurrence_name.replace(".csv", "_test.csv")
    # file_out_lexicon_distances_name = file_out_lexicon_distances_name.replace(".txt", "_test.txt")
    file_out_dissect_5neighbour_name = file_out_dissect_5neighbour_name.replace(".txt", "_test.txt")
    file_out_dissect_5neighbour_distances_name = file_out_dissect_5neighbour_distances_name.replace(".txt", "_test.txt")
    file_out_shared_lemmas_name = file_out_shared_lemmas_name.replace(".txt", "_test.txt")
    file_out_lexicon_vocabulary_name = file_out_lexicon_vocabulary_name.replace(".txt", "_test.txt")
    # file_out_dissect_distances_name = file_out_dissect_distances_name.replace(".csv", "_test.csv")
    hist_file_name = hist_file_name.replace(".png", "_test.png")
    summary_stats_dissect_5neighbours_file_name = summary_stats_dissect_5neighbours_file_name.replace(".txt",
                                                                                                      "_test.txt")
    summary_dissect_lexicon_distances_file_name = summary_dissect_lexicon_distances_file_name.replace(".txt",
                                                                                                      "_test.txt")
    dissect_lexicon_distances_neighbours_file_name = dissect_lexicon_distances_neighbours_file_name.replace(".txt",
                                                                                                            "_test.txt")
    dissect_lexicon_distances_synonyms_file_name = dissect_lexicon_distances_synonyms_file_name.replace(".txt",
                                                                                                        "_test.txt")
    summary_dissect_lexicon_overlap_file_name = summary_dissect_lexicon_overlap_file_name.replace(".txt", "_test.txt")
    log_file_name = log_file_name.replace(".txt", "_test.txt")

# Initialize objects:

lexicon_id2lemma = dict()  # maps an id to its lemma in AG WordNet
lexicon_lemma2id = dict()  # maps an Lexicon lemma to its id
synsets = dict()  # maps a synset ID to the list of its contents
# synsetid2def = dict()  # maps a synset ID to its English definition
lexicon_cooccurrence = defaultdict(
    dict)  # multidimensional dictionary: maps each pair of *shared* lemmas to 1 if they co-occur in the same lexicon synset,
# and 0 otherwise
lexicon_coordinates = dict()  # maps a *shared* Lexicon lemma id to the list of its 0/1 coordinates in the Lexicon space
# lexicon_distances = list()  # indexes an Lexicon lemma id to the array of cosine distances with other Lexicon lemmas
lexicon_dissect_5neighbours = list()  # list of lemmas shared between Lexicon and DISSECT space with 5 neighbours
dissect_lemmas_5neighbours = list()  # list of lemmas in DISSECT space, for which we have the top 5 neighbours
lemma_neighbour2distance = defaultdict(
    dict)  # maps a pair of lemma and each of its top 5 DISSECT neighbours to their distance,
# excluding the cases where the neighbour is the lemma itself
dissect_id2lemma = dict()  # list of lemmas in DISSECT semantic space's rows: an index corresponds to a lemma
dissect_lemma2coordinates = dict()  # indexes a lemma to the array of its coordinates in the DISSECT semantic space
dissect_lemma2cosinedistance = list()  # indexes a lemma to the array of cosine distances with other lemmas in
# the DISSECT semantic space
# dissect_distances = list()  # list of distance values from dissect_lemma2cosinedistance
dissect_distances_5neighbour = list()  # list of distance values from dissect_lemma2cosinedistance only considering
# the top 5 neighbours for each lemma
dissect_distances_lexicon_dissect_5neighbour = list()  # list of distance values from dissect_lemma2cosinedistance only
# considering the lemmas shared between top 5 neighbours in DISSECT and Lexicon lemmas
precisions = list()  # list of precision values for third evaluation approach
recalls = list()  # list of recall values for third evaluation approach
lemma2precision = dict()  # maps a shared lemma to the precision calculated from the overlap between its Lexicon
# synonyms and its DISSECT neighbours, divided by the length of the neighbourset
lemma2recall = dict()  # maps a shared lemma to the precision calculated from the overlap between its Lexicon
# synonyms and its DISSECT neighbours, divided by the length of the synset
# Open log file:

log_file = open(os.path.join(dir_out, log_file_name), 'w', encoding="UTF-8")

now = datetime.datetime.now()

log_file.write(str(now.strftime("%Y-%m-%d %H:%M")) + "\n")

# ----------------------------------------
# Index lemmas in DISSECT semantic space:
# ----------------------------------------

print("Reading DISSECT lemmas...")
log_file.write("Reading DISSECT lemmas..." + "\n")

ssrows_file = open(os.path.join(dir_ss_rows, ssrows_file_name), 'r', encoding="UTF-8")
row_count_rows = sum(1 for line in ssrows_file)
ssrows_file.close()

ssrows_file = open(os.path.join(dir_ss_rows, ssrows_file_name), 'r', encoding="UTF-8")
count_n = -1

lemma = ""
for line in ssrows_file:
    count_n += 1

    if (istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_rows):
        if count_n % 1000 == 0:
            log_file.write("Reading rows in DISSECT, line " + str(count_n) + "\n")
        line = line.rstrip('\n')
        dissect_id2lemma[count_n] = line
        if (line == "κομιδή") or (line == "ἐπιμέλεια"):
            log_file.write("\tTest!" + "\n")
            log_file.write("\tdissect id: " + str(count_n) + " , lemma: " + dissect_id2lemma[count_n] + "\n")

ssrows_file.close()

log_file.write("Examples..." + "\n")
log_file.write("dissect id = 0, lemma: " + dissect_id2lemma[0] + "\n")
log_file.write("dissect id = 29, lemma: " + dissect_id2lemma[29] + "\n")
log_file.write("dissect id = 188, lemma: " + dissect_id2lemma[188] + "\n")

# --------------------------------------
# Read WordNet file to collect synsets:
# --------------------------------------

print("-----------------\nReading Lexicon...\n-----------------")
log_file.write("-----------------\nReading Lexicon...\n-----------------" + "\n")

lexicon_file = open(os.path.join(dir_lexicon, lexicon_file_name), 'r', encoding="UTF-8")
row_count_lexicon = sum(1 for line in lexicon_file)
lexicon_file.close()

# Read synsets from WordNet file:

count_n = 1  # counts lines read
lexicon_file = open(os.path.join(dir_lexicon, lexicon_file_name), 'r', encoding="UTF-8")
lexicon_reader = csv.reader(lexicon_file, delimiter="\t")
next(lexicon_reader)  # This skips the first row of the CSV file

synset_def = ""
synset_lemma = ""

for row in lexicon_reader:
    count_n += 1
    if ((istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_lexicon)) and (
                count_n > 1):

        if count_n % 1000 == 0:
            log_file.write("WordNet: line " + str(count_n) + " out of " + str(row_count_lexicon) + "\n")

        row[0] = row[0].replace("#", "")
        synset_id = row[0]
        if row[1] == "eng:def":
            synset_def = row[3]
            # log_file.write("Def:", synset_def + "\n")
            # synsetid2def[synset_id] = synset_def
        else:
            synset_lemma = row[2]

            if synset_id in synsets:
                synsets_this_id = synsets[synset_id]
                synsets_this_id.append(synset_lemma)
                synsets[synset_id] = synsets_this_id
            else:
                synsets[synset_id] = [synset_lemma]
                # log_file.write("\tSynset ID:", str(synset_id), "; lemma:", synset_lemma + "\n")

lexicon_file.close()

try:
    log_file.write("Examples:" + "\n")
    log_file.write("synset_id: 00267522-n , synonyms: " + str(synsets['00267522-n']) + "\n")
    log_file.write("synset_id: 02078399-a , synonyms: " + str(synsets['02078399-a']) + "\n")
except:
    KeyError

# ---------------------------------------------
# read data from DISSECT semantic space
# ---------------------------------------------

# neighbours in semantic space:

print(
    "-----------------------------------------\nReading neighbours in DISSECT "
    "space....\n----------------------------------------")
log_file.write(
    "-----------------------------------------\nReading neighbours in DISSECT "
    "space....\n----------------------------------------" + "\n")

neighbours_file = open(os.path.join(dir_ss_neighbours, neighbours_file_name), 'r', encoding="UTF-8")
row_count_neighbours = sum(1 for line in neighbours_file)
neighbours_file.close()

neighbours_file = open(os.path.join(dir_ss_neighbours, neighbours_file_name), 'r', encoding="UTF-8")
count_n = 0

lemma = ""
for line in neighbours_file:
    count_n += 1

    if (istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_neighbours):
        if count_n % 1000 == 0:
            log_file.write("Reading neighbours in DISSECT, line " + str(count_n) + "\n")
        line = line.rstrip('\n')
        if not line.startswith("\t"):
            # log_file.write("lemma!", line + "\n")
            lemma = line
            dissect_lemmas_5neighbours.append(lemma)
        else:
            # log_file.write("neighbour!", line + "\n")
            fields = line.split("\t")
            # log_file.write("fields:", fields + "\n")
            neighbour_distance = fields[1]
            # log_file.write("neighbour_distance:", neighbour_distance + "\n")
            neighbour_distance_fields = neighbour_distance.split(" ")
            # log_file.write("neighbour_distance_fields:", neighbour_distance_fields + "\n")
            neighbour = neighbour_distance_fields[0]
            # log_file.write("neighbour:", neighbour + "\n")
            neighbour_distance = neighbour_distance_fields[1]
            # log_file.write("distance:", neighbour_distance + "\n")
            if neighbour != lemma:
                dissect_lemmas_5neighbours.append(neighbour)
                lemma_neighbour2distance[lemma, neighbour] = float(neighbour_distance)
                dissect_distances_5neighbour.append(neighbour_distance)
                if lemma in ["κομιδή", "ἐπιμέλεια", "ἴασις"]:
                    log_file.write("\tTest!" + "\n")
                    log_file.write(
                        "\tlemma: " + lemma + ", neighbour: " + neighbour + ", line: " + str(count_n) +
                        ", lemma_neighbour2distance[lemma, neighbour]: " +
                        str(lemma_neighbour2distance[lemma, neighbour]) + "\n")

neighbours_file.close()

log_file.write("Examples:" + "\n")
for lemma, neighbour in lemma_neighbour2distance:
    if lemma in ["ἐπιμέλεια", "κομιδή", "ἴασις"]:
        log_file.write("Lemma: " + lemma + ", its neighbour and distance: " + neighbour + " "
                       + str(lemma_neighbour2distance[lemma, neighbour]) + "\n")

# print out list of DISSECT lemmas with top 5 neighbours:

with open(os.path.join(dir_out, file_out_dissect_5neighbour_name), 'w',
          encoding="UTF-8") as file_out_dissect_5neighbour:
    for lemma in dissect_lemmas_5neighbours:
        file_out_dissect_5neighbour.write("%s\n" % lemma)

# print out distances between DISSECT lemmas with top 5 neighbours and their top 5 neighbours:

with open(os.path.join(dir_out, file_out_dissect_5neighbour_distances_name), 'w',
          encoding="UTF-8") as file_out_dissect_5neighbour_distances:
    file_out_dissect_5neighbour_distances_writer = csv.writer(file_out_dissect_5neighbour_distances, delimiter="\t")

    file_out_dissect_5neighbour_distances_writer.writerow(["lemma", "neighbour", "distance"])
    for lemma, neighbour in lemma_neighbour2distance:
        # log_file.write("lemma:", lemma, "neighbour:", neighbour, "distance:", lemma_neighbour2distance[lemma, neighbour] + "\n")
        file_out_dissect_5neighbour_distances_writer.writerow(
            [lemma, neighbour, str(lemma_neighbour2distance[lemma, neighbour])])

# ---------------------------------------------------------------------------------------
# Print co-occurrence matrix from WordNet synsets for lemmas shared with DISSECT lemmas
# ---------------------------------------------------------------------------------------

# create vocabulary list and co-occurrence pairs:

print(
    "------------------------------------------\nDefining Lexicon co-occurrence pairs...\n------------------------------------------")
log_file.write(
    "------------------------------------------\nDefining Lexicon co-occurrence pairs...\n------------------------------------------" + "\n")

count_s = 0
for synset_id in synsets:

    count_s += 1
    print("Line: " + str(count_s) + ", synset ID: " + str(synset_id) + "\n")

    if count_s % 1000 == 0:
        log_file.write("Line: "+str(count_s) + ", synset ID: " + str(synset_id) + "\n")
        log_file.write("Lemmas: " + str(synsets[synset_id]) + "\n")


    if synset_id == "09620078-n":
        log_file.write("\tTest! \n")
        log_file.write("Line: "+str(count_s) + ", synset ID: " + str(synset_id) + "\n")
        log_file.write("Synset_id: " + str(synset_id) + ", lemmas: " + str(synsets[synset_id]) + "\n")

    synsets_this_lemma = synsets[synset_id]

    count_l = 0
    for lemma1 in synsets_this_lemma:

        count_l += 1
        id1 = synset_id + str(count_l)
        # log_file.write("Defining co-occurrences of lemma1: "+ lemma1 + "\n")
        lexicon_id2lemma[id1] = lemma1
        lexicon_lemma2id[lemma1] = id1
        # log_file.write("Vocabulary so far: "+ str(lexicon_vocabulary) + "\n")

        for lemma2 in synsets_this_lemma:

            if lemma1 in dissect_lemmas_5neighbours and lemma2 in dissect_lemmas_5neighbours:
                # if lemma1 is not lemma2:
                if (lemma1, lemma2) in lexicon_cooccurrence:
                    lexicon_cooccurrence[lemma1, lemma2] += 1
                else:
                    lexicon_cooccurrence[lemma1, lemma2] = 1

                if count_s % 1000 == 0:
                    log_file.write(
                        str(count_s) + ": " + ", lemma1: " + lemma1 + ", lemma2: " + lemma2 + ", co-occurrence: " +
                        str(lexicon_cooccurrence[lemma1, lemma2]) + "\n")

                if lemma1 in ["κομιδή", "ἐπιμέλεια"]:  # , "οἰκήτωρ", "οἰκήτωρ", "αἴθων", "εὐθυθάνατος"]:
                    log_file.write("\tTest!\n")
                    log_file.write("\tLemma1: " + lemma1 + ", lemma2: " + lemma2 + ", co-occurrence: " +
                                   str(lexicon_cooccurrence[lemma1, lemma2]) + "\n")

log_file.write("Writing lexicon_id2lemma file....\n")
# lexicon_lemma2id_keys = ["0"] + list(lexicon_lemma2id.keys()) # I add an empty string as first element of this list, to account for the fact that the
lexicon_lemma2id_keys = list(lexicon_lemma2id.keys())
lexicon_lemma2id_keys.sort()
# lexicon_lemma2id["0"] = "0"
# first lemma in the file file_out_lexicon_vocabulary_name (in line 1) corresponds to the item at position 1 in the list

with open(os.path.join(dir_out, file_out_lexicon_vocabulary_name), 'w', encoding="UTF-8") as lexicon_vocabulary_file:
    count_n = 0
    for lemma in lexicon_lemma2id_keys:
        count_n += 1
        if count_n > 1:
            lexicon_vocabulary_file.write(lexicon_lemma2id[lemma] + "\t" + lemma + "\n")

        if lemma in ["0", "κομιδή", "ἐπιμέλεια"]:  # , "οἰκήτωρ", "οἰκήτωρ", "αἴθων", "εὐθυθάνατος"]:
            log_file.write("\tTest!\n")
            # log_file.write("\tLine number: " + str(count_n-1) + ", lemma_id: " + lexicon_lemma2id[lemma] +
            # ", lemma: " + lemma + "\n")
            log_file.write("\tLine number: " + str(count_n - 1) + ", lemma_id: " + lexicon_lemma2id[lemma] +
                           ", lemma: " + lemma + "\n")

# -------------------------------------------------------------------------------
# find lemmas present in both Lexicon and DISSECT space (called “shared lemmas”)
# -------------------------------------------------------------------------------

print(
    "---------------------------------------------\nFind shared lemmas between Lexicon and DISSECT space....\n---------------------------------------")
log_file.write(
    "---------------------------------------------\nFind shared lemmas between Lexicon and DISSECT "
    "space....\n---------------------------------------" + "\n")

lexicon_dissect_5neighbours = list((set(dissect_lemmas_5neighbours).intersection(set(lexicon_id2lemma.values()))))
lexicon_dissect_5neighbours.sort()

count_n = 0
with open(os.path.join(dir_out, file_out_shared_lemmas_name), 'w',
          encoding="UTF-8") as out_shared_lemmas_file:
    for lemma in lexicon_dissect_5neighbours:
        count_n += 1
        out_shared_lemmas_file.write("%s\n" % lemma)
        if lemma in ["0", "κομιδή", "ἐπιμέλεια"]:  # , "οἰκήτωρ", "οἰκήτωρ", "αἴθων", "εὐθυθάνατος"]:
            log_file.write("\tTest!\n")
            log_file.write("\tLine number: " + str(count_n) + ", lemma_id: " + lexicon_lemma2id[lemma] +
                           ", lemma: " + lemma + "\n")

# ------------------------------------------
# finish defining Lexicon co-occurrence pairs:
# ------------------------------------------

print(
    "----------------------------------------\nFinish defining Lexicon co-occurrence pairs...\n-------------------------------------------------")
log_file.write(
    "----------------------------------------\nFinish defining Lexicon co-occurrence pairs...\n-------------------------------------------------" + "\n")

count_n = 0

with open(os.path.join(dir_out, file_out_lexicon_cooccurrence_name), 'w',
          encoding="UTF-8") as file_out_lexicon_cooccurrence:
    file_out_lexicon_cooccurrence_writer = csv.writer(file_out_lexicon_cooccurrence, delimiter="\t")
    file_out_lexicon_cooccurrence_writer.writerow([""] + lexicon_dissect_5neighbours)

    for lemma1 in lexicon_dissect_5neighbours:
        count_n += 1
        coordinates_lemmaid1 = list()
        count_n2 = 0

        for lemma2 in lexicon_dissect_5neighbours:
        # for lemma2 in lexicon_lemma2id_keys:
            if lemma2 is not "0":
                count_n2 += 1
                if ((lemma1, lemma2) not in lexicon_cooccurrence):  # or lemma1 is lemma2):
                    lexicon_cooccurrence[lemma1, lemma2] = 0

                if count_n % 5000 == 0 and count_n2 % 500 == 0:
                    log_file.write("Index (in the list of shared lemmas) of lemma1: " +
                                   str(count_n) + " out of " + str(len(lexicon_dissect_5neighbours)) +
                                   " and index (in the list of Lexicon lemmas) of lemma2 " + str(count_n2) +
                                   ": " + " co-occurrence of shared lemmas " + lemma1 + " and " + lemma2 + ": " +
                                   str(lexicon_cooccurrence[lemma1, lemma2]) + "\n")

                if lemma1 in ["κομιδή", "ἐπιμέλεια"]:  # , "οἰκήτωρ", "πόλις", "αἴθων", "εὐθυθάνατος"]:
                    if lexicon_cooccurrence[lemma1, lemma2] > 0:
                        log_file.write("\tTest!" + "\n")
                        log_file.write("\tIndex (in the list of shared lemmas) of lemma1: " +
                                       str(count_n) + " out of " + str(len(lexicon_dissect_5neighbours)) +
                                       " and index (in the list of Lexicon lemmas) of lemma2 " + str(count_n2) +
                                       ", lemma1: " + lemma1 + " (id: " + lexicon_lemma2id[lemma1] + ") " +
                                       ", lemma2: " + lemma2 + " (id: " + lexicon_lemma2id[lemma2] + ") " +
                                       ", co-occurrence of shared lemmas: " + str(
                            lexicon_cooccurrence[lemma1, lemma2]) + "\n")

                coordinates_lemmaid1.append(lexicon_cooccurrence[lemma1, lemma2])

        file_out_lexicon_cooccurrence_writer.writerow([lemma1] + [coordinates_lemmaid1])

        lexicon_coordinates[lemma1] = coordinates_lemmaid1

        if lemma1 in ["κομιδή", "ἐπιμέλεια"]:  # , "οἰκήτωρ", "πόλις", "αἴθων", "εὐθυθάνατος"]:
            log_file.write("\tTest!" + "\n")
            log_file.write(
                "\tCount: " + str(count_n) + " Lemma1:" + lemma1 + " (id: " + lexicon_lemma2id[lemma1] + ")" + "\n")
            log_file.write("\tNon-zero Lexicon coordinates at positions/lemmas:" + "\n")
            l1_c = lexicon_coordinates[lemma1]
            # log_file.write(str([(i, lexicon_dissect_5neighbours[i]) for i, e in enumerate(l1_c) if e != 0]) + "\n")
            log_file.write(str([(i, lexicon_lemma2id_keys[i]) for i, e in enumerate(l1_c) if e != 0]) + "\n")

log_file.write("Lexicon co-occurrence:" + "\n")
# log_file.write(str(lexicon_cooccurrence) + "\n")
try:
    log_file.write("Examples:" + "\n")
    log_file.write(
        "κομιδή " + " (id:  " + lexicon_lemma2id["κομιδή"] + ") " + ", ἐπιμέλεια " + " (id: " + lexicon_lemma2id[
            "ἐπιμέλεια"] + ") " +
        str(lexicon_cooccurrence["κομιδή", "ἐπιμέλεια"]) + "\n")
    log_file.write(
        "ἐπιμέλεια " + " (id:  " + lexicon_lemma2id["ἐπιμέλεια"] + ") " + ", κομιδή " + "(id: " + lexicon_lemma2id[
            "κομιδή"] + ") " +
        str(lexicon_cooccurrence["ἐπιμέλεια", "κομιδή"]) + "\n")

    log_file.write("Lexicon coordinates:" + "\n")
    log_file.write("Examples:" + "\n")
except:
    pass

try:
    log_file.write("κομιδή (id: " + lexicon_lemma2id["κομιδή"] + "):" + "\n")
    log_file.write("Non-zero Lexicon coordinates at positions/lemmas:" + "\n")
    l1_c = lexicon_coordinates["κομιδή"]
    log_file.write(str([(i, lexicon_lemma2id_keys[i]) for i, e in enumerate(l1_c) if e != 0]) + "\n")
except:
    pass

try:
    log_file.write("ἐπιμέλεια (id: " + lexicon_lemma2id["ἐπιμέλεια"] + "):" + "\n")
    log_file.write("Non-zero Lexicon coordinates at positions/lemmas:" + "\n")
    l1_c = lexicon_coordinates["ἐπιμέλεια"]
    log_file.write(str([(i, lexicon_lemma2id_keys[i]) for i, e in enumerate(l1_c) if e != 0]) + "\n")
except:
    pass

try:
    log_file.write("οἰκήτωρ (id: " + lexicon_lemma2id["οἰκήτωρ"] + "):" + "\n")
    log_file.write("Non-zero Lexicon coordinates at positions/lemmas:" + "\n")
    l1_c = lexicon_coordinates["οἰκήτωρ"]
    log_file.write(str([(i, lexicon_lemma2id_keys[i]) for i, e in enumerate(l1_c) if e != 0]) + "\n")
except:
    pass

try:
    log_file.write("πόλις (id: " + lexicon_lemma2id["πόλις"] + "):" + "\n")
    log_file.write("Non-zero Lexicon coordinates at positions/lemmas:" + "\n")
    l1_c = lexicon_coordinates["πόλις"]
    log_file.write(str([(i, lexicon_lemma2id_keys[i]) for i, e in enumerate(l1_c) if e != 0]) + "\n")
except:
    pass

try:
    log_file.write("βάξις (id: " + lexicon_lemma2id["βάξις"] + "):" + "\n")
    log_file.write("Non-zero Lexicon coordinates at positions/lemmas:" + "\n")
    l1_c = lexicon_coordinates["βάξις"]
    log_file.write(str([(i, lexicon_lemma2id_keys[i]) for i, e in enumerate(l1_c) if e != 0]) + "\n")
except:
    pass

# read semantic space coordinates:

print("--------------------------------\nReading semantic space coordinates\n---------------------------------------")
log_file.write(
    "--------------------------------\nReading semantic space coordinates\n---------------------------------------" + "\n")

ss_file = open(os.path.join(dir_ss_rows, ss_file_name), 'r', encoding="UTF-8")
row_count_ss = sum(1 for line in ss_file)
ss_file.close()

ss_file = open(os.path.join(dir_ss_rows, ss_file_name), 'r', encoding="UTF-8")
count_n = 0
for line in ss_file:
    count_n += 1
    if (istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_ss):

        if count_n % 1000 == 0:
            log_file.write("Reading DISSECT coordinates, line " + str(count_n) + "\n")
        line = line.rstrip('\n')
        coordinates = line.split('\t')
        lemma = coordinates[0]
        coordinates.pop(0)
        dissect_lemma2coordinates[lemma] = np.asarray(list(np.float_(coordinates)))

        if lemma == "κομιδή" or lemma == "ἐπιμέλεια":
            log_file.write("\tTest!" + "\n")
            log_file.write("\tlemma:" + lemma + "\n")
            log_file.write("\tDISSECT coordinates:" + "\n")
            log_file.write(str(dissect_lemma2coordinates[lemma]) + "\n")

ss_file.close()

log_file.write("Examples:" + "\n")
log_file.write("lemma: ἐπιμέλεια" + "\n")
log_file.write("DISSECT coordinates:" + "\n")
try:
    log_file.write(str(dissect_lemma2coordinates["ἐπιμέλεια"]) + "\n")
except:
    log_file.write("No ἐπιμέλεια")

log_file.write("lemma: ἐπιμέλεια" + "\n")
log_file.write("DISSECT coordinates :" + "\n")
try:
    log_file.write(str(dissect_lemma2coordinates["κομιδή"]) + "\n")
except:
    log_file.write("No κομιδή")

# ---------------------------------------------
# Evaluation approaches:
# ---------------------------------------------


print("There are", str(len(lexicon_id2lemma.values())), " lemmas in Lexicon, ", str(len(dissect_lemmas_5neighbours)),
      " lemmas/neighbours in DISSECT space with 5 top neighbours", " and ", str(len(lexicon_dissect_5neighbours)),
      " in the intersection.")
log_file.write(
    "There are " + str(len(lexicon_id2lemma.values())) + " lemmas in Lexicon, " + str(len(dissect_lemmas_5neighbours)) +
    " lemmas/neighbours in DISSECT space with 5 top neighbours " + " and " + str(len(lexicon_dissect_5neighbours)) +
    " in the intersection." + "\n")
# There are 22828 lemmas in Lexicon,  44740 lemmas in DISSECT space with 5 top neighbours and 12899 in the intersection.

if first_evaluation_approach_yes == "yes":
    print(
        "---------------------------------------------------------\nFirst evaluation approach: distances in DISSECT space\n---------------------------------------------------------")
    log_file.write(
        "---------------------------------------------------------\nFirst evaluation approach: distances in DISSECT space\n---------------------------------------------------------" + "\n")

    # 1.	Distances in DISSECT space
    # a.	Calculate distance between pairs of shared lemmas that are among top 5 neighbours in DISSECT space
    # b.	Compare this distance with average distance between pairs of lemmas and their neighbours in DISSECT space

    summary_stats_dissect_5neighbours_file = open(os.path.join(dir_out, summary_stats_dissect_5neighbours_file_name),
                                                  'w')
    # calculate summary statistics of distance between pairs of lemmas and their top 5 neighbours in DISSECT space:

    # log_file.write(str(type(dissect_distances_5neighbour)) + "\n")
    # dissect_distances_5neighbour = np.asarray(dissect_distances_5neighbour)
    dissect_distances_5neighbour = np.array(dissect_distances_5neighbour, dtype=np.float32)
    # log_file.write(str(type(dissect_distances_5neighbour)) + "\n")
    # log_file.write(str(dissect_distances_5neighbour) + "\n")
    mean_dissect_distances_5neighbour = np.mean(dissect_distances_5neighbour)
    std_dissect_distances_5neighbour = dissect_distances_5neighbour.std()
    min_dissect_distances_5neighbour = dissect_distances_5neighbour.min()
    max_dissect_distances_5neighbour = dissect_distances_5neighbour.max()
    median_dissect_distances_5neighbour = np.median(dissect_distances_5neighbour)
    perc25_dissect_distances_5neighbour = np.percentile(dissect_distances_5neighbour, 25)
    perc75_dissect_distances_5neighbour = np.percentile(dissect_distances_5neighbour, 75)

    # plot distribution of distances between lemmas and top 5 neighbours in DISSECT space:

    plt.hist(dissect_distances_5neighbour)
    plt.title('Distribution of cosine distance in DISSECT space 5 neighbours')
    plt.xlabel('cosine dist')
    plt.ylabel('number')
    # plt.show()
    plt.savefig(os.path.join(dir_out, hist_file_name))

    # Define list of DISSECT distances between pairs of shared lemmas:

    count_n = 0
    for lemma, neighbour in lemma_neighbour2distance:
        if lemma in lexicon_dissect_5neighbours and neighbour in lexicon_dissect_5neighbours:
            count_n += 1
            if count_n % 1000 == 0:
                log_file.write("lemma: " + lemma + ", neighbour: " + neighbour + ", distance: " +
                               str(lemma_neighbour2distance[lemma, neighbour]) + "\n")
            if lemma in ["κομιδή", "ἐπιμέλεια", "ἴασις"]:
                log_file.write("\tTest!" + "\n")
                log_file.write("\tlemma: " + lemma + ", neighbour: " + neighbour + ", distance: " +
                               str(lemma_neighbour2distance[lemma, neighbour]) + "\n")

            dissect_distances_lexicon_dissect_5neighbour.append(lemma_neighbour2distance[lemma, neighbour])

    # Calculate summary statistics of distance between pairs of shared lemmas that are among top 5 neighbours in DISSECT space:

    if len(lexicon_dissect_5neighbours) > 0:
        dissect_distances_lexicon_dissect_5neighbour = np.array(dissect_distances_lexicon_dissect_5neighbour,
                                                                dtype=np.float32)

        mean_dissect_distances_lexicon_dissect_5neighbour = np.mean(dissect_distances_lexicon_dissect_5neighbour)
        std_dissect_distances_lexicon_dissect_5neighbour = dissect_distances_lexicon_dissect_5neighbour.std()
        min_dissect_distances_lexicon_dissect_5neighbour = dissect_distances_lexicon_dissect_5neighbour.min()
        max_dissect_distances_lexicon_dissect_5neighbour = dissect_distances_lexicon_dissect_5neighbour.max()
        median_dissect_distances_lexicon_dissect_5neighbour = np.median(dissect_distances_lexicon_dissect_5neighbour)
        perc25_dissect_distances_lexicon_dissect_5neighbour = np.percentile(
            dissect_distances_lexicon_dissect_5neighbour, 25)
        perc75_dissect_distances_lexicon_dissect_5neighbour = np.percentile(
            dissect_distances_lexicon_dissect_5neighbour, 75)

        summary_stats_dissect_5neighbours_file.write(
            "Mean of DISSECT distances 5 neighbours: " + str(mean_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "Mean of DISSECT distances of shared lemmas between DISSECT top 5 neighbours "
            "and Lexicon " + lexicon + " lemmas: " + str(
                mean_dissect_distances_lexicon_dissect_5neighbour) + "\n")
        log_file.write("Mean of DISSECT distances 5 neighbours: " + str(mean_dissect_distances_5neighbour) + "\n")
        log_file.write("Mean of DISSECT distances of shared lemmas between DISSECT top 5 neighbours "
                       "and Lexicon " + lexicon + "lemmas: " + str(
            mean_dissect_distances_lexicon_dissect_5neighbour) + "\n")

        summary_stats_dissect_5neighbours_file.write(
            "STD of DISSECT distances 5 neighbours: " + str(std_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "STD of distances of shared lemmas between DISSECT top 5 neighbours "
            "and Lexicon " + lexicon + " lemmas: " + str(
                std_dissect_distances_lexicon_dissect_5neighbour) + "\n")

        summary_stats_dissect_5neighbours_file.write(
            "Min of DISSECT distances 5 neighbours: " + str(min_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "Min of DISSECT distances of shared lemmas between DISSECT top 5 neighbours and Lexicon " + lexicon + " lemmas: " + str(
                min_dissect_distances_lexicon_dissect_5neighbour) + "\n")

        summary_stats_dissect_5neighbours_file.write(
            "Max of DISSECT distances 5 neighbours: " + str(max_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "Max of DISSECT distances of shared lemmas between DISSECT top 5 neighbours and Lexicon" + lexicon + " lemmas: " + str(
                max_dissect_distances_lexicon_dissect_5neighbour) + "\n")

        summary_stats_dissect_5neighbours_file.write(
            "Median of DISSECT distances 5 neighbours: " + str(median_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "Median of DISSECT distances of shared lemmas between DISSECT top 5 neighbours and Lexicon" + lexicon + " lemmas: " + str(
                median_dissect_distances_lexicon_dissect_5neighbour) + "\n")

        summary_stats_dissect_5neighbours_file.write(
            "25th percentile of DISSECT distances 5 neighbours: " + str(perc25_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "25th percentile of DISSECT distances of shared lemmas between DISSECT top 5 neighbours and Lexicon" + lexicon + " lemmas: " + str(
                perc25_dissect_distances_lexicon_dissect_5neighbour) + "\n")

        summary_stats_dissect_5neighbours_file.write(
            "75th percentile of DISSECT distances 5 neighbours: " + str(perc75_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "75th percentile of DISSECT distances of shared lemmas between DISSECT top 5 "
            "neighbours and Lexicon" + lexicon + " lemmas: " + str(
                perc75_dissect_distances_lexicon_dissect_5neighbour) + "\n")

    else:
        summary_stats_dissect_5neighbours_file.write(
            "Mean of DISSECT distances 5 neighbours: " + str(mean_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "STD of DISSECT distances 5 neighbours: " + str(std_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "Min of DISSECT distances 5 neighbours: " + str(min_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "Max of DISSECT distances 5 neighbours: " + str(max_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "Median of DISSECT distances 5 neighbours: " + str(median_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "25th percentile of DISSECT distances 5 neighbours: " + str(perc25_dissect_distances_5neighbour) + "\n")
        summary_stats_dissect_5neighbours_file.write(
            "75th percentile of DISSECT distances 5 neighbours: " + str(perc75_dissect_distances_5neighbour) + "\n")

        summary_stats_dissect_5neighbours_file.write("No intersection.")
        log_file.write(
            "No intersection between DISSECT lemmas with top 5 neighbours and Lexicon" + lexicon + "." + "\n")

    summary_stats_dissect_5neighbours_file.close()

if second_evaluation_approach_yes == "yes":
    print(
        "---------------------------------------------------------\nSecond evaluation approach: distances in Lexicon/DISSECT spaces between synonyms/neighbours\n-------------------------------------------------")
    log_file.write(
        "---------------------------------------------------------\nSecond evaluation approach: distances in Lexicon/DISSECT spaces between synonyms/neighbours\n-------------------------------------------------" + "\n")

    # --------------------------------------------------------------------------------------------
    # 2.	Distances in Lexicon vs DISSECT space
    # a.	Calculate distance between pairs of shared lemmas that are synonyms in Lexicon space
    # b.	Calculate distance between pairs of shared lemmas that are neighbours in DISSECT space
    # c.	For each pair, compare distance in Lexicon space with distance in DISSECT space
    # --------------------------------------------------------------------------------------------

    # calculate cosine distances in Lexicon space between pairs of shared lemmas that are synonyms in Lexicon space:

    lexicon_distances_shared_lexicon_synonyms = list()  # cosine distances in the Lexicon space between pairs of Lexicon synonyms

    # calculate cosine distances in DISSECT space between pairs of shared lemmas that are synonyms in Lexicon space:

    dissect_distances_shared_lexicon_synonyms = list()

    # loop over all pairs of Lexicon synonyms and collect Lexicon/DISSECT distances between them:

    log_file.write("Calculating Lexicon and DISSECT distances between pairs of Lexicon synonyms..." + "\n")

    # log_file.write("lexicon_dissect_5neighbours:", str(lexicon_dissect_5neighbours) + "\n")

    dissect_lexicon_distances_synonyms_file = open(os.path.join(dir_out, dissect_lexicon_distances_synonyms_file_name),
                                                   'w')
    dissect_lexicon_distances_synonyms_file.write(
        "synonym1\tLexicon_id1\tsynonym2\tLexicon_id2\tLexicon_distance\tDISSECT_distance\n")

    # loop over all pairs of shared lemmas:

    count_n = 0
    for (lemma1, lemma2) in lexicon_cooccurrence:

        count_n += 1

        if (istest == "yes" and count_n < lines_read_testing) or istest == "no" and \
                        lemma1 in lexicon_dissect_5neighbours and lemma2 in lexicon_dissect_5neighbours:
            id1 = lexicon_lemma2id[lemma1]
            id2 = lexicon_lemma2id[lemma2]
            norm_lemma1 = norm(lexicon_coordinates[lemma1])
            norm_lemma2 = norm(lexicon_coordinates[lemma2])

            lexicon_cos_distance_lemma1_lemma2 = None
            dissect_cos_distance_lemma1_lemma2 = None

            if count_n % 500000 == 0:
                log_file.write(
                    "Consider synonyms" + lemma1 + " and " + lemma2 + "Lexicon IDs are" + str(id1) + "and" + str(
                        id2) + "\n")

            if (lemma1 is not lemma2) and (id1 < id2) and norm_lemma1 > 0 and norm_lemma2 > 0:

                try:
                    lexicon_cos_distance_lemma1_lemma2 = distance.cosine(lexicon_coordinates[lemma1],
                                                                         lexicon_coordinates[lemma2])

                except:
                    log_file.write("Count " + str(count_n) + " out of " + str(len(lexicon_cooccurrence)) + "\n")
                    log_file.write("lemma1: " + lemma1 + "\n")
                    log_file.write("lemma2: " + lemma2 + "\n")
                    log_file.write("Lexicon non-zero coordinates for lemma1:" + "\n")
                    l1_c = lexicon_coordinates[lemma1]
                    log_file.write(str([(i, lexicon_lemma2id_keys[i], e) for i, e in enumerate(l1_c) if e != 0]) + "\n")
                    log_file.write("Lexicon non-zero coordinates for lemma2:" + "\n")
                    l1_c = lexicon_coordinates[lemma2]
                    log_file.write(str([(i, lexicon_lemma2id_keys[i], e) for i, e in enumerate(l1_c) if e != 0]) + "\n")
                    log_file.write("Error with Lexicon cosine distance!\n")

                if not np.isfinite(lexicon_coordinates[lemma1]).all() or not np.isfinite(
                        lexicon_coordinates[lemma2]).all():
                    log_file.write("Count " + str(count_n) + " out of " + str(len(lexicon_cooccurrence)) + "\n")
                    log_file.write("lemma1: " + lemma1 + "\n")
                    log_file.write("lemma2: " + lemma2 + "\n")
                    log_file.write("Lexicon non-zero coordinates for lemma1:" + "\n")
                    l1_c = lexicon_coordinates[lemma1]
                    log_file.write(str([(i, lexicon_lemma2id_keys[i], e) for i, e in enumerate(l1_c) if e != 0]) + "\n")
                    log_file.write("Lexicon non-zero coordinates for lemma2:" + "\n")
                    l1_c = lexicon_coordinates[lemma2]
                    log_file.write(str([(i, lexicon_lemma2id_keys[i], e) for i, e in enumerate(l1_c) if e != 0]) + "\n")
                    # cos_sim = dot(a, b) / (norm(a) * norm(b))
                    log_file.write("NAN, INF, or NINF Error with Lexicon cosine distance!\n")

                if norm(lexicon_coordinates[lemma1]) == 0 or norm(lexicon_coordinates[lemma1]) == 0:
                    log_file.write("Count " + str(count_n) + " out of " + str(len(lexicon_cooccurrence)) + "\n")
                    log_file.write("lemma1: " + lemma1 + "\n")
                    log_file.write("lemma2: " + lemma2 + "\n")
                    log_file.write("Lexicon non-zero coordinates for lemma1:" + "\n")
                    l1_c = lexicon_coordinates[lemma1]
                    log_file.write(str([(i, lexicon_lemma2id_keys[i], e) for i, e in enumerate(l1_c) if e != 0]) + "\n")
                    log_file.write("Lexicon non-zero coordinates for lemma2:" + "\n")
                    l1_c = lexicon_coordinates[lemma2]
                    log_file.write(str([(i, lexicon_lemma2id_keys[i], e) for i, e in enumerate(l1_c) if e != 0]) + "\n")
                    log_file.write(
                        "Norm of lexicon_coordinates[lemma1]: " + str(norm(lexicon_coordinates[lemma1])) + "\n")
                    log_file.write(
                        "Norm of lexicon_coordinates[lemma2]: " + str(norm(lexicon_coordinates[lemma2])) + "\n")
                    # cos_sim = dot(a, b) / (norm(a) * norm(b))
                    log_file.write("Zeros! Error with Lexicon cosine distance!\n")

                # log_file.write("DISSECT coordinates for lemma1:", str(dissect_lemma2coordinates[lemma1][0:4]) + "\n")
                # log_file.write("DISSECT coordinates for lemma2:", str(dissect_lemma2coordinates[lemma2][0:4]) + "\n")
                try:
                    dissect_cos_distance_lemma1_lemma2 = distance.cosine(dissect_lemma2coordinates[lemma1],
                                                                         dissect_lemma2coordinates[lemma2])

                except:
                    log_file.write("Count " + str(count_n) + " out of " + str(len(lexicon_cooccurrence)) + "\n")
                    log_file.write("lemma1: " + lemma1 + "\n")
                    log_file.write("lemma2: " + lemma2 + "\n")
                    # log_file.write("DISSECT coordinates for lemma1:" + str(dissect_lemma2coordinates[lemma1]) + "\n")
                    # log_file.write("DISSECT coordinates for lemma2:" + str(dissect_lemma2coordinates[lemma2]) + "\n")
                    log_file.write("Error with DISSECT cosine distance!\n")

                if lemma1 == "ἐπιμέλεια" and lemma2 == "κομιδή":
                    log_file.write("\tTest!" + "\n")
                    log_file.write(
                        "\tSynonyms " + lemma1 + " and " + lemma2 + ", Lexicon IDs are " + str(id1) + " and " + str(
                            id2) + "\n")
                    if lemma1 in lexicon_dissect_5neighbours and lemma2 in lexicon_dissect_5neighbours:
                        log_file.write("\tThey are shared!" + "\n")
                        if id1 < id2:
                            log_file.write("\t and id1 < id2!" + "\n")
                        log_file.write(
                            "DISSECT coordinates for lemma1:" + str(dissect_lemma2coordinates[lemma1]) + "\n")
                        log_file.write(
                            "DISSECT coordinates for lemma2:" + str(dissect_lemma2coordinates[lemma2]) + "\n")
                        log_file.write(" Lexicon cosine distance: " + str(
                            lexicon_cos_distance_lemma1_lemma2) + " DISSECT cosine distance: " + str(
                            dissect_cos_distance_lemma1_lemma2) + "\n")

                if count_n % 10000 == 0:
                    if vm != "yes":
                        log_file.write("Count " + str(locale.format("%d", count_n, grouping=True)) +
                                   " out of " + str(locale.format("%d", len(lexicon_cooccurrence), grouping=True)) +
                                   " synonyms " + lemma1 +
                                   " and " + lemma2 + ", Lexicon IDs are " + str(id1) + " and " + str(id2) +
                                   ", Lexicon cosine distance: " + str(lexicon_cos_distance_lemma1_lemma2) +
                                   ", DISSECT cosine distance:" + str(dissect_cos_distance_lemma1_lemma2) + "\n")
                        print("Count " + str(locale.format("%d", count_n, grouping=True)) +
                          " out of " + str(locale.format("%d", len(lexicon_cooccurrence), grouping=True)) +
                          " synonyms " + lemma1 +
                          " and " + lemma2 + ", Lexicon IDs are " + str(id1) + " and " + str(id2) +
                          ", Lexicon cosine distance: " + str(lexicon_cos_distance_lemma1_lemma2) +
                          ", DISSECT cosine distance:" + str(dissect_cos_distance_lemma1_lemma2) + "\n")
                    else:
                        log_file.write("Count " + str(count_n) +
                                       " out of " + str(len(lexicon_cooccurrence)) +
                                       " synonyms " + lemma1 +
                                       " and " + lemma2 + ", Lexicon IDs are " + str(id1) + " and " + str(id2) +
                                       ", Lexicon cosine distance: " + str(lexicon_cos_distance_lemma1_lemma2) +
                                       ", DISSECT cosine distance:" + str(dissect_cos_distance_lemma1_lemma2) + "\n")
                        print("Count " + str(count_n) +
                              " out of " + str(len(lexicon_cooccurrence)) +
                              " synonyms " + lemma1 +
                              " and " + lemma2 + ", Lexicon IDs are " + str(id1) + " and " + str(id2) +
                              ", Lexicon cosine distance: " + str(lexicon_cos_distance_lemma1_lemma2) +
                              ", DISSECT cosine distance:" + str(dissect_cos_distance_lemma1_lemma2) + "\n")

                if lemma1 == "ἐπιμέλεια" and lemma2 == "κομιδή":
                    log_file.write("\tTest!" + "\n")
                    log_file.write(
                        "\tSynonyms " + lemma1 + " and " + lemma2 + ", Lexicon IDs are " + str(id1) + " and " + str(
                            id2) +
                        " Lexicon cosine distance: " + str(
                            lexicon_cos_distance_lemma1_lemma2) + " DISSECT cosine distance: " + str(
                            dissect_cos_distance_lemma1_lemma2) + "\n")

                if lexicon_cos_distance_lemma1_lemma2 != "NA" and dissect_cos_distance_lemma1_lemma2 != "NA":
                    lexicon_distances_shared_lexicon_synonyms.append(lexicon_cos_distance_lemma1_lemma2)
                    dissect_distances_shared_lexicon_synonyms.append(dissect_cos_distance_lemma1_lemma2)

                    dissect_lexicon_distances_synonyms_file.write(
                        lemma1 + "\t" + id1 + "\t" + lemma2 + "\t" + id2 + "\t" + str(
                            lexicon_cos_distance_lemma1_lemma2) +
                        "\t" + str(dissect_cos_distance_lemma1_lemma2) + "\n")

    log_file.write("lexicon_distances_shared_lexicon_synonyms:" + str(lexicon_distances_shared_lexicon_synonyms) + "\n")
    log_file.write("dissect_distances_shared_lexicon_synonyms:" + str(dissect_distances_shared_lexicon_synonyms) + "\n")

    dissect_lexicon_distances_synonyms_file.close()

    # calculate cosine distances in Lexicon space between pairs of shared lemmas that are neighbours in DISSECT space:
    # calculate cosine distances in DISSECT space between pairs of shared lemmas that are neighbours in DISSECT space:

    log_file.write("Calculating Lexicon and DISSECT distances between pairs of DISSECT neighbours..." + "\n")

    dissect_lexicon_distances_neighbours_file = open(
        os.path.join(dir_out, dissect_lexicon_distances_neighbours_file_name), 'w')
    dissect_lexicon_distances_neighbours_file.write(
        "lemma\tLexicon_id1\tneighbour\tLexicon_id2\tLexicon_distance\tDISSECT_distance\n")

    lexicon_distances_shared_dissect_neighbours = list()  # cosine distances in the Lexicon space between pairs of DISSECT neighbours
    dissect_distances_shared_dissect_neighbours = list()  # cosine distances in the Lexicon space between pairs of DISSECT neighbours

    # loop over all pairs of lemma and its neighbours:


    count_n = 0
    for (lemma, neighbour) in lemma_neighbour2distance:

        count_n += 1

        if count_n % 10000 == 0:
            if vm != "yes":
                log_file.write("Count " + str(locale.format("%d", count_n, grouping=True)) +
                           " out of " + str(locale.format("%d", len(lemma_neighbour2distance), grouping=True)) +
                           "Consider lemma+neighbour " + lemma + " and " + neighbour + "\n")
                print("Count " + str(locale.format("%d", count_n, grouping=True)) +
                  " out of " + str(locale.format("%d", len(lemma_neighbour2distance), grouping=True)) +
                  "Consider lemma+neighbour " + lemma + " and " + neighbour + "\n")
            else:
                log_file.write("Count " + str(count_n) +
                               " out of " + str(len(lemma_neighbour2distance)) +
                               "Consider lemma+neighbour " + lemma + " and " + neighbour + "\n")
                print("Count " + str(count_n) +
                      " out of " + str(len(lemma_neighbour2distance)) +
                      "Consider lemma+neighbour " + lemma + " and " + neighbour + "\n")

        if (lemma in lexicon_dissect_5neighbours) and (neighbour in lexicon_dissect_5neighbours):

            id1 = lexicon_lemma2id[lemma]
            id2 = lexicon_lemma2id[neighbour]

            norm_lemma = norm(lexicon_coordinates[lemma])
            norm_neighbour = norm(lexicon_coordinates[neighbour])

            # if count_n % 100 == 0:
            # log_file.write(
            #    "Consider shared lemma+neighbour " + lemma + " and " + neighbour + " Lexicon IDs are: " + str(id1) + " and " + str(
            #        id2) + "\n")

            if (lemma is not neighbour) and (id1 < id2) and norm_lemma > 0 and norm_neighbour > 0:

                # Lexicon distance:

                try:
                    lexicon_cos_distance_lemma_neighbour = distance.cosine(lexicon_coordinates[lemma],
                                                                           lexicon_coordinates[neighbour])
                    lexicon_distances_shared_dissect_neighbours.append(lexicon_cos_distance_lemma_neighbour)
                    dissect_lexicon_distances_neighbours_file.write(
                        lemma + "\t" + id1 + "\t" + neighbour + "\t" + id2 + "\t" + str(
                            lexicon_cos_distance_lemma_neighbour))
                except:
                    log_file.write("Count " + str(count_n) + " out of " + str(len(lemma_neighbour2distance)) + "\n")
                    log_file.write("lemma: " + lemma + "\n")
                    log_file.write("neighbour: " + neighbour + "\n")
                    log_file.write("Lexicon coordinates for lemma:" + str(lexicon_coordinates[lemma]) + "\n")
                    log_file.write("Lexicon coordinates for neighbour:" + str(lexicon_coordinates[neighbour]) + "\n")
                    log_file.write("Error with Lexicon cosine distance!\n")

                # DISSECT distance:
                try:
                    dissect_cos_distance_lemma_neighbour = lemma_neighbour2distance[lemma, neighbour]
                    dissect_lexicon_distances_neighbours_file.write(
                        "\t" + str(dissect_cos_distance_lemma_neighbour) + "\n")
                except:
                    log_file.write("Count " + str(count_n) + " out of " + str(len(lemma_neighbour2distance)) + "\n")
                    log_file.write("lemma: " + lemma + ", id: " + str(lexicon_lemma2id[lemma]) + "\n")
                    log_file.write("neighbour: " + neighbour + ", id: " + str(lexicon_lemma2id[neighbour]) + "\n")
                    # log_file.write("DISSECT distance:" + str(lemma_neighbour2distance[lemma, neighbour]) + "\n")
                    log_file.write("Error with DISSECT cosine distance!\n")

                if count_n % 10000 == 0:
                    log_file.write(str(count_n) + "out of " + str(
                        len(lemma_neighbour2distance)) + " consider neighbours " + lemma + " and " +
                                   neighbour + "Lexicon IDs are " + str(id1) + " and " + str(id2) + "\n")
                    log_file.write("Lexicon cosine distance: " + str(lexicon_cos_distance_lemma_neighbour) + "\n")
                    log_file.write("DISSECT cosine distance: " + str(dissect_cos_distance_lemma_neighbour) + "\n")

                if lemma == "ἐπιμέλεια" or lemma == "κομιδή":
                    log_file.write("\tTest!" + "\n")
                    log_file.write(
                        "\tConsider neighbours " + lemma + " and " + neighbour + ", Lexicon IDs are " + str(
                            id1) + " and " + str(
                            id2) + "\n")
                    log_file.write("\tLexicon cosine distance: " + str(lexicon_cos_distance_lemma_neighbour) + "\n")
                    log_file.write("\tDISSECT cosine distance: " + str(dissect_cos_distance_lemma_neighbour) + "\n")

                dissect_distances_shared_dissect_neighbours.append(dissect_cos_distance_lemma_neighbour)

    # log_file.write("lexicon_distances_shared_dissect_neighbours:" + str(lexicon_distances_shared_dissect_neighbours) + "\n")
    # log_file.write("dissect_distances_shared_dissect_neighbours:" + str(dissect_distances_shared_dissect_neighbours) + "\n")

    dissect_lexicon_distances_neighbours_file.close()

    # print out summary results:

    summary_dissect_lexicon_distances_file = open(os.path.join(dir_out, summary_dissect_lexicon_distances_file_name),
                                                  'w')

    # Synonyms:

    summary_dissect_lexicon_distances_file.write("lexicon_distances_shared_lexicon_synonyms and ")
    summary_dissect_lexicon_distances_file.write("dissect_distances_shared_lexicon_synonyms: \n")
    log_file.write("length of lexicon_distances_shared_lexicon_synonyms: " + str(
        len(lexicon_distances_shared_lexicon_synonyms)) + "\n")
    # summary_dissect_lexicon_distances_file.write(str(lexicon_distances_shared_lexicon_synonyms) + "\n")
    log_file.write(
        "length of dissect_distances_shared_lexicon_synonyms: " + str(
            len(dissect_distances_shared_lexicon_synonyms)) + "\n")
    # summary_dissect_lexicon_distances_file.write(str(dissect_distances_shared_lexicon_synonyms) + "\n")

    # Pearson's correlation coefficient:
    corr_p, p_value_p = pearsonr(lexicon_distances_shared_lexicon_synonyms, dissect_distances_shared_lexicon_synonyms)
    log_file.write("Pearson's correlation: " + str(corr_p) + ", " + str(p_value_p) + "\n")
    summary_dissect_lexicon_distances_file.write(
        "Pearson's correlation: " + str(corr_p) + ", p-value: " + str(p_value_p) + "\n")

    # Spearman's correlation coefficient:
    corr_s, p_value_s = spearmanr(lexicon_distances_shared_lexicon_synonyms, dissect_distances_shared_lexicon_synonyms)
    log_file.write("Spearmnan's correlation: " + str(corr_s) + ", " + str(p_value_s) + "\n")
    summary_dissect_lexicon_distances_file.write(
        "Spearman's correlation: " + str(corr_s) + ", p-value:" + str(p_value_s) + "\n")

    # Neighbours:

    log_file.write(
        "length of lexicon_distances_shared_dissect_neighbours: " + str(
            len(lexicon_distances_shared_dissect_neighbours)) + "\n")
    summary_dissect_lexicon_distances_file.write("lexicon_distances_shared_dissect_neighbours and ")
    # summary_dissect_lexicon_distances_file.write(str(lexicon_distances_shared_dissect_neighbours) + "\n")

    log_file.write("length of dissect_distances_shared_dissect_neighbours: " + str(
        len(dissect_distances_shared_dissect_neighbours)) + "\n")
    summary_dissect_lexicon_distances_file.write("dissect_distances_shared_dissect_neighbours: \n")
    # summary_dissect_lexicon_distances_file.write(str(dissect_distances_shared_dissect_neighbours) + "\n")

    # Pearson's correlation coefficient:
    corr_p, p_value_p = pearsonr(lexicon_distances_shared_dissect_neighbours,
                                 dissect_distances_shared_dissect_neighbours)
    log_file.write("Pearson's correlation: " + str(corr_p) + ", " + str(p_value_p) + "\n")
    summary_dissect_lexicon_distances_file.write(
        "Pearson's correlation: " + str(corr_p) + ", p-value:" + str(p_value_p) + "\n")

    # Spearman's correlation coefficient:
    corr_s, p_value_s = spearmanr(lexicon_distances_shared_dissect_neighbours,
                                  dissect_distances_shared_dissect_neighbours)
    log_file.write("Spearmnan's correlation: " + str(corr_s) + str(p_value_s) + "\n")
    summary_dissect_lexicon_distances_file.write(
        "Spearman's correlation: " + str(corr_s) + ", p-value:" + str(p_value_s) + "\n")

    summary_dissect_lexicon_distances_file.close()

if third_evaluation_approach_yes:

    print(
        "---------------------------------------------------------\nThird evaluation approach: overlap between synsets in Lexicon and neighbour sets in DISSECT spaces\n-------------------------------------------------")
    log_file.write(
        "---------------------------------------------------------\nThird evaluation approach: overlap between synsets in Lexicon and neighbour sets in DISSECT spaces\n-------------------------------------------------" + "\n")

    # -------------------------------------------------------------------------------------
    # 3.	Synonym/neighbour comparison
    # a.	Compare overlap between synsets in Lexicon space and neighbours in DISSECT space
    # -------------------------------------------------------------------------------------

    # Define synsets in Lexicon space containing shared lemmas:

    log_file.write("Define lemma-to-synset mapping" + "\n")

    lemma2synset = dict()  # maps a shared lemma to the list of its Lexicon synonyms:

    count_n = 0
    for synset_id in synsets:

        count_n += 1

        if count_n % 1000 == 0:
            log_file.write(str(count_n) + synset_id + " out of " + str(len(synsets)) + "\n")

        synonyms = synsets[synset_id]
        if synset_id == "02078399-a":
            log_file.write("\tTest!" + "\n")
            log_file.write("\tSynset_id: " + str(synset_id) + "\n")
            log_file.write("\tSynonyms:" + str(synonyms) + "\n")

        for lemma in synonyms:

            if lemma in lexicon_dissect_5neighbours:
                synonyms_lemma = synonyms
                # synonyms_lemma.remove(lemma)
                lemma2synset[lemma] = synonyms_lemma
                # lemma2synset[lemma].remove(lemma)

            if lemma in ["κομιδή", "ἐπιμέλεια", "ἄατος"]:
                log_file.write("\tTest!" + "\n")
                log_file.write("\tLemma: " + lemma + "\n")
                log_file.write("\tSynonyms:" + str(synonyms) + "\n")
                log_file.write("\tSynonyms without lemma:" + str(synonyms_lemma) + "\n")

            if synset_id == "02078399-a":
                log_file.write("\tTest!" + "\n")
                log_file.write("\tSynset_id: " + synset_id + "\n")
                log_file.write("\tLemma: " + lemma + "\n")
                log_file.write("\tSynonyms:" + str(synonyms) + "\n")
                if lemma in lexicon_dissect_5neighbours:
                    log_file.write("\tlemma is shared\n")

            if lemma in ["κομιδή", "ἐπιμέλεια", "ἄατος"]:
                log_file.write("\tTest!" + "\n")
                try:
                    log_file.write("\t " + synset_id + ", lemma:" + lemma + ", its synonyms: " +
                                   str(lemma2synset[lemma]) + "\n")
                except:
                    pass

    try:
        log_file.write("Examples:" + "\n")
        log_file.write("Synonyms of κομιδή:  " + str(lemma2synset["κομιδή"]) + "\n")
        log_file.write("Synonyms of ἐπιμέλεια:  " + str(lemma2synset["ἐπιμέλεια"]) + "\n")
        log_file.write("Synonyms of ἄατος:  " + str(lemma2synset["ἄατος"]) + "\n")
    except:
        pass

    # Define sets of neighbours in DISSECT space:

    log_file.write("Define lemma-to-neighbourset mapping" + "\n")

    lemma2neighbourset = dict()  # maps a shared lemma to the list of its DISSECT neighbours:

    count_n = 0
    for lemma, neighbour in lemma_neighbour2distance:
        neighbours = list()
        count_n += 1

        if lemma in lexicon_dissect_5neighbours:

            if count_n % 1000 == 0:
                log_file.write(
                    str(count_n) + "lemma: " + lemma + " out of " + str(len(lemma_neighbour2distance)) + "\n")

            if lemma in lemma2neighbourset:
                neighbours_this_lemma = lemma2neighbourset[lemma]
                neighbours_this_lemma.append(neighbour)
                lemma2neighbourset[lemma] = neighbours_this_lemma
            else:
                lemma2neighbourset[lemma] = [neighbour]

            if lemma in ["κομιδή", "ἐπιμέλεια", "ἄατος"]:
                log_file.write("\tTest!" + "\n")
                log_file.write("\t " + lemma + str(lemma2neighbourset[lemma]) + "\n")

    try:
        log_file.write("Examples:" + "\n")
        log_file.write("Neighbours of κομιδή:  " + str(lemma2neighbourset["κομιδή"]) + "\n")
        log_file.write("Neighbours of ἐπιμέλεια:  " + str(lemma2neighbourset["ἐπιμέλεια"]) + "\n")
        log_file.write("Neighbours of ἄατος:  " + str(lemma2neighbourset["ἄατος"]) + "\n")
    except:
        pass

    # Calculate overlap between synsets and neighbour sets:

    log_file.write("Finding overlap between synsets and neighboursets" + "\n")

    count_n = 0

    # print out results:

    summary_dissect_lexicon_overlap_file = open(os.path.join(dir_out, summary_dissect_lexicon_overlap_file_name), 'w')

    summary_dissect_lexicon_overlap_file.write("lemma\tsynset\tneighbourset\toverlap\tprecision\trecall\n")

    for lemma in lexicon_dissect_5neighbours:
        count_n += 1
        synset = []
        neighbourset = []
        overlap = []
        if count_n % 100 == 0:
            log_file.write(str(count_n) + "lemma: " + lemma + "\n")

        try:
            synset = list(set(lemma2synset[lemma]))
            synset.remove(lemma)
            synset = set(lemma2synset[lemma])
            neighbourset = set(lemma2neighbourset[lemma])
            overlap = list(synset.intersection(neighbourset))
            precision = len(overlap)/float(len(neighbourset))
            recall = len(overlap) / float(len(synset))
            if count_n % 100 == 0:
                log_file.write(str(count_n) + "lemma: " + lemma + "\n")
                log_file.write("synset: " + str(synset) + "\n")
                log_file.write("neighbourset: " + str(neighbourset) + "\n")
                log_file.write("overlap: " + str(overlap) + "\n")
                log_file.write("precision: " + str(precision) + "\n")
                log_file.write("recall: " + str(recall) + "\n")
        except:
            log_file.write("error for lemma " + lemma + "\n")

        lemma2precision[lemma] = precision
        lemma2recall[lemma] = recall
        summary_dissect_lexicon_overlap_file.write(lemma+"\t"+str(synset)+"\t"+str(neighbourset)+"\t"+str(overlap)+
        "\t"+str(precision)+"\t"+str(recall)+"\n")
        precisions.append(precision)
        recalls.append(recall)

        if lemma in ["κομιδή", "ἐπιμέλεια", "ἄατος"]:
            log_file.write("\tTest!" + "lemma: " + lemma + "\n")
            try:
                synset = set(lemma2synset[lemma])
                synset.remove(lemma)
                log_file.write("\tsynset: " + str(synset) + "\n")
                try:
                    neighbourset = set(lemma2neighbourset[lemma])
                    log_file.write("\tneighbourset: " + str(neighbourset) + "\n")
                    try:
                        synset = set(lemma2synset[lemma])
                        neighbourset = set(lemma2neighbourset[lemma])
                        overlap = list(synset.intersection(neighbourset))
                        precision = len(overlap) / float(len(neighbourset))
                        recall = len(overlap) / float(len(synset))
                        log_file.write("\toverlap: " + str(overlap) + "\n")
                        try:
                            log_file.write("\tprecision: " + str(len(overlap)) + "/" + str(len(neighbourset)) + "=" +
                                           str(len(overlap) / len(neighbourset)) + "=" + str(precision) + "\n")
                            log_file.write("\t " + lemma + str(lemma2precision[lemma]) + "\n")
                            log_file.write("\trecall: " + str(len(overlap)) + "/" + str(len(synset)) + "=" +
                                           str(len(overlap) / len(synset)) + "=" + str(recall) + "\n")
                            log_file.write("\t " + lemma + str(lemma2recall[lemma]) + "\n")
                        except:
                            log_file.write("No precision/recall for lemma " + lemma + "\n")
                    except:
                        log_file.write("No overlap for lemma " + lemma + "\n")
                except:
                    log_file.write("No neighbourset for lemma " + lemma + "\n")
            except:
                log_file.write("No synset for lemma " + lemma + "\n")

    try:
        log_file.write("Examples:" + "\n")
        log_file.write("Precision for κομιδή: " + str(lemma2precision["κομιδή"]) + "\n")
        log_file.write("Recall for κομιδή: " + str(lemma2recall["κομιδή"]) + "\n")
        log_file.write("Precision for ἐπιμέλεια: " + str(lemma2precision["ἐπιμέλεια"]) + "\n")
        log_file.write("Recall for ἐπιμέλεια: " + str(lemma2recall["ἐπιμέλεια"]) + "\n")
        log_file.write("Precision for ἄατος: " + str(lemma2precision["ἄατος"]) + "\n")
        log_file.write("Recall for ἄατος: " + str(lemma2recall["ἄατος"]) + "\n")
    except:
        pass


    precisions = np.array(precisions, dtype=np.float32)

    mean_precisions = np.mean(precisions)
    std_precisions = precisions.std()
    min_precisions = precisions.min()
    max_precisions = precisions.max()
    median_precisions = np.median(precisions)
    perc25_precisions = np.percentile(precisions, 25)
    perc75_precisions = np.percentile(precisions, 75)

    summary_dissect_lexicon_overlap_file.write("Mean of precision:" + str(mean_precisions) + "\n")
    summary_dissect_lexicon_overlap_file.write("STD of precision:" + str(std_precisions) + "\n")
    summary_dissect_lexicon_overlap_file.write("Min of precision:" + str(min_precisions) + "\n")
    summary_dissect_lexicon_overlap_file.write("Max of precision:" + str(max_precisions) + "\n")
    summary_dissect_lexicon_overlap_file.write("Median of precision:" + str(median_precisions) + "\n")
    summary_dissect_lexicon_overlap_file.write("25th percentile of precision:" + str(perc25_precisions) + "\n")
    summary_dissect_lexicon_overlap_file.write("75th percentile of precision:" + str(perc75_precisions) + "\n")

    recalls = np.array(recalls, dtype=np.float32)

    mean_recalls = np.mean(recalls)
    std_recalls = recalls.std()
    min_recalls = recalls.min()
    max_recalls = recalls.max()
    median_recalls = np.median(recalls)
    perc25_recalls = np.percentile(recalls, 25)
    perc75_recalls = np.percentile(recalls, 75)

    summary_dissect_lexicon_overlap_file.write("Mean of recall:" + str(mean_recalls) + "\n")
    summary_dissect_lexicon_overlap_file.write("STD of recall:" + str(std_recalls) + "\n")
    summary_dissect_lexicon_overlap_file.write("Min of recall:" + str(min_recalls) + "\n")
    summary_dissect_lexicon_overlap_file.write("Max of recall:" + str(max_recalls) + "\n")
    summary_dissect_lexicon_overlap_file.write("Median of recall:" + str(median_recalls) + "\n")
    summary_dissect_lexicon_overlap_file.write("25th percentile of recall:" + str(perc25_recalls) + "\n")
    summary_dissect_lexicon_overlap_file.write("75th percentile of recall:" + str(perc75_recalls) + "\n")

    summary_dissect_lexicon_overlap_file.close()

now = datetime.datetime.now()
log_file.write("Finished." + "\n")
log_file.write(str(now.strftime("%Y-%m-%d %H:%M")) + "\n")

log_file.close()
