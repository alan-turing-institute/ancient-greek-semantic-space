## -*- coding: utf-8 -*-
# Author: Barbara McGillivray
# Date: 11/03/2019
# Python version: 3
# Script version: 1.0
# Script for processing the data from Ancient Greek WordNet and prepare the data for the evaluation of the semantic space.


# ----------------------------
# Initialization
# ----------------------------


# Import modules:

import os
import csv
from collections import defaultdict
import numpy as np
from scipy.spatial import distance

# Default parameters:

window_default = 5
freq_threshold_default = 2
istest_default = "yes"
skip_read_files_default = "yes"

# Parameters:

window = input("What is the window size? Leave empty for default (" + str(window_default) + ").") # 5 or 10 # size of context window for co-occurrence matrix
freq_threshold = input("What is the frequency threshold for vocabulary lemmas? Leave empty for default (" +
                       str(freq_threshold_default) + ").")# 2 or 50 # frequency threshold for lemmas in vocabulary
istest = input("Is this a test? Leave empty for default (" + str(istest_default) + ").")
skip_read_files = input("Do you want to skip the first step that read input files? Leave empty for default (" + str(skip_read_files_default) + ").")
lines_read_testing = 10 # lines read in test case

if window == "":
    window = window_default

if freq_threshold == "":
    freq_threshold = freq_threshold_default

if istest == "":
    istest = istest_default

if skip_read_files == "":
    skip_read_files = skip_read_files_default

# Directory and file names:

directory = os.path.join("/Users", "bmcgillivray", "Documents", "OneDrive", "The Alan Turing Institute",
                         "Martina Astrid Rodda - MAR dphil project")
dir_wn = os.path.join(directory, "TAL paper", "wordnet", "Open Ancient Greek WordNet 0.5")
dir_ss = os.path.join(directory, "semantic space", "sem_space_output")
dir_out = os.path.join(directory, "Evaluation", "input")

# create output directory if it doesn't exist:
if not os.path.exists(os.path.join(directory, "Evaluation", "input")):
    os.makedirs(os.path.join(directory, "Evaluation", "input"))

# Input files:

wn_file_name = "wn-data-grc.tab"
ss_file_name = "CORE_SS.matrix_w" + str(window) + "_t" + str(freq_threshold) + ".ppmi.svd_300.dm"
neighbours_file_name = "NEIGHBOURS.matrix_w" + str(window) + "_t" + str(freq_threshold) + ".rows.CORE_SS.matrix_w" \
                       + str(window) + "_t" + str(freq_threshold) + ".ppmi.svd_300.cos"

# Output files:

file_out_wn_cooccurrence_name = "AGwordnet_co-occurrences.csv" # matrix with 0s and 1s depending on whether two lemmas co-occur in the same WN synset
file_out_dissect_5neighbour_name = "semantic-space_w" + str(window) + "_t" + str(freq_threshold) + "lemmas.txt"
file_out_shared_lemmas_name = "AGWN-semantic-space_w" + str(window) + "_t" + str(freq_threshold) + "_shared-lemmas.txt"
file_out_agwn_vocabulary_name = "AGWM_lemmas.txt"
file_out_dissect_distances_name = "semantic_space_w" + str(window) + "_t" + str(freq_threshold) + "distances.csv"

if istest == "yes":
    file_out_wn_cooccurrence_name = file_out_wn_cooccurrence_name.replace(".csv", "_test.csv")
    file_out_dissect_5neighbour_name = file_out_dissect_5neighbour_name.replace(".txt", "_test.txt")
    file_out_shared_lemmas_name = file_out_shared_lemmas_name.replace(".txt", "_test.txt")
    file_out_agwn_vocabulary_name = file_out_agwn_vocabulary_name.replace(".txt", "_test.txt")
    file_out_dissect_distances_name = file_out_dissect_distances_name.replace(".csv", "_test.csv")

# Initialize objects:

agwn_vocabulary = list() # list of all lemmas in AG WordNet
dissect_lemmas_5neighbours = list()  # list of lemmas in DISSECT space, for which we have the top 5 neighbours
agwn_dissect_5neighbours = list()  # list of lemmas shared between AGWN and DISSECT space with 5 neighbours
agwn_cooccurrence = defaultdict(dict) # multidimensional dictionary: maps each pair of lemmas to 1 if they co-occur in the same WN synset, and 0 otherwise
lemma2neighbour = defaultdict(dict) # maps a pair of lemmas and each of its top 5 DISSECT neighbours to their distance,
    # excluding the cases where the neighbour is the lemma itself
dissect_lemmas2coordinates = dict() # maps a lemma to the array of its coordinates in the DISSECT semantic space
dissect_lemmas2cosinedistance = defaultdict(dict) # maps a pair of lemmas to the cosine distance between them in the DISSECT semantic space

# --------------------------------------
# Read WordNet file to collect synsets
# --------------------------------------

if skip_read_files == "no":
    # Read wordnet file to count the number of its lines:

    wn_file = open(os.path.join(dir_wn, wn_file_name), 'r', encoding="UTF-8")

    row_count_wn = sum(1 for line in wn_file)

    wn_file.close()

    # Read synsets from WordNet file:

    count_n = 1 # counts lines read

    synsets = dict()  # maps a synset ID to the list of its contents
    synsetid2def = dict() # maps a synset ID to its English definition

    wn_file = open(os.path.join(dir_wn, wn_file_name), 'r', encoding="UTF-8")

    wn_reader = csv.reader(wn_file, delimiter = "\t")

    next(wn_reader) # This skips the first row of the CSV file

    synset_def = ""
    synset_lemma = ""

    for row in wn_reader:
        count_n += 1
        if ((istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_wn)) and (count_n > 1):

                print("WordNet: line", str(count_n), " out of ", str(row_count_wn))

                row[0] = row[0].replace("#", "")
                synset_id = row[0]
                if row[1] == "eng:def":
                    synset_def = row[3]
                    #print("Def:", synset_def)
                    synsetid2def[synset_id] = synset_def
                else:
                    synset_lemma = row[2]

                    if synset_id in synsets:
                        synsets_this_id = synsets[synset_id]
                        synsets_this_id.append(synset_lemma)
                        synsets[synset_id] = synsets_this_id
                    else:
                        synsets[synset_id] = [synset_lemma]

                    #print("\tSynset ID:", str(synset_id), "; lemma:", synset_lemma)


    wn_file.close()

    # --------------------------------------------------
    # Print co-occurrence matrix from WordNet synsets
    # -------------------------------------------------

    # create vocabulary list and co-occurrence pairs:

    for synset_id in synsets:
        print("Synset ID:", synset_id)
        print("Definition:", synsetid2def[synset_id])
        print("Lemmas:", str(synsets[synset_id]))
        synsets_this_lemma = synsets[synset_id]
        for lemma1 in synsets_this_lemma:
            #print("Defining co-occurrences of lemma1:", lemma1)
            if lemma1 not in agwn_vocabulary:
                agwn_vocabulary.append(lemma1)
                #print("Vocabulary so far:", str(wn_vocabulary))
            for lemma2 in synsets_this_lemma:
                #print("Lemma2:", lemma2)
                agwn_cooccurrence[lemma1][lemma2] = 1
                #print("Co-occurrence:", str(wn_cooccurrence[lemma1][lemma2]))
                #print("Lemma1:", lemma1, "Lemma2:", lemma2, "Co-occurrence:", str(agwn_cooccurrence[lemma1][lemma2]))

    agwn_vocabulary = sorted(agwn_vocabulary)

    with open(os.path.join(dir_out, file_out_agwn_vocabulary_name), 'w', encoding="UTF-8") as agwn_vocabulary_file:
        for lemma in agwn_vocabulary:
            agwn_vocabulary_file.write("%s\n" % lemma)

    #print("Vocabulary:", str(wn_vocabulary))

    # finish defining co-occurrence pairs:

    with open(os.path.join(dir_out, file_out_wn_cooccurrence_name), 'w', encoding="UTF-8") as file_out_wn_cooccurrence:

        for lemma1 in agwn_vocabulary:
            print("Printing Co-occurrences of lemma1:", lemma1)
            for lemma2 in agwn_vocabulary:
                #print("Co-occurrence of ", lemma1, "and", lemma2, ":")
                try:
                    #print("Co-occurrence of", lemma1, "and", lemma2, ":", str(agwn_cooccurrence[lemma1][lemma2]))
                    file_out_wn_cooccurrence.write("\t" + str(agwn_cooccurrence[lemma1][lemma2]))
                    #if lemma1 == "κτίσμα":
                    #    print("Co-occurrence of ", lemma1, "and", lemma2, ":", str(wn_cooccurrence[lemma1][lemma2]))
                except KeyError:
                    agwn_cooccurrence[lemma1][lemma2] = 0
                    #if lemma1 == "κτίσμα":
                    #    print("Co-occurrence of ", lemma1, "and", lemma2, ":", str(wn_cooccurrence[lemma1][lemma2]))
                    file_out_wn_cooccurrence.write("\t" + str(agwn_cooccurrence[lemma1][lemma2]))
            file_out_wn_cooccurrence.write("\n")


    # ---------------------------------------------
    # read data from DISSECT semantic space
    # ---------------------------------------------

    # neighbours in semantic space:

    neighbours_file = open(os.path.join(dir_ss, neighbours_file_name), 'r', encoding="UTF-8")
    row_count_neighbours = sum(1 for line in neighbours_file)
    neighbours_file.close()

    neighbours_file = open(os.path.join(dir_ss, neighbours_file_name), 'r', encoding="UTF-8")
    count_n = 0

    lemma = ""
    for line in neighbours_file:
        count_n +=1

        if ((istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_neighbours)):
            print("Reading neighbours in DISSECT, line", str(count_n))
            line = line.rstrip('\n')
            if not line.startswith("\t"):
                #print("lemma!", line)
                lemma = line
                dissect_lemmas_5neighbours.append(lemma)
            else:
                #print("neighbour!", line)
                fields = line.split("\t")
                #print("fields:", fields)
                neighbour_distance = fields[1]
                #print("neighbour_distance:", neighbour_distance)
                neighbour_distance_fields = neighbour_distance.split(" ")
                #print("neighbour_distance_fields:", neighbour_distance_fields)
                neighbour = neighbour_distance_fields[0]
                #print("neighbour:", neighbour)
                neighbour_distance = neighbour_distance_fields[1]
                #print("distance:", neighbour_distance)
                if neighbour != lemma:
                    lemma2neighbour[lemma][neighbour] = neighbour_distance


    neighbours_file.close()

    with open(os.path.join(dir_out, file_out_dissect_5neighbour_name), 'w', encoding="UTF-8") as file_out_dissect_5neighbour:
        for lemma in dissect_lemmas_5neighbours:
            file_out_dissect_5neighbour.write("%s\n" % lemma)

    #for lemma, neighbour in lemma2neighbour.items():
    #    print("lemma:", lemma)
    #    for n in neighbour:
    #        print("Corpus neighbour:", n, "distance:", neighbour[n])


    #-------------------------------------------------------------------------------
    # find lemmas present in both AGWN and DISSECT space (called “shared lemmas”)
    #-------------------------------------------------------------------------------

    agwn_dissect_5neighbours = list((set(dissect_lemmas_5neighbours).intersection(set(agwn_vocabulary))))
    print("There are", str(len(agwn_vocabulary)), "lemmas in AGWN, ", str(len(dissect_lemmas_5neighbours)),
          "lemmas in DISSECT space with 5 top neighbours", "and", str(len(agwn_dissect_5neighbours)),
          "in the intersection.")
    #There are 22828 lemmas in AGWN,  44740 lemmas in DISSECT space with 5 top neighbours and 12899 in the intersection.

    with open(os.path.join(dir_out, file_out_shared_lemmas_name), 'w', encoding="UTF-8") as out_shared_lemmas_file:
        for lemma in agwn_dissect_5neighbours:
            out_shared_lemmas_file.write("%s\n" % lemma)

    # read semantic space coordinates:

    ss_file = open(os.path.join(dir_ss, ss_file_name), 'r', encoding="UTF-8")
    row_count_ss = sum(1 for line in ss_file)
    ss_file.close()

    ss_file = open(os.path.join(dir_ss, ss_file_name), 'r', encoding="UTF-8")
    count_n = 0
    for line in ss_file:
        count_n += 1
        if ((istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_ss)):
            print("Reading DISSECT coordinates, line", str(count_n), ":", line)
            line = line.rstrip('\n')
            coordinates = line.split('\t')
            lemma = coordinates[0]
            coordinates.pop(0)
            dissect_lemmas2coordinates[lemma] = np.asarray(list(np.float_(coordinates)))

    ss_file.close()

    # calculate distance between pairs of lemmas in DISSECT space:

    count_n = 0
    with open(os.path.join(dir_out, file_out_dissect_distances_name), 'w',
              encoding="UTF-8") as out_dissect_distances_file:
        dissect_distances_writer = csv.writer(out_dissect_distances_file, delimiter="\t")

        dissect_distances_writer.writerow(["lemma1", "lemma2", "distance"])

        for lemma1 in dissect_lemmas2coordinates:
            count_n += 1
            print("Printing distances for", lemma1, ":", str(count_n), "out of", str(len(dissect_lemmas2coordinates.keys)))
            for lemma2 in dissect_lemmas2coordinates:
                #print("Printing distances between", lemma1, "and", lemma2, ":")
                #print(str(dissect_lemmas2coordinates[lemma1]))
                #print(str(type(dissect_lemmas2coordinates[lemma1])))
                #print(str(dissect_lemmas2coordinates[lemma2]))
                #print(str(type(dissect_lemmas2coordinates[lemma2])))
                #print(str(distance.cosine(dissect_lemmas2coordinates[lemma1], dissect_lemmas2coordinates[lemma2])))
                dissect_lemmas2cosinedistance[lemma1][lemma2] = \
                    distance.cosine(dissect_lemmas2coordinates[lemma1], dissect_lemmas2coordinates[lemma2])

                # print out distances between pairs of lemmas in DISSECT space:

                dissect_distances_writer.writerow(
                    [lemma1, lemma2, str(dissect_lemmas2cosinedistance[lemma1][lemma2])])


else:

    # read list of AGWN lemmas:

    agwn_vocabulary = [line.rstrip('\n') for line in open(os.path.join(dir_out, file_out_agwn_vocabulary_name), 'r', encoding="UTF-8")]

    # read list of DISSECT lemmas with top 5 neighbours:

    dissect_lemmas_5neighbours = [line.rstrip('\n') for line in open(os.path.join(dir_out, file_out_dissect_5neighbour_name), 'r', encoding="UTF-8")]

    # read list of shared lemmas between AGWN and DISSECT lemmas with top 5 neighbours:

    agwn_dissect_5neighbours = [line.rstrip('\n') for line in open(os.path.join(dir_out, file_out_shared_lemmas_name), 'r', encoding="UTF-8")]

    print("There are", str(len(agwn_vocabulary)), "lemmas in AGWN, ", str(len(dissect_lemmas_5neighbours)),
          "lemmas in DISSECT space with 5 top neighbours", "and", str(len(agwn_dissect_5neighbours)),
          "in the intersection.")

    # read mapping between pairs of AGWM lemmas and their co-occurrence value in AGWN:

    file_wn_cooccurrence = open(os.path.join(dir_out, file_out_wn_cooccurrence_name), 'r', encoding="UTF-8")
    row_count_wn_cooccurrence = sum(1 for line in file_wn_cooccurrence)
    file_wn_cooccurrence.close()

    file_wn_cooccurrence = open(os.path.join(dir_out, file_out_wn_cooccurrence_name), 'r', encoding="UTF-8")
    wn_cooccurrence_reader = csv.reader(file_wn_cooccurrence, delimiter = "\t")

    next(wn_cooccurrence_reader)

    lemma1 = ""
    lemma2 = ""
    coocc = ""

    count_n = 1

    for row in wn_cooccurrence_reader:
        count_n += 1
        if ((istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_wn_cooccurrence)):

            print("AGWN Co-occurrence: line", str(count_n), "out of", str(row_count_wn_cooccurrence))

            lemma1 = row[0]
            lemma2 = row[1]
            coocc = row[2]
            agwn_cooccurrence[lemma1][lemma2] = coocc

    file_wn_cooccurrence.close()

    # read mapping between pairs of lemmas and their cosine distance in the DISSECT semantic space:

    file_dissect_distances = open(os.path.join(dir_out, file_out_dissect_distances_name), 'r', encoding="UTF-8")
    row_count_dissect_distances = sum(1 for line in file_dissect_distances)
    file_dissect_distances.close()

    file_dissect_distances = open(os.path.join(dir_out, file_out_dissect_distances_name), 'r', encoding="UTF-8")
    dissect_distances_reader = csv.reader(file_dissect_distances, delimiter="\t")

    next(dissect_distances_reader)

    lemma1 = ""
    lemma2 = ""
    cos_distance = 0

    count_n = 1

    for row in dissect_distances_reader:
        count_n += 1
        if ((istest == "yes" and count_n < lines_read_testing) or (
                istest == "no" and count_n <= row_count_dissect_distances)):
            print("DISSECT distance: line", str(count_n), "out of", str(row_count_dissect_distances))

            lemma1 = row[0]
            lemma2 = row[1]
            cos_distance = float(row[2])
            dissect_lemmas2coordinates[lemma1][lemma2] = cos_distance

    file_dissect_distances.close()

# ---------------------------------------------

#1.	Distances in DISSECT space
#a.	Calculate distance between pairs of shared lemmas that are neighbours in DISSECT space
#b.	Compare this distance with average distance between pairs of lemmas in DISSECT space

# calculate average distance between pairs of lemmas in DISSECT space:

#....
#dissect_lemmas2coordinates

#for lemma1 in dissect_lemmas_5neighbours:
#    for lemma2 in dissect_lemmas_5neighbours:
