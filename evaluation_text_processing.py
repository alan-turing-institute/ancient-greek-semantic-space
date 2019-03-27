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
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# import seaborn; seaborn.set()  # set plot style

# Default parameters:

window_default = 5
freq_threshold_default = 2
istest_default = "yes"
skip_read_files_default = "yes"

# Parameters:

window = input("What is the window size? Leave empty for default (" + str(
    window_default) + ").")  # 5 or 10 # size of context window for co-occurrence matrix
freq_threshold = input("What is the frequency threshold for vocabulary lemmas? Leave empty for default (" +
                       str(freq_threshold_default) + ").")  # 2 or 50 # frequency threshold for lemmas in vocabulary
istest = input("Is this a test? Leave empty for default (" + str(istest_default) + ").")
skip_read_files = input("Do you want to skip the first step that read input files? Leave empty for default (" + str(
    skip_read_files_default) + "). NB Only skip if you've already created the files in a previous run (with the same parameters)!")
lines_read_testing = 10000  # lines read in test case

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
dir_ss = os.path.join(directory, "semantic_space", "sem_space_output")
dir_ss_rows = os.path.join(dir_ss, "ppmi_svd300")
dir_out = os.path.join(directory, "Evaluation", "input")

# create output directory if it doesn't exist:
if not os.path.exists(os.path.join(directory, "Evaluation", "input")):
    os.makedirs(os.path.join(directory, "Evaluation", "input"))

# Input files:

agwn_file_name = "wn-data-grc.tab"
ss_file_name = "CORE_SS.matrix_w" + str(window) + "_t" + str(freq_threshold) + ".ppmi.svd_300.dm"
ssrows_file_name = "CORE_SS.matrix_w" + str(window) + "_t" + str(freq_threshold) + ".ppmi.svd_300.rows"
neighbours_file_name = "NEIGHBOURS.matrix_w" + str(window) + "_t" + str(freq_threshold) + ".rows.CORE_SS.matrix_w" \
                       + str(window) + "_t" + str(freq_threshold) + ".ppmi.svd_300.cos"

# Output files:

file_out_agwn_cooccurrence_name = "AGWN_co-occurrences.csv"  # matrix with 0s and 1s depending on whether two lemmas
# co-occur in the same AGWN synset
# file_out_agwn_distances_name = "AGWN_distances.txt"  # matrix with cosine distances between pairs of AGWN lemmas
file_out_dissect_5neighbour_name = "semantic-space_w" + str(window) + "_t" + str(
    freq_threshold) + "5neighbours_lemmas.txt"
file_out_dissect_5neighbour_distances_name = "semantic-space_w" + str(window) + "_t" + str(
    freq_threshold) + "5neighbours_distances.txt"
file_out_shared_lemmas_name = "AGWN-semantic-space_w" + str(window) + "_t" + str(freq_threshold) + "_shared-lemmas.txt"
file_out_agwn_vocabulary_name = "AGWN_lemmas.txt"
# file_out_dissect_distances_name = "semantic_space_w" + str(window) + "_t" + str(freq_threshold) + "distances.csv"
hist_file_name = "cos-distances-5neighbours_semantic-space_w" + str(window) + "_t" + str(freq_threshold) + "hist.png"
summary_stats_dissect_5neighbours_file_name = "summary_statistics_distance_semantic-space_w" + str(window) + "_t" + \
                                              str(freq_threshold) + "_5neighbours.txt"
summary_dissect_agwn_distances_file_name = "summary_comparison_distances_AGWN_semantic-space_w" + str(window) + "_t" + \
                                           str(freq_threshold) + "_5neighbours.txt"
summary_dissect_agwn_overlap_file_name = "summary_overlap_AGWN_semantic-space_w" + str(window) + "_t" + \
                                           str(freq_threshold) + "_5neighbours.txt"

if istest == "yes":
    file_out_agwn_cooccurrence_name = file_out_agwn_cooccurrence_name.replace(".csv", "_test.csv")
    # file_out_agwn_distances_name = file_out_agwn_distances_name.replace(".txt", "_test.txt")
    file_out_dissect_5neighbour_name = file_out_dissect_5neighbour_name.replace(".txt", "_test.txt")
    file_out_dissect_5neighbour_distances_name = file_out_dissect_5neighbour_distances_name.replace(".txt", "_test.txt")
    file_out_shared_lemmas_name = file_out_shared_lemmas_name.replace(".txt", "_test.txt")
    file_out_agwn_vocabulary_name = file_out_agwn_vocabulary_name.replace(".txt", "_test.txt")
    # file_out_dissect_distances_name = file_out_dissect_distances_name.replace(".csv", "_test.csv")
    hist_file_name = hist_file_name.replace(".png", "_test.png")
    summary_stats_dissect_5neighbours_file_name = summary_stats_dissect_5neighbours_file_name.replace(".txt",
                                                                                                      "_test.txt")
    summary_dissect_agwn_distances_file_name = summary_dissect_agwn_distances_file_name.replace(".txt", "_test.txt")
    summary_dissect_agwn_overlap_file_name = summary_dissect_agwn_overlap_file_name.replace(".txt", "_test.txt")

# Initialize objects:

agwn_vocabulary = list()  # list of all lemmas in AG WordNet
agwn_lemma2id = dict()  # maps an AGWN lemma to its id
synsets = dict()  # maps a synset ID to the list of its contents
synsetid2def = dict()  # maps a synset ID to its English definition
agwn_cooccurrence = defaultdict(
    dict)  # multidimensional dictionary: maps each pair of lemmas to 1 if they co-occur in the same WN synset, and 0 otherwise
agwn_coordinates = dict()  # maps an AGWN lemma id to the list of its 0/1 coordinates in the AGWN space
# agwn_distances = list()  # indexes an AGWN lemma id to the array of cosine distances with other AGWN lemmas
agwn_dissect_5neighbours = list()  # list of lemmas shared between AGWN and DISSECT space with 5 neighbours
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
dissect_distances_agwn_dissect_5neighbour = list()  # list of distance values from dissect_lemma2cosinedistance only
# considering the lemmas shared between top 5 neighbours in DISSECT and AGWN lemmas

# ----------------------------------------
# Index lemmas in DISSECT semantic space:
# ----------------------------------------

print("Reading DISSECT lemmas...")

ssrows_file = open(os.path.join(dir_ss_rows, ssrows_file_name), 'r', encoding="UTF-8")
row_count_rows = sum(1 for line in ssrows_file)
ssrows_file.close()

ssrows_file = open(os.path.join(dir_ss_rows, ssrows_file_name), 'r', encoding="UTF-8")
count_n = -1

lemma = ""
for line in ssrows_file:
    count_n += 1

    if (istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_rows):
        if count_n % 500 == 0:
            print("Reading rows in DISSECT, line", str(count_n))
        line = line.rstrip('\n')
        dissect_id2lemma[count_n] = line
        if (line == "κομιδή") or (line == "ἐπιμέλεια"):
            print("\tTest!")
            print("\tκομιδή or ἐπιμέλεια:", str(count_n), dissect_id2lemma[count_n])

ssrows_file.close()

print("Examples...")
print("0:", dissect_id2lemma[0])
print("29:", dissect_id2lemma[29])
print("188:", dissect_id2lemma[188])

# --------------------------------------
# Read WordNet file to collect synsets:
# --------------------------------------

print("-----------------\nReading AGWN...\n-----------------")

agwn_file = open(os.path.join(dir_wn, agwn_file_name), 'r', encoding="UTF-8")
row_count_agwn = sum(1 for line in agwn_file)
agwn_file.close()

# Read synsets from WordNet file:

count_n = 1  # counts lines read
agwn_file = open(os.path.join(dir_wn, agwn_file_name), 'r', encoding="UTF-8")
agwn_reader = csv.reader(agwn_file, delimiter="\t")
next(agwn_reader)  # This skips the first row of the CSV file

synset_def = ""
synset_lemma = ""

for row in agwn_reader:
    count_n += 1
    if ((istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_agwn)) and (
                count_n > 1):

        if count_n % 500 == 0:
            print("WordNet: line", str(count_n), " out of ", str(row_count_agwn))

        row[0] = row[0].replace("#", "")
        synset_id = row[0]
        if row[1] == "eng:def":
            synset_def = row[3]
            # print("Def:", synset_def)
            synsetid2def[synset_id] = synset_def
        else:
            synset_lemma = row[2]

            if synset_id in synsets:
                synsets_this_id = synsets[synset_id]
                synsets_this_id.append(synset_lemma)
                synsets[synset_id] = synsets_this_id
            else:
                synsets[synset_id] = [synset_lemma]

                # print("\tSynset ID:", str(synset_id), "; lemma:", synset_lemma)

agwn_file.close()

try:
    print("Examples:")
    print("synset_id: 00267522-n", str(synsets['00267522-n']))
except:
    KeyError

if skip_read_files == "no":

    # ---------------------------------------------
    # read data from DISSECT semantic space
    # ---------------------------------------------

    # neighbours in semantic space:

    print(
        "-----------------------------------------\nReading neighbours in DISSECT space....\n----------------------------------------")

    neighbours_file = open(os.path.join(dir_ss, neighbours_file_name), 'r', encoding="UTF-8")
    row_count_neighbours = sum(1 for line in neighbours_file)
    neighbours_file.close()

    neighbours_file = open(os.path.join(dir_ss, neighbours_file_name), 'r', encoding="UTF-8")
    count_n = 0

    lemma = ""
    for line in neighbours_file:
        count_n += 1

        if (istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_neighbours):
            if count_n % 1000 == 0:
                print("Reading neighbours in DISSECT, line", str(count_n))
            line = line.rstrip('\n')
            if not line.startswith("\t"):
                # print("lemma!", line)
                lemma = line
                dissect_lemmas_5neighbours.append(lemma)
            else:
                # print("neighbour!", line)
                fields = line.split("\t")
                # print("fields:", fields)
                neighbour_distance = fields[1]
                # print("neighbour_distance:", neighbour_distance)
                neighbour_distance_fields = neighbour_distance.split(" ")
                # print("neighbour_distance_fields:", neighbour_distance_fields)
                neighbour = neighbour_distance_fields[0]
                # print("neighbour:", neighbour)
                neighbour_distance = neighbour_distance_fields[1]
                # print("distance:", neighbour_distance)
                if neighbour != lemma:
                    lemma_neighbour2distance[lemma, neighbour] = neighbour_distance
                    dissect_distances_5neighbour.append(neighbour_distance)
                    if lemma == "κομιδή" or lemma == "ἐπιμέλεια":
                        print("\tTest!")
                        print("\tlemma:", lemma, "neighbour:", neighbour, "count_n:", str(count_n),
                              "lemma_neighbour2distance[lemma,neighbour]:",
                              str(lemma_neighbour2distance[lemma, neighbour]))

    neighbours_file.close()

    print("Examples:")
    print("Lemma: ἐπιμέλεια/κομιδή, its neighbours and distances:")
    for lemma, neighbour in lemma_neighbour2distance:
        if lemma == "ἐπιμέλεια" or lemma == "κομιδή":
            print(lemma, neighbour, str(lemma_neighbour2distance[lemma, neighbour]))

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
            # print("lemma:", lemma, "neighbour:", neighbour, "distance:", lemma_neighbour2distance[lemma, neighbour])
            file_out_dissect_5neighbour_distances_writer.writerow(
                [lemma, neighbour, str(lemma_neighbour2distance[lemma, neighbour])])

    # ---------------------------------------------------------------------------------------
    # Print co-occurrence matrix from WordNet synsets for lemmas shared with DISSECT lemmas
    # ---------------------------------------------------------------------------------------

    # create vocabulary list and co-occurrence pairs:

    print(
        "------------------------------------------\nDefining AGWN co-occurrence pairs...\n------------------------------------------")

    count_s = 0
    for synset_id in synsets:

        count_s += 1
        if count_s % 100 == 0:
            print(str(count_s), ":", "synset ID:", synset_id)
            # print("Definition:", synsetid2def[synset_id])
            print("Lemmas:", str(synsets[synset_id]))

        synsets_this_lemma = synsets[synset_id]

        for lemma1 in synsets_this_lemma:

            # print("Defining co-occurrences of lemma1:", lemma1)
            if lemma1 not in agwn_vocabulary:
                agwn_vocabulary.append(lemma1)
                # print("Vocabulary so far:", str(agwn_vocabulary))

            for lemma2 in synsets_this_lemma:
                if (lemma1, lemma2) in agwn_cooccurrence:
                    agwn_cooccurrence[lemma1, lemma2] += 1
                else:
                    agwn_cooccurrence[lemma1, lemma2] = 1

                if count_s % 1000 == 0:
                    print(str(count_s), ":", "lemma1:", lemma1, "lemma2:", lemma2, "co-occurrence:",
                          str(agwn_cooccurrence[lemma1, lemma2]))
                # print("Lemma1:", lemma1, "Lemma2:", lemma2, "Co-occurrence:", str(agwn_cooccurrence[lemma1][lemma2]))

                if (lemma1 == "κομιδή") and (lemma2 == "ἐπιμέλεια"):
                    print("\tTest!")
                    print("\tLemma1:", lemma1, "lemma2:", lemma2, "co-occurrence:",
                          str(agwn_cooccurrence[lemma1, lemma2]))

                    # if (synset_id in ["00267522-n", "00654885-n", "00829378-n", "05615869-n", "05650579-n", "05702275-n", "05853636-n", "07524529-n", "01824736-v", "01824736-v"]):
                    # if ((synset_id == "00267522-n" or synset_id == "00654885-n")) and (lemma1 == "κομιδή") and (lemma2 == "ἐπιμέλεια"):
                    #    print("Test!")
                    #    print("Synset_id:", synset_id)
                    #    print("Lemma1:", lemma1, "lemma2:", lemma2, "co-occurrence:",
                    #          str(agwn_cooccurrence[(lemma1, lemma2)]))

    # agwn_vocabulary = sorted(agwn_vocabulary)

    with open(os.path.join(dir_out, file_out_agwn_vocabulary_name), 'w', encoding="UTF-8") as agwn_vocabulary_file:
        for lemma in agwn_vocabulary:
            agwn_vocabulary_file.write("%s\n" % lemma)

    # -------------------------------------------------------------------------------
    # find lemmas present in both AGWN and DISSECT space (called “shared lemmas”)
    # -------------------------------------------------------------------------------

    print(
                "---------------------------------------------\nFind shared lemmas between AGWN and DISSECT space....\n---------------------------------------")

    agwn_dissect_5neighbours = list((set(dissect_lemmas_5neighbours).intersection(set(agwn_vocabulary))))

    with open(os.path.join(dir_out, file_out_shared_lemmas_name), 'w',
                      encoding="UTF-8") as out_shared_lemmas_file:
        for lemma in agwn_dissect_5neighbours:
                    out_shared_lemmas_file.write("%s\n" % lemma)

    # ------------------------------------------
    # finish defining AGWN co-occurrence pairs:
    # ------------------------------------------

    print("----------------------------------------\nFinish defining AGWN co-occurrence pairs...\n-------------------------------------------------")

    count_n = -1
    with open(os.path.join(dir_out, file_out_agwn_cooccurrence_name), 'w',
              encoding="UTF-8") as file_out_agwn_cooccurrence:

        file_out_agwn_cooccurrence_writer = csv.writer(file_out_agwn_cooccurrence, delimiter="\t")

        #for lemma1 in agwn_vocabulary:
        for lemma1 in agwn_dissect_5neighbours:
            count_n += 1
            coordinates_lemmaid1 = list()
            # print("Printing Co-occurrences of lemma1 (", str(count_n), "out of", str(len(agwn_vocabulary)), "):",
            #      lemma1)
            count_n2 = 0

            #for lemma2 in agwn_vocabulary:
            for lemma2 in agwn_dissect_5neighbours:
                count_n2 += 1
                # print("Co-occurrence of ", lemma1, "and", lemma2, ":")
                # try:
                #    print("Co-occurrence of", lemma1, "and", lemma2, ":", str(agwn_cooccurrence[(lemma1,lemma2)]))
                #    # file_out_agwn_cooccurrence.write("\t" + str(agwn_cooccurrence[lemma1][lemma2]))
                #    # if lemma1 == "κτίσμα":
                #    #    print("Co-occurrence of ", lemma1, "and", lemma2, ":", str(wn_cooccurrence[lemma1][lemma2]))
                # except KeyError:
                #    agwn_cooccurrence[(lemma1, lemma2)] = 0
                if (lemma1, lemma2) not in agwn_cooccurrence:
                    agwn_cooccurrence[lemma1, lemma2] = 0
                    # if lemma1 == "κτίσμα":
                    #    print("Co-occurrence of ", lemma1, "and", lemma2, ":", str(agwn_cooccurrence[lemma1][lemma2]))
                    # file_out_agwn_cooccurrence.write("\t" + str(agwn_cooccurrence[lemma1][lemma2]))
                if count_n % 5000 == 0 and count_n2 % 500 == 0:
                    print(str(count_n), "out of", len(agwn_dissect_5neighbours), "and", str(count_n2), ":", "co-occurrence of", lemma1, "and", lemma2, ":",
                          str(agwn_cooccurrence[lemma1, lemma2]))

                coordinates_lemmaid1.append(agwn_cooccurrence[lemma1, lemma2])

            file_out_agwn_cooccurrence_writer.writerow(coordinates_lemmaid1)

            agwn_coordinates[lemma1] = coordinates_lemmaid1

            if (lemma1 == "κομιδή") or (lemma1 == "ἐπιμέλεια"):
                print("\tTest!")
                print("\tCount:", str(count_n), "Lemma1:", lemma1)
                print("\tNon-zero AGWN coordinates at positions/lemmas:")
                l1_c = agwn_coordinates[lemma1]
                print(str([(i, agwn_vocabulary[i]) for i, e in enumerate(l1_c) if e != 0]))

    print("AGWN co-occurrence:")
    # print(str(agwn_cooccurrence))
    print("Examples:")
    print("κομιδή", "ἐπιμέλεια", str(agwn_cooccurrence["κομιδή", "ἐπιμέλεια"]))
    print("ἐπιμέλεια", "κομιδή", str(agwn_cooccurrence["ἐπιμέλεια", "κομιδή"]))

    print("AGWN coordinates:")
    # print(str(agwn_coordinates))
    print("Examples:")
    # print("0", str(agwn_coordinates[0]))

    try:
        print("κομιδή (" + agwn_vocabulary["κομιδή"] + "):")
        print("Non-zero AGWN coordinates at positions/lemmas:")
        l1_c = agwn_coordinates["κομιδή"]
        print(str([(i, agwn_vocabulary[i]) for i, e in enumerate(l1_c) if e != 0]))
    except:
        pass

    try:
        print("ἐπιμέλεια (" + agwn_vocabulary["ἐπιμέλεια"] + "):")
        print("Non-zero AGWN coordinates at positions/lemmas:")
        l1_c = agwn_coordinates["ἐπιμέλεια"]
        print(str([(i, agwn_vocabulary[i]) for i, e in enumerate(l1_c) if e != 0]))
    except:
        pass

    # print("ἐπιμέλεια", "κομιδή", str(agwn_cooccurrence[("ἐπιμέλεια", "κομιδή")]))

else:

    # read list of AGWN lemmas:

    agwn_vocabulary = [line.rstrip('\n') for line in
                       open(os.path.join(dir_out, file_out_agwn_vocabulary_name), 'r', encoding="UTF-8")]

    # read list of DISSECT lemmas with top 5 neighbours:

    dissect_lemmas_5neighbours = [line.rstrip('\n') for line in
                                  open(os.path.join(dir_out, file_out_dissect_5neighbour_name), 'r', encoding="UTF-8")]

    # read list of shared lemmas between AGWN and DISSECT lemmas with top 5 neighbours:

    agwn_dissect_5neighbours = [line.rstrip('\n') for line in
                                open(os.path.join(dir_out, file_out_shared_lemmas_name), 'r', encoding="UTF-8")]

    # read mapping between pairs of AGWN lemmas and their co-occurrence value in AGWN:

    file_agwn_cooccurrence = open(os.path.join(dir_out, file_out_agwn_cooccurrence_name), 'r', encoding="UTF-8")
    row_count_agwn_cooccurrence = sum(1 for line in file_agwn_cooccurrence)
    file_agwn_cooccurrence.close()

    file_agwn_cooccurrence = open(os.path.join(dir_out, file_out_agwn_cooccurrence_name), 'r', encoding="UTF-8")
    agwn_cooccurrence_reader = csv.reader(file_agwn_cooccurrence, delimiter="\t")

    count_n = -1

    for row in agwn_cooccurrence_reader:
        count_n += 1
        if ((istest == "yes" and count_n < lines_read_testing) or (
                        istest == "no" and count_n <= row_count_agwn_cooccurrence)):

            if count_n % 1000 == 0:
                print("AGWN Co-occurrence: line", str(count_n), "out of", str(row_count_agwn_cooccurrence))

            lemma1 = agwn_vocabulary[count_n]
            if lemma1 == "κομιδή" or lemma1 == "ἐπιμέλεια":
                print("\tTest!")
                print("\tκομιδή or ἐπιμέλεια:", lemma1, str(count_n), str(agwn_vocabulary[count_n]))

            # print(str(row))
            coocc = [float(i) for i in row]
            agwn_coordinates[lemma1] = coocc
            for ic in range(0, len(coocc)):
                if coocc[ic] == 1:
                    agwn_cooccurrence[lemma1, agwn_vocabulary[ic]] = 1
                else:
                    agwn_cooccurrence[lemma1, agwn_vocabulary[ic]] = 0

                if lemma1 == "κομιδή" or lemma1 == "ἐπιμέλεια":
                    print("\tTest!")
                    print("\tκομιδή or ἐπιμέλεια:", lemma1, str(ic), str(agwn_vocabulary[ic]),
                          str(agwn_cooccurrence[lemma1, agwn_vocabulary[ic]]))

    file_agwn_cooccurrence.close()

    print("AGWN co-occurrence:")
    # print(str(agwn_cooccurrence))
    print("Examples:")
    print("κομιδή", "ἐπιμέλεια", str(agwn_cooccurrence["κομιδή", "ἐπιμέλεια"]))
    print("ἐπιμέλεια", "κομιδή", str(agwn_cooccurrence["ἐπιμέλεια", "κομιδή"]))

    print("AGWN coordinates:")
    # print(str(agwn_coordinates))
    print("Examples:")

    print("ἐπιμέλεια (" + agwn_vocabulary["ἐπιμέλεια"] + "):")
    print("Non-zero AGWN coordinates at positions/lemmas:")
    l1_c = agwn_coordinates["ἐπιμέλεια"]
    print(str([(i, agwn_vocabulary[i]) for i, e in enumerate(l1_c) if e != 0]))

    print("κομιδή (" + agwn_vocabulary["κομιδή"] + "):")
    print("Non-zero AGWN coordinates at positions/lemmas:")
    l1_c = agwn_coordinates["κομιδή"]
    print(str([(i, agwn_vocabulary[i]) for i, e in enumerate(l1_c) if e != 0]))

    # read mapping between pairs of lemmas and their cosine distance in the DISSECT semantic space:

    # file_dissect_distances = open(os.path.join(dir_out, file_out_dissect_distances_name), 'r', encoding="UTF-8")
    # row_count_dissect_distances = sum(1 for line in file_dissect_distances)
    # file_dissect_distances.close()

    # file_dissect_distances = open(os.path.join(dir_out, file_out_dissect_distances_name), 'r', encoding="UTF-8")
    # dissect_distances_reader = csv.reader(file_dissect_distances, delimiter="\t")

    # next(dissect_distances_reader)

    # count_n = 0

    # for row in dissect_distances_reader:
    #    count_n += 1
    #    if ((istest == "yes" and count_n < lines_read_testing) or (
    #            istest == "no" and count_n <= row_count_dissect_distances)):
    #        print("DISSECT distance: line", str(count_n), "out of", str(row_count_dissect_distances))
    #        cos_distances = float(row)
    #        dissect_lemmaid2cosinedistance.append(cos_distances)

    # file_dissect_distances.close()

    # read mapping between pairs of lemmas and their top 5 neighbours, and their cosine distance in the DISSECT semantic space:

    file_dissect_5neighbour_distances = open(os.path.join(dir_out, file_out_dissect_5neighbour_distances_name), 'r',
                                             encoding="UTF-8")
    row_count_dissect_5neighbour_distances = sum(1 for line in file_dissect_5neighbour_distances)
    file_dissect_5neighbour_distances.close()

    file_dissect_5neighbour_distances = open(os.path.join(dir_out, file_out_dissect_5neighbour_distances_name), 'r',
                                             encoding="UTF-8")
    dissect_5neighbour_distances_reader = csv.reader(file_dissect_5neighbour_distances, delimiter="\t")

    next(dissect_5neighbour_distances_reader)

    lemma = ""
    neighbour = ""
    cos_distance = 0

    count_n = 0

    for row in dissect_5neighbour_distances_reader:
        count_n += 1
        if ((istest == "yes" and count_n < lines_read_testing) or (
                        istest == "no" and count_n <= row_count_dissect_5neighbour_distances)):

            if count_n % 1000 == 0:
                print("DISSECT distance for neighbours: line", str(count_n), "out of",
                      str(row_count_dissect_5neighbour_distances))
            lemma = row[0]
            neighbour = row[1]
            cos_distance = row[2]
            lemma_neighbour2distance[lemma, neighbour] = float(cos_distance)
            dissect_distances_5neighbour.append(cos_distance)

            if lemma == "κομιδή" or lemma == "ἐπιμέλεια":
                print("\tTest!")
                print("\tlemma:", lemma, "neighbour:", neighbour, "count_n:", str(count_n),
                      "lemma_neighbour2distance[lemma,neighbour]:", str(lemma_neighbour2distance[lemma, neighbour]))

    print("Examples:")
    print("Lemma: ἐπιμέλεια, its neighbours and distances:")
    for lemma, neighbour in lemma_neighbour2distance:
        if lemma == "ἐπιμέλεια":
            print(neighbour, str(lemma_neighbour2distance[lemma, neighbour]))

# read semantic space coordinates:

print("--------------------------------\nReading semantic space coordinates\n---------------------------------------")

ss_file = open(os.path.join(dir_ss_rows, ss_file_name), 'r', encoding="UTF-8")
row_count_ss = sum(1 for line in ss_file)
ss_file.close()

ss_file = open(os.path.join(dir_ss_rows, ss_file_name), 'r', encoding="UTF-8")
count_n = 0
for line in ss_file:
    count_n += 1
    if (istest == "yes" and count_n < lines_read_testing) or (istest == "no" and count_n <= row_count_ss):

        if count_n % 1000 == 0:
            print("Reading DISSECT coordinates, line", str(count_n))
        line = line.rstrip('\n')
        coordinates = line.split('\t')
        lemma = coordinates[0]
        coordinates.pop(0)
        dissect_lemma2coordinates[lemma] = np.asarray(list(np.float_(coordinates)))

        if lemma == "κομιδή" or lemma == "ἐπιμέλεια":
            print("\tTest!")
            print("\tlemma:", lemma)
            print("\tNon-zero DISSECT coordinates at positions/lemmas:")
            l1_c = dissect_lemma2coordinates[lemma]
            print(str([(i, dissect_id2lemma[i]) for i, e in enumerate(l1_c) if e != 0]))

ss_file.close()

print("Examples:")
print("lemma: ἐπιμέλεια")
print("Non-zero DISSECT coordinates at positions/lemmas:")
l1_c = dissect_lemma2coordinates["ἐπιμέλεια"]
print(str([(i, dissect_id2lemma[i]) for i, e in enumerate(l1_c) if e != 0]))
print("lemma: κομιδή")
print("Non-zero DISSECT coordinates at positions/lemmas:")
l1_c = dissect_lemma2coordinates["κομιδή"]
print(str([(i, dissect_id2lemma[i]) for i, e in enumerate(l1_c) if e != 0]))

# ---------------------------------------------
# Evaluation approaches:
# ---------------------------------------------


print("There are", str(len(agwn_vocabulary)), "lemmas in AGWN, ", str(len(dissect_lemmas_5neighbours)),
      "lemmas in DISSECT space with 5 top neighbours", "and", str(len(agwn_dissect_5neighbours)),
      "in the intersection.")
# There are 22828 lemmas in AGWN,  44740 lemmas in DISSECT space with 5 top neighbours and 12899 in the intersection.

print(
    "---------------------------------------------------------\nFirst evaluation approach: distances in DISSECT space\n---------------------------------------------------------")
# 1.	Distances in DISSECT space
# a.	Calculate distance between pairs of shared lemmas that are among top 5 neighbours in DISSECT space
# b.	Compare this distance with average distance between pairs of lemmas and their neighbours in DISSECT space

summary_stats_dissect_5neighbours_file = open(os.path.join(dir_out, summary_stats_dissect_5neighbours_file_name), 'w')
# calculate summary statistics of distance between pairs of lemmas and their top 5 neighbours in DISSECT space:

# print(str(type(dissect_distances_5neighbour)))
# dissect_distances_5neighbour = np.asarray(dissect_distances_5neighbour)
dissect_distances_5neighbour = np.array(dissect_distances_5neighbour, dtype=np.float32)
# print(str(type(dissect_distances_5neighbour)))
# print(str(dissect_distances_5neighbour))
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

# Define list of DISSECT distances between pairs of shared lemmas that are among top 5 neighbours in DISSECT space:

count_n = 0
for lemma, neighbour in lemma_neighbour2distance:
    if lemma in agwn_dissect_5neighbours:
        count_n += 1
        if count_n % 100 == 0:
            print("lemma:", lemma, "neighbour:", neighbour, "distance:",
                  str(lemma_neighbour2distance[lemma, neighbour]))
        if lemma == "κομιδή" or lemma == "ἐπιμέλεια":
            print("\tTest!")
            print("\tlemma:", lemma, "neighbour:", neighbour, "distance:",
                  str(lemma_neighbour2distance[lemma, neighbour]))

        dissect_distances_agwn_dissect_5neighbour.append(lemma_neighbour2distance[lemma, neighbour])

# Calculate summary statistics of distance between pairs of shared lemmas that are among top 5 neighbours in DISSECT space:

if len(agwn_dissect_5neighbours) > 0:
    dissect_distances_agwn_dissect_5neighbour = np.array(dissect_distances_agwn_dissect_5neighbour, dtype=np.float32)

    mean_dissect_distances_agwn_dissect_5neighbour = np.mean(dissect_distances_agwn_dissect_5neighbour)
    std_dissect_distances_agwn_dissect_5neighbour = dissect_distances_agwn_dissect_5neighbour.std()
    min_dissect_distances_agwn_dissect_5neighbour = dissect_distances_agwn_dissect_5neighbour.min()
    max_dissect_distances_agwn_dissect_5neighbour = dissect_distances_agwn_dissect_5neighbour.max()
    median_dissect_distances_agwn_dissect_5neighbour = np.median(dissect_distances_agwn_dissect_5neighbour)
    perc25_dissect_distances_agwn_dissect_5neighbour = np.percentile(dissect_distances_agwn_dissect_5neighbour, 25)
    perc75_dissect_distances_agwn_dissect_5neighbour = np.percentile(dissect_distances_agwn_dissect_5neighbour, 75)

    summary_stats_dissect_5neighbours_file.write(
        "Mean of DISSECT distances 5 neighbours:" + str(mean_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "Mean of DISSECT distances of shared lemmas between DISSECT top 5 neighbours "
        "and AGWN lemmas:" + str(
            mean_dissect_distances_agwn_dissect_5neighbour) + "\n")
    print("Mean of DISSECT distances 5 neighbours:" + str(mean_dissect_distances_5neighbour))
    print("Mean of DISSECT distances of shared lemmas between DISSECT top 5 neighbours "
          "and AGWN lemmas:" + str(mean_dissect_distances_agwn_dissect_5neighbour))

    summary_stats_dissect_5neighbours_file.write(
        "STD of DISSECT distances 5 neighbours:" + str(std_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write("STD of distances of shared lemmas between DISSECT top 5 neighbours "
                                                 "and AGWN lemmas:" + str(
        std_dissect_distances_agwn_dissect_5neighbour) + "\n")

    summary_stats_dissect_5neighbours_file.write(
        "Min of DISSECT distances 5 neighbours:" + str(min_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "Min of DISSECT distances of shared lemmas between DISSECT top 5 neighbours and AGWN lemmas:" + str(
            min_dissect_distances_agwn_dissect_5neighbour) + "\n")

    summary_stats_dissect_5neighbours_file.write(
        "Max of DISSECT distances 5 neighbours:" + str(max_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "Max of DISSECT distances of shared lemmas between DISSECT top 5 neighbours and AGWN lemmas:" + str(
            max_dissect_distances_agwn_dissect_5neighbour) + "\n")

    summary_stats_dissect_5neighbours_file.write(
        "Median of DISSECT distances 5 neighbours:" + str(median_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "Median of DISSECT distances of shared lemmas between DISSECT top 5 neighbours and AGWN lemmas:" + str(
            median_dissect_distances_agwn_dissect_5neighbour) + "\n")

    summary_stats_dissect_5neighbours_file.write(
        "25th percentile of DISSECT distances 5 neighbours:" + str(perc25_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "25th percentile of DISSECT distances of shared lemmas between DISSECT top 5 neighbours and AGWN lemmas:" + str(
            perc25_dissect_distances_agwn_dissect_5neighbour) + "\n")

    summary_stats_dissect_5neighbours_file.write(
        "75th percentile of DISSECT distances 5 neighbours:" + str(perc75_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "75th percentile of DISSECT distances of shared lemmas between DISSECT top 5 "
        "neighbours and AGWN lemmas:" + str(
            perc75_dissect_distances_agwn_dissect_5neighbour) + "\n")

else:
    summary_stats_dissect_5neighbours_file.write(
        "Mean of DISSECT distances 5 neighbours:" + str(mean_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "STD of DISSECT distances 5 neighbours:" + str(std_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "Min of DISSECT distances 5 neighbours:" + str(min_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "Max of DISSECT distances 5 neighbours:" + str(max_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "Median of DISSECT distances 5 neighbours:" + str(median_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "25th percentile of DISSECT distances 5 neighbours:" + str(perc25_dissect_distances_5neighbour) + "\n")
    summary_stats_dissect_5neighbours_file.write(
        "75th percentile of DISSECT distances 5 neighbours:" + str(perc75_dissect_distances_5neighbour) + "\n")

    summary_stats_dissect_5neighbours_file.write("No intersection.")
    print("No intersection.")

summary_stats_dissect_5neighbours_file.close()

# --------------------------------------------------------------------------------------------
# 2.	Distances in AGWN vs DISSECT space
# a.	Calculate distance between pairs of shared lemmas that are synonyms in AGWN space
# b.	Calculate distance between pairs of shared lemmas that are neighbours in DISSECT space
# c.	For each pair, compare distance in AGWN space with distance in DISSECT space
# --------------------------------------------------------------------------------------------

print(
    "---------------------------------------------------------\nSecond evaluation approach: distances in AGWN/DISSECT spaces between synonyms/neighbours\n-------------------------------------------------")

# calculate cosine distances in AGWN space between pairs of shared lemmas that are synonyms in AGWN space:

agwn_distances_shared_agwn_synonyms = list()  # cosine distances in the AGWN space between pairs of AGWN synonyms

# calculate cosine distances in DISSECT space between pairs of shared lemmas that are synonyms in AGWN space:

dissect_distances_shared_agwn_synonyms = list()

# define lemma2id mapping for AGWN lemmas:

print("Defining lemma2id mapping for AGWN lemma")

for id in range(0, len(agwn_vocabulary)):

    agwn_lemma2id[agwn_vocabulary[id]] = id
    if id % 1000 == 0:
        print("Id:", str(id), "lemma:", agwn_vocabulary[id], "lemma2id:", str(agwn_lemma2id[agwn_vocabulary[id]]))

    if agwn_vocabulary[id] == "κομιδή" or agwn_vocabulary[id] == "ἐπιμέλεια":
        print("\tTest!")
        print("\tId:", str(id), "lemma:", agwn_vocabulary[id], "lemma2id:", str(agwn_lemma2id[agwn_vocabulary[id]]))

print("Examples:")
print("Id=1092, lemma:", agwn_vocabulary[1092], "id:", str(agwn_lemma2id[agwn_vocabulary[1092]]))
print("Id=1109, lemma:", agwn_vocabulary[1109], "id:", str(agwn_lemma2id[agwn_vocabulary[1109]]))

# loop over all pairs of AGWN synonyms and collect AGWN/DISSECT distances between them:

print("Calculating AGWN and DISSECT distances between pairs of AGWN synonyms...")

#print("agwn_dissect_5neighbours:", str(agwn_dissect_5neighbours))

count_n = 0
for [lemma1, lemma2] in agwn_cooccurrence:

    count_n += 1

    id1 = agwn_lemma2id[lemma1]
    id2 = agwn_lemma2id[lemma2]

    #if count_n % 500000 == 0:
    #    print("Consider synonyms", lemma1, "and", lemma2, "AGWN IDs are", str(id1), "and", str(id2))

    if lemma1 == "ἐπιμέλεια" and lemma2 == "κομιδή":
        print("\tTest!")
        print("\tConsider synonyms", lemma1, "and", lemma2, "AGWN IDs are", str(id1), "and", str(id2))
        if lemma1 in agwn_dissect_5neighbours and lemma2 in agwn_dissect_5neighbours:
            print("\tThey are shared!")
            if id1 < id2:
                print("\t and id1 < id2!")

    if (lemma1 in agwn_dissect_5neighbours) and (lemma1 is not lemma2) and (lemma2 in agwn_dissect_5neighbours) and (id1 < id2):

        agwn_cos_distance_lemma1_lemma2 = distance.cosine(agwn_coordinates[lemma1], agwn_coordinates[lemma2])
        # print("AGWN coordinates for lemma1:", str(agwn_coordinates[id1]))
        # print("AGWN coordinates for lemma2:", str(agwn_coordinates[id2]))

        agwn_distances_shared_agwn_synonyms.append(agwn_cos_distance_lemma1_lemma2)

        # print("DISSECT coordinates for lemma1:", str(dissect_lemma2coordinates[lemma1][0:4]))
        # print("DISSECT coordinates for lemma2:", str(dissect_lemma2coordinates[lemma2][0:4]))
        dissect_cos_distance_lemma1_lemma2 = distance.cosine(dissect_lemma2coordinates[lemma1],
                                                             dissect_lemma2coordinates[lemma2])

        if count_n % 1000 == 0:
            print("Consider synonyms", lemma1, "and", lemma2, "AGWN IDs are", str(id1), "and", str(id2), "AGWN cosine distance:", str(agwn_cos_distance_lemma1_lemma2), "DISSECT cosine distance:", str(dissect_cos_distance_lemma1_lemma2))

        dissect_distances_shared_agwn_synonyms.append(dissect_cos_distance_lemma1_lemma2)

        if lemma1 == "ἐπιμέλεια" and lemma2 == "κομιδή":
            print("\tTest!")
            print("\tConsider synonyms", lemma1, "and", lemma2, ", AGWN IDs are", str(id1), "and", str(id2),
              "AGWN cosine distance:", str(agwn_cos_distance_lemma1_lemma2), "DISSECT cosine distance:", str(dissect_cos_distance_lemma1_lemma2))

#print("agwn_distances_shared_agwn_synonyms:", str(agwn_distances_shared_agwn_synonyms))
#print("dissect_distances_shared_agwn_synonyms:", str(dissect_distances_shared_agwn_synonyms))

# calculate cosine distances in AGWN space between pairs of shared lemmas that are neighbours in DISSECT space:
# calculate cosine distances in DISSECT space between pairs of shared lemmas that are neighbours in DISSECT space:

print("Calculating AGWN and DISSECT distances between pairs of DISSECT neighbours...")

agwn_distances_shared_dissect_neighbours = list()  # cosine distances in the AGWN space between pairs of DISSECT neighbours
dissect_distances_shared_dissect_neighbours = list()  # cosine distances in the AGWN space between pairs of DISSECT neighbours

# loop over all pairs of lemma and its neighbours:


count_n = 0
for [lemma, neighbour] in lemma_neighbour2distance:

    count_n += 1

    if count_n % 100 == 0:
        print("Consider lemma+neighbour", lemma, "and", neighbour)

    if (lemma in agwn_dissect_5neighbours) and (neighbour in agwn_dissect_5neighbours):

        id1 = agwn_lemma2id[lemma]
        id2 = agwn_lemma2id[neighbour]

        #if count_n % 100 == 0:
        print("Consider shared lemma+neighbour", lemma, "and", neighbour, "AGWN IDs are:", str(id1), str(id2))

        if (lemma is not neighbour) and (id1 < id2):

            agwn_cos_distance_lemma_neighbour = distance.cosine(agwn_coordinates[lemma], agwn_coordinates[neighbour])
            agwn_distances_shared_dissect_neighbours.append(agwn_cos_distance_lemma_neighbour)

            dissect_cos_distance_lemma_neighbour = lemma2neighbour[agwn_vocabulary[id1]][agwn_vocabulary[id2]]
            if count_n % 100 == 0:
                print("Consider neighbours", lemma, "and", neighbour, "AGWN IDs are", str(id1), "and", str(id2))
                print("AGWN cosine distance:", str(agwn_cos_distance_lemma_neighbour))
                print("DISSECT cosine distance:", str(dissect_cos_distance_id1_id2))

            if lemma == "ἐπιμέλεια" or lemma == "κομιδή":
                print("\tTest!")
                print("\tConsider neighbours", lemma, "and", neighbour, "AGWN IDs are", str(id1), "and", str(id2))
                print("\tAGWN cosine distance:", str(agwn_cos_distance_lemma_neighbour))
                print("\tDISSECT cosine distance:", str(dissect_cos_distance_id1_id2))

            dissect_distances_shared_dissect_neighbours.append(dissect_cos_distance_lemma_neighbour)

# print out results:

summary_dissect_agwn_distances_file = open(os.path.join(dir_out, summary_dissect_agwn_distances_file_name), 'w')

# Synonyms:

print("length of agwn_distances_shared_agwn_synonyms:", str(len(agwn_distances_shared_agwn_synonyms)))
summary_dissect_agwn_distances_file.write("agwn_distances_shared_agwn_synonyms:")
summary_dissect_agwn_distances_file.write(str(agwn_distances_shared_agwn_synonyms) + "\n")

print("length of dissect_distances_shared_agwn_synonyms:", str(len(dissect_distances_shared_agwn_synonyms)))
summary_dissect_agwn_distances_file.write("dissect_distances_shared_agwn_synonyms:")
summary_dissect_agwn_distances_file.write(str(dissect_distances_shared_agwn_synonyms) + "\n")

# Pearson's correlation coefficient:
corr_p, p_value_p = pearsonr(agwn_distances_shared_agwn_synonyms, dissect_distances_shared_agwn_synonyms)
print("Pearson's correlation:", str(corr_p), str(p_value_p))
summary_dissect_agwn_distances_file.write("Pearson's correlation:" + str(corr_p) + ", p-value:" + str(p_value_p) + "\n")

# Spearman's correlation coefficient:
corr_s, p_value_s = spearmanr(agwn_distances_shared_agwn_synonyms, dissect_distances_shared_agwn_synonyms)
print("Spearmnan's correlation:", str(corr_s), str(p_value_s))
summary_dissect_agwn_distances_file.write("Spearman's correlation:" + str(corr_s) + ", p-value:" + str(p_value_s) + "\n")

# Neighbours:

print("length of agwn_distances_shared_dissect_neighbours:", str(len(agwn_distances_shared_dissect_neighbours)))
summary_dissect_agwn_distances_file.write("agwn_distances_shared_dissect_neighbours:")
summary_dissect_agwn_distances_file.write(str(agwn_distances_shared_dissect_neighbours) + "\n")

print("length of dissect_distances_shared_dissect_neighbours:", str(len(dissect_distances_shared_dissect_neighbours)))
summary_dissect_agwn_distances_file.write("dissect_distances_shared_dissect_neighbours:")
summary_dissect_agwn_distances_file.write(str(dissect_distances_shared_dissect_neighbours) + "\n")

# Pearson's correlation coefficient:
corr_p, p_value_p = pearsonr(agwn_distances_shared_dissect_neighbours, dissect_distances_shared_dissect_neighbours)
print("Pearson's correlation:", str(corr_p), str(p_value_p))
summary_dissect_agwn_distances_file.write("Pearson's correlation:" + str(corr_p) + ", p-value:" + str(p_value_p) + "\n")

# Spearman's correlation coefficient:
corr_s, p_value_s = spearmanr(agwn_distances_shared_dissect_neighbours, dissect_distances_shared_dissect_neighbours)
print("Spearmnan's correlation:", str(corr_s), str(p_value_s))
summary_dissect_agwn_distances_file.write("Spearman's correlation:" + str(corr_s) + ", p-value:" + str(p_value_s) + "\n")


summary_dissect_agwn_distances_file.close()

# -------------------------------------------------------------------------------------
# 3.	Synonym/neighbour comparison
# a.	Compare overlap between synsets in AGWN space and neighbours in DISSECT space
# -------------------------------------------------------------------------------------


print(
    "---------------------------------------------------------\nThird evaluation approach: overlap between synsets in AGWN and neighbour sets in DISSECT spaces\n-------------------------------------------------")

# Define synsets in AGWN space containing shared lemmas:

print("Define lemma-to-synset mapping")

lemma2synset = dict() # maps a shared lemma to the list of its AGWN synonyms:

count_n = 0
for synset_id in synsets:

    count_n += 1

    if count_n % 100 == 0:
        print(str(count_n), synset_id, "out of", len(synsets))

    synonyms = synsets[synset_id]
    for synonym in synonyms:
        if synonym in agwn_dissect_5neighbours:
            synonyms.remove(synonym)
            lemma2synset[synonym] = synonyms

        if (synonym == "κομιδή") or (synonym == "ἐπιμέλεια"):
            print("\tTest!")
            print("\t", synset_id, synonym, str(lemma2synset[synonym]))

print("Examples:")
print("Synonyms of κομιδή: ", str(lemma2synset["κομιδή"]))
print("Synonyms of ἐπιμέλεια: ", str(lemma2synset["ἐπιμέλεια"]))


# Define sets of neighbours in DISSECT space:

print("Define lemma-to-neighbourset mapping")

lemma2neighbourset = dict() # maps a shared lemma to the list of its DISSECT neighbours:

count_n = 0
for lemma, neighbour in lemma_neighbour2distance:
    neighbours = list()
    count_n += 1

    if lemma in agwn_dissect_5neighbours:

        if count_n % 100 == 0:
            print(str(count_n), "lemma:", lemma, "out of", len(lemma_neighbour2distance))

        if lemma in lemma2neighbourset:
            neighbours_this_lemma = lemma2neighbourset[lemma]
            neighbours_this_lemma.append(neighbour)
            lemma2neighbourset[lemma] = neighbours_this_lemma
        else:
            lemma2neighbourset[lemma] = [neighbour]

        if (lemma == "κομιδή") or (lemma == "ἐπιμέλεια"):
            print("\tTest!")
            print("\t", lemma, str(lemma2neighbourset[lemma]))

print("Examples:")
print("Neighbours of κομιδή: ", str(lemma2neighbourset["κομιδή"]))
print("Neighbours of ἐπιμέλεια: ", str(lemma2neighbourset["ἐπιμέλεια"]))


# Calculate overlap between synsets and neighbour sets:

lemma2overlap_ratio = dict()  # maps a shared lemma to the overlap ratio between its AGWN synonyms and its DISSECT neighbours, divided by the union# :

print("Finding overlap between synsets and neighboursets")

count_n = 0
overlap_ratios = list() # list of overlap ratio values

for lemma in agwn_dissect_5neighbours:
    count_n += 1
    if count_n % 100 == 0:
        print(str(count_n), "lemma:", lemma)


    try:
        synset = set(lemma2synset[lemma])
        neighbourset = set(lemma2neighbourset[lemma])
        union = list(set(synset + neighbourset))
        overlap_ratio = len(list(synset.intersection(neighbourset)))/len(union)
    except:
        overlap_ratio = 0

    lemma2overlap_ratio[lemma] = overlap_ratio
    overlap_ratios.append(overlap_ratio)

    if (lemma == "κομιδή") or (lemma == "ἐπιμέλεια"):
        print("\tTest!")
        print("\t", lemma, str(lemma2overlap_ratio[lemma]))

print("Examples:")
print("Overlap ratio for κομιδή:", str(lemma2overlap_ratio["κομιδή"]))
print("Overlap ratio for ἐπιμέλεια:", str(lemma2overlap_ratio["ἐπιμέλεια"]))

# print out results:

summary_dissect_agwn_overlap_file = open(os.path.join(dir_out, summary_dissect_agwn_overlap_file_name), 'w')

overlap_ratios = np.array(overlap_ratios, dtype=np.float32)

mean_overlaps = np.mean(overlap_ratios)
std_overlaps = overlap_ratios.std()
min_overlaps = overlap_ratios.min()
max_overlaps = overlap_ratios.max()
median_overlaps = np.median(overlap_ratios)
perc25_overlaps = np.percentile(overlap_ratios, 25)
perc75_overlaps = np.percentile(overlap_ratios, 75)

summary_dissect_agwn_overlap_file.write("Mean of overlap ratios:" + str(mean_overlaps) + "\n")
summary_dissect_agwn_overlap_file.write("STD of overlap ratios:" + str(std_overlaps) + "\n")
summary_dissect_agwn_overlap_file.write("Min of overlap ratios:" + str(min_overlaps) + "\n")
summary_dissect_agwn_overlap_file.write("Max of overlap ratios:" + str(max_overlaps) + "\n")
summary_dissect_agwn_overlap_file.write("Median of overlap ratios:" + str(median_overlaps) + "\n")
summary_dissect_agwn_overlap_file.write("25th percentile of overlap ratios:" + str(perc25_overlaps) + "\n")
summary_dissect_agwn_overlap_file.write("75th percentile of overlap ratios:" + str(perc75_overlaps) + "\n")

summary_dissect_agwn_overlap_file.close()