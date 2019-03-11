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

# Default parameters:

window_default = 5
freq_threshold_default = 2
istest_default = "yes"

# Parameters:

window = input("What is the window size? Leave empty for default (" + str(window_default) + ").") # 5 or 10 # size of context window for co-occurrence matrix
freq_threshold = input("What is the frequency threshold for vocabulary lemmas? Leave empty for default (" +
                       str(freq_threshold_default) + ").")# 2 or 50 # frequency threshold for lemmas in vocabulary
istest = input("Is this a test? Leave empty for default (" + str(istest_default) + ").")


if window == "":
    window = window_default

if freq_threshold == "":
    freq_threshold = freq_threshold_default

if istest == "":
    istest = istest_default

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


file_out_wn_cooccurrence = "wordnet_co-occurrences.csv" # matrix with 0s and 1s depending on whether two lemmas co-occur in the same WN synset

if istest == "yes":
    file_out_wn_cooccurrence = file_out_wn_cooccurrence.replace(".csv", "_test.csv")


# --------------------------------------
# Read WordNet file to collect synsets
# --------------------------------------

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
    if ((istest == "yes" and count_n < 1200) or (istest == "no" and count_n <= row_count_wn)) and (count_n > 1):

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

wn_vocabulary = list() # list of all lemmas in WordNet
wn_cooccurrence = defaultdict(dict) # multidimensional dictionary: maps each pair of lemmas to 1 if they co-occur in the same WN synset, and 0 otherwise

for synset_id in synsets:
    print("Synset ID:", synset_id)
    print("Definition:", synsetid2def[synset_id])
    print("Lemmas:", str(synsets[synset_id]))
    synsets_this_lemma = synsets[synset_id]
    for lemma1 in synsets_this_lemma:
        #print("Lemma1:", lemma1)
        if lemma1 not in wn_vocabulary:
            wn_vocabulary.append(lemma1)
            #print("Vocabulary so far:", str(wn_vocabulary))
        for lemma2 in synsets_this_lemma:
            #print("Lemma2:", lemma2)
            wn_cooccurrence[lemma1][lemma2] = 1
            #print("Co-occurrence:", str(wn_cooccurrence[lemma1][lemma2]))
            print("Lemma1:", lemma1, "Lemma2:", lemma2, "Co-occurrence:", str(wn_cooccurrence[lemma1][lemma2]))

wn_vocabulary = sorted(wn_vocabulary)
#print("Vocabulary:", str(wn_vocabulary))

# finish defining co-occurrence pairs:

for lemma1 in wn_vocabulary:
    for lemma2 in wn_vocabulary:
        #print("Co-occurrence of ", lemma1, "and", lemma2, ":")
        try:
            print("Co-occurrence of ", lemma1, "and", lemma2, ":", str(wn_cooccurrence[lemma1][lemma2]))
            #if lemma1 == "κτίσμα":
            #    print("Co-occurrence of ", lemma1, "and", lemma2, ":", str(wn_cooccurrence[lemma1][lemma2]))
        except KeyError:
            wn_cooccurrence[lemma1][lemma2] = 0
            #if lemma1 == "κτίσμα":
            #    print("Co-occurrence of ", lemma1, "and", lemma2, ":", str(wn_cooccurrence[lemma1][lemma2]))

# print co-occurrence matrix:
# TODO: check the right input format for correspondence analysis in Python
# https://pypi.org/project/mca/
# https://nbviewer.jupyter.org/github/esafak/mca/blob/master/docs/mca-BurgundiesExample.ipynb
# https://github.com/esafak/mca/blob/master/data/burgundies.csv
# https://github.com/esafak/mca/blob/master/docs/usage.rst

#for lemma1 in wn_vocabulary:
#    for lemma2 in wn_vocabulary:

#        wn_cooccurrence[lemma1][lemma2]

