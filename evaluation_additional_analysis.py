# # -*- coding: utf-8 -*- Author: Barbara McGillivray Date: 21/08/2019 Python version: 3 Script version: 1.0 Script
# for adding additional information about the evaluation of the Ancient Greek semantic spaces based on the reviewers' comments.


# ----------------------------
# Initialization
# ----------------------------


# Import modules:

import os
import csv
from collections import defaultdict
import numpy as np
import collections

from scipy.spatial import distance
from numpy import dot
from numpy.linalg import norm
import math

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import datetime

# User input:
istest = input("Is this a test? Reply yes or no.")

if istest == "yes":
    windows = [1]
    freq_thresholds = [1]
    lexicons = ["POLLUX"]
else:
    windows = [1, 5, 10]
    freq_thresholds = [1, 20, 50, 100]
    lexicons = ["AGWN", "POLLUX", "SCHMIDT"]

# Directory and file names:

directory = os.path.join("/Users", "bmcgillivray", "Documents", "OneDrive", "The Alan Turing Institute",
                         "Martina Astrid Rodda - MAR dphil project")
dir_out = os.path.join(directory, "Evaluation", "output_after_review")

# Output files:

coverage_file_name = "coverage_statistics_overlap_Lexicons-semantic-spaces.csv"

if istest == "yes":
    coverage_file_name = coverage_file_name.replace(".csv", "_test.csv")

# Print header of output file:

coverage_file = open(os.path.join(dir_out, coverage_file_name), 'w', encoding="UTF-8")

coverage_file.write("Window\tFreq_threshold\tLexicon\tRange number of overlapping lemmas\tFrequency of values\t" +
                    "Mean of number of overlapping lemmas\t" +
                    "Standard deviation of number of overlapping lemmas\t" +
                    "Median number of overlapping lemmas\t" +
                    "25th percentile of number of overlapping lemmas\t" +
                    "75th percentile of number of overlapping lemmas\t" +
                    "Number of rows\n")

# Read summary files:

for lexicon in lexicons:
    for window in windows:
        for freq_threshold in freq_thresholds:

            if (lexicon == "AGWN" and freq_threshold > 1) or lexicon != "AGWN":
                # Input files:

                dir_in = os.path.join(directory, "Evaluation", "output", "semantic-space-w"
                                      + str(window) + "_t" + str(freq_threshold), "Lexicon_" + lexicon)

                file_in_lexicon_cooccurrence_name = "Lexicon_" + lexicon + "_co-occurrences.csv"  # matrix with 0s
                # and 1s depending on whether two lemmas
                # co-occur in the same Lexicon synset
                file_in_dissect_neighbour_name = "semantic-space_w" + str(window) + "_t" + str(
                    freq_threshold) + "_neighbours_lemmas.txt"
                file_in_dissect_neighbour_distances_name = "semantic-space_w" + str(window) + "_t" + str(
                    freq_threshold) + "_neighbours_distances.txt"
                file_in_shared_lemmas_name = "Lexicon_" + lexicon + "-semantic-space_w" + str(window) + "_t" + str(
                    freq_threshold) + "_shared-lemmas.txt"
                file_in_lexicon_vocabulary_name = "Lexicon_" + lexicon + "_lemmas.txt"
                summary_stats_dissect_neighbours_file_name = "summary_statistics_distance_Lexicon_" + lexicon + \
                                                             "-semantic-space_w" + str(window) + "_t" + \
                                                             str(freq_threshold) + "_neighbours" + ".txt"
                summary_dissect_lexicon_distances_file_name = "summary_comparison_distances_Lexicon_" + lexicon + \
                                                              "_semantic-space_w" + str(window) + "_t" + \
                                                              str(freq_threshold) + "_neighbours" + ".txt"
                dissect_lexicon_distances_synonyms_file_name = "distances_synonyms_Lexicon_" + lexicon + "_semantic" \
                                                                                                         "-space_w" + \
                                                               str(
                                                                   window) + "_t" + \
                                                               str(freq_threshold) + "_neighbours" + ".txt"
                dissect_lexicon_distances_neighbours_file_name = "distances_neighbours_Lexicon_" + lexicon + \
                                                                 "_semantic-space_w" + str(window) + "_t" + \
                                                                 str(freq_threshold) + "_neighbours" + ".txt"
                summary_dissect_lexicon_overlap_file_name = "summary_overlap_Lexicon_" + lexicon + \
                                                            "_semantic-space_w" + str(window) + "_t" + \
                                                            str(freq_threshold) + "_neighbours" + ".txt"

                # Initialize objects:
                # list of numbers of overlapping lemmas:
                numbers_overlapping_lemmas = list()

                # ----------------- Coverage of overlaps:
                # –----------------------------------------------------------------------------- Read overlap file
                # and extract statistics about overlap coverage, i.e. how many lemmas are in the overlap between the
                # synsets and the neighboursets:
                # ----------------------------------------------------------------------------------------------------------------------

                summary_dissect_lexicon_overlap_file = open(
                    os.path.join(dir_in, summary_dissect_lexicon_overlap_file_name), 'r')

                summary_dissect_lexicon_overlap_reader = csv.reader(summary_dissect_lexicon_overlap_file,
                                                                    delimiter="\t")
                next(summary_dissect_lexicon_overlap_reader)  # This skips the first row of the CSV file

                for row in summary_dissect_lexicon_overlap_reader:
                    if not row[0].startswith("Mean") and not row[0].startswith("STD") \
                            and not row[0].startswith("Min") and not row[0].startswith("Max") \
                            and not row[0].startswith("Median") and not row[0].startswith("25") and not row[
                        0].startswith("75"):
                        # print("Reading " + str(row))
                        overlapping_lemmas = row[3]
                        # print("Overlapping lemmas:" + str(overlapping_lemmas))
                        if overlapping_lemmas == "[]":
                            number_overlapping_lemmas = 0
                        else:
                            number_overlapping_lemmas = len(overlapping_lemmas.split(","))

                        numbers_overlapping_lemmas.append(number_overlapping_lemmas)

                summary_dissect_lexicon_overlap_file.close()

                numbers_overlapping_lemmas = np.array(numbers_overlapping_lemmas, dtype=np.float32)

                mean_overlapping_lemmas = np.mean(numbers_overlapping_lemmas)
                std_overlapping_lemmas = numbers_overlapping_lemmas.std()
                min_overlapping_lemmas = numbers_overlapping_lemmas.min()
                max_overlapping_lemmas = numbers_overlapping_lemmas.max()
                median_overlapping_lemmas = np.median(numbers_overlapping_lemmas)
                perc25_overlapping_lemmas = np.percentile(numbers_overlapping_lemmas, 25)
                perc75_overlapping_lemmas = np.percentile(numbers_overlapping_lemmas, 75)
                frequencies_values = collections.Counter(numbers_overlapping_lemmas)
                coverage_file.write(
                    "{0}\t{1}\t{2}\t[{3},{4}]\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\n".format(str(window),
                                                                                             str(freq_threshold),
                                                                                             lexicon,
                                                                                             str(
                                                                                                 min_overlapping_lemmas),
                                                                                             str(
                                                                                                 max_overlapping_lemmas),
                                                                                             frequencies_values,
                                                                                             str(
                                                                                                 mean_overlapping_lemmas),
                                                                                             str(
                                                                                                 std_overlapping_lemmas),
                                                                                             str(
                                                                                                 median_overlapping_lemmas),
                                                                                             str(
                                                                                                 perc25_overlapping_lemmas),
                                                                                             str(
                                                                                                 perc75_overlapping_lemmas),
                                                                                             str(len(
                                                                                                 numbers_overlapping_lemmas))))

coverage_file.close()
