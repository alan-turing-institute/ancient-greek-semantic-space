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


now = datetime.datetime.now()


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

# Parameters:

lines_read_testing = 1000  # lines read in test case

# Directory and file names:

directory = os.path.join("/Users", "bmcgillivray", "Documents", "OneDrive", "The Alan Turing Institute",
                         "Martina Astrid Rodda - MAR dphil project")
dir_ss = os.path.join(directory, "semantic_space", "sem_space_output")
dir_out = os.path.join(directory, "Evaluation", "output_after_review")


# Output files:

coverage_file_name = "coverage_statistics_overlap_Lexicons-semantic-spaces.csv"
average_inverse_ranks_file_name = "average-inverse-ranks_Lexicons-semantic-spaces.csv"
log_file_name = "log_file_Inverse-ranks" + str(now.strftime("%Y-%m-%d %H-%M")) + ".txt"
log_file_name = log_file_name.replace("/", "")


if istest == "yes":
    coverage_file_name = coverage_file_name.replace(".csv", "_test.csv")
    average_inverse_ranks_file_name = average_inverse_ranks_file_name.replace(".csv", "_test.csv")
    log_file_name = log_file_name.replace(".txt", "_test.txt")

# ----------------------------------------------------------------------------------------------------------------------
# Coverage analysis:
# ----------------------------------------------------------------------------------------------------------------------

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

                # Input directory and files:
                dir_in = os.path.join(directory, "Evaluation", "output", "semantic-space-w"
                                      + str(window) + "_t" + str(freq_threshold), "Lexicon_" + lexicon)
                summary_dissect_lexicon_overlap_file_name = "summary_overlap_Lexicon_" + lexicon + "_semantic-space_w" + str(
                    window) + "_t" + str(freq_threshold) + "_neighbours" + ".txt"

                # Initialize objects:
                # list of numbers of overlapping lemmas:
                numbers_overlapping_lemmas = list()

                # ----------------------------------------------------------------------------
                # Coverage of overlaps:
                # ----------------------------------------------------------------------------
                # Read overlap file
                # and extract statistics about overlap coverage, i.e. how many lemmas are in the overlap between the
                # synsets and the neighboursets:

                summary_dissect_lexicon_overlap_file = open(
                    os.path.join(dir_in, summary_dissect_lexicon_overlap_file_name), 'r')

                summary_dissect_lexicon_overlap_reader = csv.reader(summary_dissect_lexicon_overlap_file,
                                                                    delimiter="\t")
                next(summary_dissect_lexicon_overlap_reader)  # This skips the first row of the CSV file

                for row in summary_dissect_lexicon_overlap_reader:
                    if not row[0].startswith("Mean") and not row[0].startswith("STD") \
                            and not row[0].startswith("Min") and not row[0].startswith("Max") \
                            and not row[0].startswith("Median") and not row[0].startswith("25") \
                            and not row[0].startswith("75"):
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
                        str(freq_threshold), lexicon, str(min_overlapping_lemmas), str(max_overlapping_lemmas),
                        frequencies_values, str(mean_overlapping_lemmas), str(std_overlapping_lemmas),
                        str(median_overlapping_lemmas), str(perc25_overlapping_lemmas),
                        str(perc75_overlapping_lemmas), str(len(numbers_overlapping_lemmas))))

coverage_file.close()

# ------------------------------------------
# Average Inverse rank:
# ------------------------------------------

log_file = open(os.path.join(dir_out, log_file_name), 'w', encoding="UTF-8")

now = datetime.datetime.now()

log_file.write(str(now.strftime("%Y-%m-%d %H:%M")) + "\n")

print("Calculating average inverse rank...")


# Print header of output file:

average_inverse_ranks_file = open(os.path.join(dir_out, average_inverse_ranks_file_name), 'w', encoding="UTF-8")

average_inverse_ranks_file.write("Window\tFreq_threshold\tLexicon\tAverage inverse rank\n")

# Read summary files:

for lexicon in lexicons:
    log_file.write("Lexicon:" + lexicon + "\n")
    print("Lexicon:" + lexicon)
    for window in windows:
        log_file.write("Window: " + str(window) + "\n")
        print("Window: " + str(window))
        for freq_threshold in freq_thresholds:
            log_file.write("Frequency threshold: " + str(freq_threshold) + "\n")
            print("Frequency threshold: " + str(freq_threshold))

            if (lexicon == "AGWN" and freq_threshold > 1) or lexicon != "AGWN":

                # Input directories and files:

                dir_in = os.path.join(directory, "Evaluation", "output", "semantic-space-w"
                                      + str(window) + "_t" + str(freq_threshold), "Lexicon_" + lexicon)
                dir_ss_neighbours = os.path.join(dir_ss, "ppmi_svd300", "w" + str(window), "w" + str(window) + "_nns")

                dissect_lexicon_distances_neighbours_file_name = "distances_neighbours_Lexicon_" + lexicon + \
                                                                 "_semantic-space_w" + str(window) + "_t" + \
                                                                 str(freq_threshold) + "_neighbours" + ".txt"
                neighbours_file_name = "NEIGHBOURS.matrix_w" + str(window) + "_t" + str(
                    freq_threshold) + ".rows.CORE_SS.matrix_w" \
                                       + str(window) + "_t" + str(freq_threshold) + ".ppmi.svd_300.cos"

                # Output directories:

                dir_out2 = os.path.join(dir_out, "semantic-space-w"
                                        + str(window) + "_t" + str(freq_threshold), "Lexicon_" + lexicon)

                if not os.path.exists(dir_out2):
                    os.makedirs(dir_out2)

                # Output files:

                # For each corpus neighbour of a lemma, this file contains its rank and inverse rank:
                inverse_ranks_file_name = "ranks_neighbours_w" + str(window) + "_t" + \
                                          str(freq_threshold) + ".csv"

                # For each corpus neighbour of a shared lemma and which is also a synonym in the lexicon,
                # this file contains its rank and inverse rank:
                dissect_lexicon_ranks_neighbours_file_name = "ranks_neighbours_Lexicon_" + lexicon + \
                                                             "_semantic-space_w" + str(window) + "_t" + \
                                                             str(freq_threshold) + "_neighbours" + ".txt"

                # For each shared lemma, this file contains the sum of the inverse ranks of all its corpus neighbours
                # which are also synonyms in the lexicon:
                #dissect_lexicon_sumranks_neighbours_file_name = "average-sum-ranks_neighbours-lemmas_Lexicon_" + \
                #                                                lexicon + "_semantic-space_w" + str(window) + "_t" + \
                #                                                str(freq_threshold) + "_neighbours" + ".txt"

                # Print header lines:

                inverse_ranks_file = open(os.path.join(dir_out2, inverse_ranks_file_name), 'w',
                                          encoding="UTF-8")

                inverse_ranks_file.write("Lemma\tNeighbour\trank\tinverse rank\n")

                dissect_lexicon_ranks_neighbours_file = open(os.path.join(dir_out2,
                                                                          dissect_lexicon_ranks_neighbours_file_name),
                                                             'w',
                                                             encoding="UTF-8")

                dissect_lexicon_ranks_neighbours_file.write("Lemma\tNeighbour\trank\tinverse rank\n")

                #dissect_lexicon_sumranks_neighbours_file = open(os.path.join(dir_out2,
                #                                                             dissect_lexicon_sumranks_neighbours_file_name),
                #                                                'w', encoding="UTF-8")
                #dissect_lexicon_sumranks_neighbours_file.write("Lemma\tsum of inverse ranks\n")

                # Initialize objects:

                lemma_neighbour2rank = dict()
                average_inverse_ranks = 0

                # Read corpus neighbours file:
                log_file.write("Reading corpus neighbours file..." + "\n")

                neighbours_file = open(os.path.join(dir_ss_neighbours, neighbours_file_name), 'r', encoding="UTF-8")
                row_count_neighbours = sum(1 for line in neighbours_file)
                neighbours_file.close()

                neighbours_file = open(os.path.join(dir_ss_neighbours, neighbours_file_name), 'r', encoding="UTF-8")
                count_n = 0
                rank = 0
                inv_rank = 0
                lemma = ""

                for line in neighbours_file:
                    count_n += 1

                    #if (istest == "yes" and count_n < lines_read_testing) or (
                    #                istest == "no" and count_n <= row_count_neighbours):

                    if count_n % 10000 == 0:
                        print("Reading neighbours in DISSECT, line " + str(count_n))
                    log_file.write("Reading neighbours in DISSECT, line " + str(count_n) + "\n")
                    line = line.rstrip('\n')
                    if not line.startswith("\t"):
                        lemma = line
                        rank = 0
                    else:
                        fields = line.split("\t")
                        neighbour_distance = fields[1]
                        neighbour_distance_fields = neighbour_distance.split(" ")
                        neighbour = neighbour_distance_fields[0]
                        neighbour_distance = neighbour_distance_fields[1]
                        if neighbour != lemma:
                            rank += 1
                            lemma_neighbour2rank[lemma, neighbour] = rank
                            inv_rank = float(1 / rank)
                            #log_file.write("Lemma: " + lemma + "\n")
                            #log_file.write("\tNeighbour " + neighbour + "\n")
                            #log_file.write("\t\tRank: " + str(rank) + "\n")
                            #log_file.write("\t\t\tInverse rank: " + str(inv_rank) + "\n")
                            inverse_ranks_file.write(
                                lemma + "\t" + neighbour + "\t" + str(rank) + "\t" + str(inv_rank) + "\n")

                neighbours_file.close()
                inverse_ranks_file.close()

                # Read distributions of lexicon distances and DISSECT distances for shared lemmas:
                log_file.write("Read distributions of lexicon distances and DISSECT distances for shared lemmas...\n")

                dissect_lexicon_distances_neighbours_file = open(
                    os.path.join(dir_in, dissect_lexicon_distances_neighbours_file_name), 'r', encoding="UTF-8")
                dissect_lexicon_distances_neighbours_reader = csv.reader(dissect_lexicon_distances_neighbours_file,
                                                                         delimiter="\t")
                next(dissect_lexicon_distances_neighbours_reader)  # This skips the first row of the CSV file

                count_n = 0
                lemma_old = ""
                sum_inv_ranks_lemma = 0
                sum_inv_ranks = 0
                num_lemmas = 0
                rank_neighbour_lemma = 0

                for row in dissect_lexicon_distances_neighbours_reader:
                    count_n += 1
                    if (istest == "yes" and count_n < lines_read_testing) or (istest == "no"):
                        lemma = row[0]
                        if count_n == 1:
                            lemma_old = lemma
                            num_lemmas = 1
                        neighbour = row[2]
                        lexicon_distance = row[4]
                        dissect_distance = row[5]
                        rank_neighbour_lemma = lemma_neighbour2rank[lemma, neighbour]
                        inv_rank = float(1 / rank_neighbour_lemma)
                        log_file.write("Shared lemma: " + lemma + "\n")
                        log_file.write("\tNeighbour: " + neighbour + "\n")
                        log_file.write("\t\tRank: " + str(rank_neighbour_lemma) + "\n")
                        log_file.write("\t\t\tInverse rank: " + str(inv_rank) + "\n")
                        if lemma == lemma_old:
                            sum_inv_ranks_lemma += inv_rank
                            log_file.write("Sum of inverse ranks for lemma " + lemma + ": " + str(sum_inv_ranks_lemma) + "\n")
                        else:
                            #dissect_lexicon_sumranks_neighbours_file.write(lemma + "\t" + str(sum_inv_ranks_lemma) + "\n")
                            sum_inv_ranks_lemma = inv_rank
                            sum_inv_ranks += sum_inv_ranks_lemma
                            num_lemmas += 1
                            log_file.write(str(num_lemmas) + " lemmas so far\n")
                            log_file.write("Sum of inverse ranks for lemma " + lemma + " so far: " + str(sum_inv_ranks_lemma) + "\n")
                            #sum_inv_ranks_lemma = 0
                        dissect_lexicon_ranks_neighbours_file.write(lemma + "\t" + neighbour + "\t" +
                                                              str(rank_neighbour_lemma) + "\t" + str(inv_rank) + "\n")

                        lemma_old = lemma

                dissect_lexicon_distances_neighbours_file.close()
                dissect_lexicon_ranks_neighbours_file.close()

                # Average inverse ranks:
                average_inverse_ranks = float(sum_inv_ranks / num_lemmas)
                log_file.write("Average inverse ranks = " + str(sum_inv_ranks) + "/" + str(num_lemmas) + " = " +
                               str(average_inverse_ranks)  + "\n")

                # Write to summary file:

                average_inverse_ranks_file.write(
                    "{0}\t{1}\t{2}\t{3}\n".format(str(window), str(freq_threshold), lexicon,
                                                  str(average_inverse_ranks)))

coverage_file.close()
