#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import libraries
from lxml import etree
import argparse
from collections import defaultdict
import os
from os import listdir
from os.path import isfile, join
import csv

#todo: for parameters: use argparse
    #remember to update all parameter-defined elements below, 'welp!'

#open xml files with ElementTree
    #todo: iterate over all files in corpus
import xml.etree.ElementTree as ET

#todo: set up window size parameter and delete this
window_size = 1

#todo: set up frequency threshold parameter and delete this
freq_threshold = 50

#todo: path variable
path = "C:/Users/mrodda/OneDrive - The Alan Turing Institute/desktop files/Diorisis"

#define output path
path_out = "C:/Users/mrodda/OneDrive - The Alan Turing Institute/MAR dphil project"

#define output file
out_file = 'debug_w'+str(window_size)+'_t'+str(freq_threshold)+'.sm'
out_cols = 'debug_w'+str(window_size)+'_t'+str(freq_threshold)+'.cols'
out_rows = 'debug_w'+str(window_size)+'_t'+str(freq_threshold)+'.rows'

#step: import stoplist
#todo: add link to AVatri github repo
#parameter: exclude_stops
    # todo: set parameter

    # """This list comes from the Perseus Hopper source [http://sourceforge.net/projects/perseus-hopper],
    # found at "/sgml/reading/build/stoplists", though this only contained acute accents on the ultima.
    # There has been added to this grave accents to the ultima of each.
    # Perseus source is made available under the Mozilla Public License 1.1 (MPL 1.1) [http://www.mozilla.org/MPL/1.1/]."""
    # 	__author__ = ['Kyle P. Johnson <kyle@kyle-p-johnson.com>']
    # 	__license__ = 'GPL License.'
    #Vatri: added support for tonos vs oxia acute accent
    #MAR: added 'None'
STOPS_LIST = ['αὐτὸς',
                           'αὐτός',
                           'γε',
                           'γὰρ',
                           'γάρ',
                           "δ'",
                           'δαὶ',
                           'δαὶς',
                           'δαί',
                           'δαίς',
                           'διὰ',
                           'διά',
                           'δὲ',
                           'δέ',
                            'δὴ',
                           'δή',
                           'εἰ',
                           'εἰμὶ',
                           'εἰμί',
                           'εἰς',
                           'εἴμι',
                           'κατὰ',
                           'κατά',
                           'καὶ',
                           'καί',
                           'μετὰ',
                           'μετά',
                           'μὲν',
                           'μέν',
                           'μὴ',
                           'μή',
                           'οἱ',
                           'οὐ',
                           'οὐδεὶς',
                           'οὐδείς',
                           'οὐδὲ',
                           'οὐδέ',
                           'οὐκ',
                           'οὔτε',
                           'οὕτως',
                           'οὖν',
                           'οὗτος',
                           'παρὰ',
                           'παρά',
                           'περὶ',
                           'περί',
                           'πρὸς',
                           'πρός',
                           'σὸς',
                           'σός',
                           'σὺ',
                           'σὺν',
                           'σύ',
                           'σύν',
                           'τε',
                           'τι',
                           'τις',
                           'τοιοῦτος',
                           'τοὶ',
                           'τοί',
                           'τοὺς',
                           'τούς',
                           'τοῦ',
                           'τὰ',
                           'τά',
                           'τὴν',
                           'τήν',
                           'τὶ',
                           'τὶς',
                           'τί',
                           'τίς',
                           'τὸ',
                           'τὸν',
                           'τό',
                           'τόν',
                           'τῆς',
                           'τῇ',
                           'τῶν',
                           'τῷ',
                           "ἀλλ'",
                           'ἀλλὰ',
                           'ἀλλά',
                           'ἀπὸ',
                           'ἀπό',
                           'ἂν',
                           'ἄλλος',
                           'ἄν',
                           'ἄρα',
                           'ἐγὼ',
                           'ἐγώ',
                           'ἐκ',
                           'ἐξ',
                           'ἐμὸς',
                           'ἐμός',
                           'ἐν',
                           'ἐπὶ',
                           'ἐπί',
                           'ἐὰν',
                           'ἐάν',
                           'ἑαυτοῦ',
                           'ἔτι',
                           'ἡ',
                           'ἢ',
                           'ἤ',
                           'ὁ',
                           'ὃδε',
                           'ὃς',
                           'ὅδε',
                           'ὅς',
                           'ὅστις',
                           'ὅτι',
                           'ὑμὸς',
                           'ὑμός',
                           'ὑπὲρ',
                           'ὑπέρ',
                           'ὑπὸ',
                           'ὑπό',
                           'ὡς',
                           'ὥστε',
                           'ὦ',
                           'ξύν',
                           'ξὺν',
                           'σύν',
                           'σὺν',
                           'τοῖς',
                           'τᾶς',
                           'αὐτός',
                           'γάρ',
                           'δαί',
                           'δαίς',
                           'διά',
                           'δέ',
                           'δή',
                           'εἰμί',
                           'κατά',
                           'καί',
                           'μετά',
                           'μέν',
                           'μή',
                           'οὐδείς',
                           'οὐδέ',
                           'παρά',
                           'περί',
                           'πρός',
                           'σός',
                           'σύ',
                           'σύν',
                           'τοί',
                           'τούς',
                           'τά',
                           'τήν',
                           'τί',
                           'τίς',
                           'τό',
                           'τόν',
                           'ἀλλά',
                           'ἀπό',
                           'ἐγώ',
                           'ἐμός',
                           'ἐπί',
                           'ἐάν',
                           'ὑμός',
                           'ὑπέρ',
                           'ὑπό',
                           'None',]

files = [f for f in listdir(path) if isfile(join(path, f))]

#initialise dictionary for frequency count
wfreq = {}

#initialise multidimensional dictionary for cooccurence count
ccfreq = defaultdict(lambda : defaultdict(int))

for file in files:
    print(file)
    tree = ET.parse(join(path,file))
    root = tree.getroot()

#parameter: exclude BC/AD
#todo: set up parameter
    #extract date from teiHeader/profileDesc/creation/date

#step: extract lemmas from xml files
    #todo: for None lemmatag extract word form
    #output: lists
    for sentence in root.findall("./text/body/"):
        sentencelemmas = []
        for lemmatag in sentence.findall("./word/"):
            lemma = lemmatag.get('entry')
            sentencelemmas.append(lemma)
        # print check
            # print(sentencelemmas)

        #step: frequency count
            #output: dictionary
        #if word not in stoplist count word freq across lists
        for lemma in sentencelemmas:
            if not lemma in STOPS_LIST:
            #reads lists and updates count dictionary
                if lemma in wfreq:
                    wfreq[lemma] = wfreq[lemma] + 1
                else:
                    wfreq[lemma] = 1
        
        #step: extract cooccurrence frequency from lemma string
            #output: multidimensional dictionary ((l1: (l2: frequency); (l3: frequency), etc.)
        #parameter: set context window
        for i in range(len(sentencelemmas)):
            if not sentencelemmas[i] in STOPS_LIST:          
                for j in range(max(0, i-window_size), min(i+window_size+1, len(sentencelemmas))):
                    if j != i: 
                        if not sentencelemmas[j] in STOPS_LIST:
                            w1, w2 = ([sentencelemmas[i], sentencelemmas[j]])
                            ccfreq[w1][w2] += 1

    #print checks
    #frequency count:
    # for lemma in wfreq:
    #     print("{}: {}".format(lemma, wfreq[lemma]))
    #cooccurrence pairs:
    # for pair in ccfreq:
    #     print("{}: {}".format(pair, ccfreq[pair]))

#initialise row/column lists for semantic space
rows = []
cols = []

#open output file
if not os.path.exists(path_out):
    os.makedirs(path_out)
with open(join(path_out, out_file), 'w', encoding = 'UTF-8') as output:
    outwriter = csv.writer(output, delimiter = '\t', quotechar = '|', quoting = csv.QUOTE_MINIMAL)

    # step: extract cooccurrence frequency for frequency-filtered lemmas
        #output: smaller multidimensional dictionary
        #parameter: set filter frequency
    for w1 in ccfreq:
        if wfreq[w1] >= freq_threshold and w1 is not None:
                rows.append(w1)
        for w2 in ccfreq[w1]:
            # print(w1, w2)
            #hack to filter out null results
            #todo: sort out null results properly
            #ISSUE: nope, we still have the null results
            if wfreq[w2] >= freq_threshold and w2 is not None:
                cols.append(w2)
                # print(w1,w2,ccfreq[w1][w2])
                outwriter.writerow([w1,w2,str(ccfreq[w1][w2])])

    rows = list(set(rows))
    cols = list(set(cols))

    print(len(rows))
    print(len(cols))
    print(len(list(set(rows) & set(cols))))

#open output file
with open(join(path_out, out_cols), 'w', encoding = 'UTF-8') as output_cols:
    for col in cols:
        output_cols.write(col+'\n')
    # outwriter_cols = csv.writer(output_cols, delimiter = '\t', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
    # output_cols.write('\n', join(cols))

#open output file
with open(join(path_out, out_rows), 'w', encoding = 'UTF-8') as output_rows:
    for row in rows:
        output_rows.write(row+'\n')
    # outwriter_rows = csv.writer(output_rows, delimiter = '\t', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
    # output_rows.write('\n', join(rows))