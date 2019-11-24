# Ancient Greek Semantic Spaces

Code relative to the project on Ancient Greek semantic space. 

Project team (in alphabetical order): Barbara McGillivray (Turing/Cambridge), Philomen Probert (Oxford), Martina Astrid Rodda (Oxford/Turing), enrichment student at the Turing from January to June 2019.

## corpusprocess_2.0.py

This Python 3 script was developed by Martina Astrid Rodda in February-April 2019. It takes as input a series of parameter values, as well as .xml files from the Diorisis Ancient Greek corpus, and returns a series of files that can be used as input for the DISSECT tool.

Use:

1. Set window size and frequency filter on lines 18-19.
2. Set input and output path on lines 22, 25.

The script produces three output files (a sparse matrix, .sm, plus a .rows and .cols file indexing the matrix) that can be used to build a semantic space using DISSECT (https://github.com/composes-toolkit/dissect).

## pollux_lexicon.txt and schmidt_lexicon.txt

These two text files contain lists of lemmas used as a benchmark for the accuracy of the semantic space.

The former, compiled by Philomen Probert, is from Pollux's *Onomasticon* (2nd century AD), and lists 'headwords' used to search for nouns that are listed as synonyms of said headwords by Pollux.

The latter, compiled by Martina Astrid Rodda is from J.H.H. Schmidt's *Synonymik der Griechischen Sprache* (1876-1886), and lists all of the nouns in each section of Schmidt's dictionary, indexed by an English definition for each section.

These files have been used as input for evaluation_text_processing.py.

## evaluation_text_processing.py

This Python 3 script was developed by Barbara McGillivray in March-April 2019. It takes as input a series of parameter values, as well as files relative to various semantic spaces created by the DISSECT tool on the Diorisis Ancient Greek corpus and files relative to three Ancient Greek lexicons, and returns a series of files containing the results of three evaluation approaches.

The script can be run on a local computer, or on a linux virtual machine. 

Use:

1. Change definition of main directory in lines 77-81.
2. Run script from command line: python evaluation_text_processing.py
3. Enter values of parameters:

  a. "What is the window size?". Enter 1, 5, or 10, depending on the size of context window for co-occurrence matrix.
  
  b. "What is the frequency threshold for vocabulary lemmas?" Enter 1, 20, 50, or 100, depending on the frequency threshold for lemmas in vocabulary used to create the semantic space.
  
  c. "Is this a test?". Enter yes or no, depending on whether you want to run the script in test mode (i.e. only reading the first lines of the input files) or not.
  
  d. "Which lexicon do you want to consider?" Enter AGWN, SCHMIDT, or POLLUX, depending on which lexicon you're interested in comparing the semantic space with.
  
  e. "Do you want to follow the first evaluation approach?". Enter yes or no, depending on whether you want to run the first evaluation approach or not.
  
  f. "Do you want to follow the second evaluation approach?". Enter yes or no, depending on whether you want to run the second evaluation approach or not.
  
  g. "Do you want to follow the third evaluation approach?". Enter yes or no, depending on whether you want to run the third evaluation approach or not.
  
  The output files are saved in folders like .../evaluation/output/semantic-space-w1_t20/SCHMIDT (this is an example for window 1, frequency threshold 20, Schmidt's lexicon), and contain:
  
  * summary_statistics_distance_Lexicon_SCHMIDT-semantic-space_w1_t20_neighbours.txt: various statistics about cosine distance in DISSECT space. You can ignore it.
  
  * summary_comparison_distances_Lexicon_SCHMIDT_semantic-space_w1_t20_neighbours.txt: results of Pearson and Spearman correlation tests (see article).
  
  * summary_overlap_Lexicon_SCHMIDT_semantic-space_w1_t20_neighbours.txt: synsets, neighboursets, overlap, precision and recall by lemma, and summary statistics of precision and recall (see article).


## evaluation_analysis.Rmd

This R markdown script was developed by Barbara McGillivray in April 2019 and runs on RStudio. It analyses the results of the third evaluation approach (see below) on all combinations of parameters for the semantic spaces, to see if there is any association between precision/recall metrics and polysemy/frequency of lemmas. 

Use:

1. Change definition of main directory (variable "path") on line 29.
2. Update chunk on line 265 ff. by uncommenting lines 265, 267, and 269 and commenting lines 266, 268, and 270.
3. Knit file and check evaluation_analysis.html.


## visualization_evaluation.R

This R script was developed by Barbara McGillivray in April 2019. It visualizes the patterns of association displayed in the evaluation summary contained in Summary_evaluation.xlsx.

Use:

1. Change definition of directory (variable "path") on line 13.
2. Run script.
3. Use plots in subdirectory "plots".

## proximity_1.3_centroid.py

This Python 3 script was developed by Martina Astrid Rodda in May 2019. It takes as input a series of parameter values, a plaintext file with a list of target words, as well as .dm file produced by DISSECT (containing a semantic space), and computes the cosine distance between the centroid of all target words and each individual target word.

Use:

1. Set directory and file names for input and output files in lines 31-49.
2. Enter values of parameters:

  a. "What is the window size?". Enter 1, 5, or 10, depending on the size of context window for co-occurrence matrix.
  
  b. "What is the frequency threshold for vocabulary lemmas?" Enter 1, 20, 50, or 100, depending on the frequency threshold for lemmas in vocabulary used to create the semantic space.
  
  c. "What is the target file name? (format: target_$, enter $)" Enter the second half of the name for the list of target words.

## References

Rodda, M. A., Probert, P. and McGillivray, B. (forthcoming). "Vector space models of Ancient Greek word meaning, and a case study on Homer".
