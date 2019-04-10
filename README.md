# Ancient Greek Semantic Spaces

Code relative to the project on Ancient Greek semantic space. 
Project team (in alphabetical order): Barbara McGillivray (Turing/Cambridge), Philomen Probert (Oxford), Martina Astrid Rodda (Oxford/Turing), enrichment student at Turing from January to June 2019.

* evaluation_text_processing.py. This script was developed by Barbara McGillivray in March-April 2019. It takes as input a series of parameter values, as well as files relative to various semantic spaces created by the DISSECT tool on the Diorisis Ancient Greek corpus and files relative to three Ancient Greek lexicons, and returns a series of files containing the results of three evaluation approaches.

The script can be run on a local computer, or on a linux virtual machine. It uses Python 3.

Use:

1. Change definition of main directory on lines 77-81.
2. Run script from command line: python evaluation_text_processing.py
3. Enter values of parameters:
  a. "What is the window size?". Enter 1, 5, or 10, depending on the size of context window for co-occurrence matrix.
  b. "What is the frequency threshold for vocabulary lemmas?" Enter 1, 20, 50, or 100, depending on the frequency threshold for lemmas in vocabulary used to create the semantic space.
  c. "Is this a test?". Enter yes or no, depending on whether you want to run the script in test mode (i.e. only reading the first lines of the input files) or not.
  d. "Which lexicon do you want to consider?" Enter AGWN, SCHMIDT, or POLLUX, depending on which lexicon you're interested in comparing the semantic space with.
  e. "Do you want to follow the first evaluation approach?". Enter yes or no, depending on whether you want to run the first evaluation approach or not.
  f. "Do you want to follow the second evaluation approach?". Enter yes or no, depending on whether you want to run the second evaluation approach or not.
  g. "Do you want to follow the third evaluation approach?". Enter yes or no, depending on whether you want to run the third evaluation approach or not.

* evaluation_analysis.Rmd. This script was developed by Barbara McGillivray in April 2019.
