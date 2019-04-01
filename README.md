# COMP6721_project2
Project2- Spam/Ham filtering

use test/classifier_test.py to run the classifier

Folder src contains:

  classifier.py : will be the main class for the project execution, is in progress
  
  conditional_probability.py : takes the Ordered dictionary along with all the values and calculate the probability and send data to writer
  to write to external file
  
  file_path.py : contains constant file locations
  
  inverted_index.py : used to create the ordered dixtionary of the terms occured in the ham/spam files, and used to calculate the term
  frequency and also the vocabulary size of each file types.
        Right now, we are calculating vocab size as the count of new words in each file type.
       
  smooth.py : will be used to add details about smoothning, is in progress
  
  writer.py : used to write the calculated data to external file
  
Project2-Test Folder : contains the test file for the classifier
Project2-Train Folder : contains the training file of the classifier
model.txt : is the file that contains the data of the classifier in below format
    S.no  word  word_count_in_ham  conditional_probability_in_ham  word_count_in_spam  conditional_probability_in_spam 
    1 abc 3 0.003 40 0.4
    2 airplane 3 0.003 40 0.4
    3 password 40 0.4 50 0.03
    4 zucchini 0.7 0.003 0 0.000001
English-stop-wrod.txt : contains the stop word that will be removed while creating the classifier , when the user sets stop_word_flag as      true
