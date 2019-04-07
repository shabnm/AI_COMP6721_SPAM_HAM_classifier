# COMP6721_project2
Project2- Spam/Ham filtering

to run system project root folder run:

    python3 ./test/classifier_test.py 

Results are in ```./data/out/``` folder

project root folder:
  
  src:

    * data_provider.py -- class responsible for listing test/train files 
    * naive_bayes_model.py  -- model implementation

  test:

    * classifier_test.py -- entry point for program

  data:

    * data_train -- train files
    * data_test -- test files
    * data_out -- model output files

  English_stop_word.txt -- stop words

Required dependencies:

    import os
    import re
    import math
    import unittest
