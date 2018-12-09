README
Title: Review Classification of Yelp Data

Description: The purpose of the project is to create a python program that could rate a comment or review of a restaurant in the number of stars. 
It could be between 1-5 depending upon the performance of the restaurant. 
In this case the program will be able to classify the comment given by a customer on the 'yelp' using a SVM classifier. 
The user will be able to see the accuracy of the program and the confusion matrix depicting the true positives and true negatives. 
The program used different natural language processing tools such as n-grams. 
The comments are fist preprocessed using data slicing and data balancing and TfidfVectorizer that removes the words that occur most frequently in the text data. 
In this file, the data is not balanced, i.e. the number of reviews for each star ratings remain the same. 
Then the 80% of the data, called test data, is ran on SVM classifier and the model is trained. 
The model is tested on the remaining 20% of the data and the results are shown as output.  

Getting Started: You need to run this file on command line using the script:
[Enter] python balanced_yelp_project.py yelp.csv 2
Where, ' balanced_yelp_project.py' is the python file, 'yelp.csv' is the csv file and 2 can be the n-gram that user want to use.

Prerequisites: You can just run this file on command line using the script provided.

Installing: Open the command line.
Change the directory to the one where the python and csv file is located.
Run the script provided.
You might also need a python tool such as spyder to run the python code.

The output will show the results that included the confusion matrix and accuracy of the model in different cases:
CASE 1: Then the program breaks the data in 2 parts, i.e. 'positive' and 'negative'. 
The star ratings 1 & 2 are considered as negative while 4 & 5 are considered as 'positive' while 3 is taken as 'neutral'.

CASE 2: Then the program breaks the data in 2 parts, i.e. 'positive' and 'negative'. 
The star ratings 1 & 2 are considered as negative while 3, 4 & 5 are considered as 'positive'.

CASE 3: Then the program breaks the data in 2 parts, i.e. 'positive' and 'negative'. 
The star ratings 1, 2 & 3 are considered as negative while 4 & 5 are considered as 'positive'.

Case 4. Then the program breaks the data in 2 parts, i.e. 'positive' and 'negative'. 
The star ratings 1 & 2 are considered as negative while 4 & 5 are considered as 'positive'.

Example:
Confusion Matrix
[[ 128	180]
    14	1678]]
	
Accuracy
0.903

Authors

    Preetpal Singh - Initial work + Data preprocessing
    Ahmed - Classification model + testing
