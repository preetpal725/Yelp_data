# -*- coding: utf-8 -*-
"""
THE PROBLEM:
The purpose of the project is to create a python program that could rate a comment or review of a restaurant
in the number of stars. It could be between 1-5 depending upon the performance of the restaurant. In this 
case the program will be able to classify the comment given by a customer on the 'yelp' using a SVM classifier. 
The user will be able to see the accuracy of the program and the confusion matrix depicting the true positives 
and true negatives. The program used different natural language processing tools such as n-grams. The comments 
are fist preprocessed using data slicing and TfidfVectorizer that removes the words that occur most frequently
in the text data. The data is then balanced according to the least number of comments for a star rating. e.g. if
there are 500 reviews for 1 star rating and more than that for other star ratings, then the program will only take
500 reviews of all the star ratings in order to balance the output. Then the 70% of the data, called test data, is
ran on SVM classifier and the model is trained. The model is tested on the remaining 30% of the data and the results 
are shown as output.  

YELP SCRIPT:
[Enter] python balanced_yelp_project.py yelp.csv
[Output] Accuracy and confusion matrix

USAGE:
The program is called 'balanced_yelp_project.py', and it should be run from the command line or linux.

ALGORITHM:
1. The program starts with prompting the user to enter the inputs: 
    "python balanced_yelp_project.py(the python file) yelp.csv(the csv data file) ngram"
    For e.g. "python balanced_yelp_project.py yelp.csv 2"
2. The program will then read the files and import the dataset.
3. It will seperate the columns of the .cs file using ',' as delimiter.
4. It will store the list of text comments in texts list and star rating in stars list
5. It will store the ngrams given by the user.
5. It will call the method that will balance the data according to the least number of reviews for 
    any star rating.
6. It will break the words into single tokens and bi-grams then then it will vectorize them according to 
    the number of times it is repeating.
7. It will build up the vocabulary from all the reviews and turns each indivdual text into a matrix of numbers
8. Now, the SVM classifier model is created and trained on 70% of the data.
9. The classifier model is then checked and run on remaining test data.
10. CASE 1: Then the program breaks the data in 2 parts, i.e. 'positive' and 'negative'. The star ratings 1 & 2 are 
    considered as negative while 4 & 5 are considered as 'positive' while 3 is taken as 'neutral'.
11. CASE 2: Then the program breaks the data in 2 parts, i.e. 'positive' and 'negative'. The star ratings 1 & 2 are 
    considered as negative while 3, 4 & 5 are considered as 'positive'.
12. CASE 3: Then the program breaks the data in 2 parts, i.e. 'positive' and 'negative'. The star ratings 1, 2 & 3 
    are considered as negative while 4 & 5 are considered as 'positive'.
13. Case 4. Then the program breaks the data in 2 parts, i.e. 'positive' and 'negative'. The star ratings 1 & 2
    are considered as negative while 4 & 5 are considered as 'positive'.    
    
COURSE: AIT 690 - Natural Language Proccessing
AUTHOR’s NAME: Preetpal Singh and Ahmed Alshaibani
DATE: 5 December 2018
"""

'''Importing the libraries '''

import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from collections import Counter
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import sys
import csv
import pandas as pd


''' Read the yelp data for text and star ratings '''
texts = []
stars = []
argList = sys.argv
with open(argList[1], 'r') as my_file:
    csv_reader = csv.reader(my_file, delimiter = ',')
    line_count = 0
    for row in csv_reader:
        if line_count > 0:
            texts.append(row[4])
            stars.append(int(row[3]))
        line_count += 1
#text = pd.read_csv('yelp_file', usecols=['text'])
#star = pd.read_csv('yelp.csv', usecols=['stars'])
#print(texts)
#print(stars)
        
        
''' Read the n-grams from the user '''
n = int(argList[2])
#n = 2

#''' Making list of texts and ratings fromt the data above '''
#texts = []
#stars = []
#for i in range(len(text)):
#    texts.append(text['text'][i])
#for i in range(len(star)):
#    stars.append(star['stars'][i])
#print(text+" "+ star)
#print(texts)
#print(stars)
    
    

#print("Results with balanced data !!!")
#print("N-Gram is set as:"+str(n))


''' Balancing the data according to the number of reviwes available for each rating '''
def balance_classes(xs, ys):
    ''' Undersample xs, ys to balance classes '''
    freqs = Counter(ys)
    ''' The least common rating is the maximum number we want for all the other ratings '''
    max_allowable = freqs.most_common()[-1][1]
    num_added = {clss: 0 for clss in freqs.keys()}
    new_ys = []
    new_xs = []
    for i, y in enumerate(ys):
        if num_added[y] < max_allowable:
            new_ys.append(y)
            new_xs.append(xs[i])
            num_added[y] += 1
    return new_xs, new_ys


''' This shows the number of stars for each star rating '''
print('\n\nThis shows the results for the Balanced data..')
print('\n\nInital number of reviews of each star rating:') 
print(Counter(stars))


''' This will balance the data-- the least number was 749 '''
balanced_x, balanced_y = balance_classes(texts, stars)
print('\n\nThe number of reviews of each star rating after balancing:') 
print(Counter(balanced_y))
#print(balanced_x)
#print(balanced_y)




''' This vectorizer breaks text into single words and bi-grams and then calculates the TF-IDF representation '''
vectorizer = TfidfVectorizer(ngram_range=(1,n))

 
''' The 'fit' builds up the vocabulary from all the reviews 
while the 'transform' step turns each indivdual text into a matrix of numbers '''
vectors = vectorizer.fit_transform(balanced_x)


X_train, X_test, y_train, y_test = train_test_split(vectors, balanced_y, test_size=0.3, random_state=42)

 
''' Initialise and train the SVM classifier '''
classifier = LinearSVC()
classifier.fit(X_train, y_train)



''' Test the classifier and check the accuracy '''
preds = classifier.predict(X_test)

print('\n\nAccuracy:')
print(accuracy_score(y_test, preds))
#print(classification_report(y_test, preds))
print('\n\nConfusion Matrix for star ratings')
print(confusion_matrix(y_test, preds))

print("\n\n----------------------------------------------------------------------------------------------------------")

keep = set([1,2,3,4,5])

print("\n\nCASE 1: The below results consider 1 & 2 stars as 'negative', 4 & 5 star as 'positive' and 3 star as 'neutral'")

''' to keep the examples we want to keep within the keep set '''
keep_train_is = [i for i, y in enumerate(y_train) if y in keep]
keep_test_is = [i for i, y in enumerate(y_test) if y in keep]

''' convert the stars in the train set to stars 1 and 2 as "negative" and 4 and 5 as "positive" '''
X_train2 = X_train[keep_train_is, :]
y_train2 = [y_train[i] for i in keep_train_is]
y_train2 = ["n" if (y == 1 or y == 2) else ("p" if  (y == 4 or y == 5) else "non" )for y in y_train2] 


''' convert the test set to stars 1 and 2 as "n" and the rest which is 4 and 5 as "p" '''
X_test2 = X_test[keep_test_is, :]
y_test2 = [y_test[i] for i in keep_test_is]
y_test2 = ["n" if (y == 1 or y == 2) else ("p" if  (y == 4 or y == 5) else "non" )for y in y_test2]

classifier.fit(X_train2, y_train2)
preds = classifier.predict(X_test2)

print('\nConfusion matrix')
print(confusion_matrix(y_test2, preds))
print('\nFinal accuracy')
print( accuracy_score(y_test2, preds))


print("\n\nCASE 2: The below results consider 1, 2 stars as 'negative' and 3, 4 & 5 star as 'positive'")
y_train2 = [y_train[i] for i in keep_train_is]
y_train2 = ["n" if (y == 1 or y == 2) else "p" for y in y_train2]


''' convert the test set to stars 1 and 2 as "n" and the rest which is 4 and 5 as "p" '''
X_test2 = X_test[keep_test_is, :]
y_test2 = [y_test[i] for i in keep_test_is]
y_test2 = ["n" if (y == 1 or y == 2) else "p" for y in y_test2]

classifier.fit(X_train2, y_train2)
preds = classifier.predict(X_test2)


print('\nConfusion matrix')
print(confusion_matrix(y_test2, preds))
print('\nFinal accuracy')
print( accuracy_score(y_test2, preds))


print("\n\nCASE 3: The below results consider 1, 2 & 3 stars as 'negative' and 4 & 5 star as 'positive'")
y_train2 = [y_train[i] for i in keep_train_is]
y_train2 = ["n" if (y == 1 or y == 2 or y == 3) else "p" for y in y_train2]


''' convert the test set to stars 1 and 2 as "n" and the rest which is 4 and 5 as "p" '''
X_test2 = X_test[keep_test_is, :]
y_test2 = [y_test[i] for i in keep_test_is]
y_test2 = ["n" if (y == 1 or y == 2) else "p" for y in y_test2]

classifier.fit(X_train2, y_train2)
preds = classifier.predict(X_test2)


print('\nConfusion matrix')
print(confusion_matrix(y_test2, preds))
print('\nFinal accuracy')
print( accuracy_score(y_test2, preds))


print("\n\nCASE 4: The below results consider 1 & 2 stars as 'negative' and 4 & 5 star as 'positive'")
''' convert the stars in the train set to stars 1 and 2 as "n" and the rest which is 4 and 5 as "p" '''
keep = set([1,2,4,5])
keep_train_is = [i for i, y in enumerate(y_train) if y in keep]
keep_test_is = [i for i, y in enumerate(y_test) if y in keep]
y_train2 = [y_train[i] for i in keep_train_is]
y_train2 = ["n" if (y == 1 or y == 2) else "p" for y in y_train2]
X_train2 = X_train[keep_train_is, :]


''' convert the test set to stars 1 and 2 as "n" and the rest which is 4 and 5 as "p" '''
X_test2 = X_test[keep_test_is, :]
y_test2 = [y_test[i] for i in keep_test_is]
y_test2 = ["n" if (y == 1 or y == 2) else "p" for y in y_test2]

classifier.fit(X_train2, y_train2)
preds = classifier.predict(X_test2)


print('\nConfusion matrix')
print(confusion_matrix(y_test2, preds))
print('\nFinal accuracy')
print( accuracy_score(y_test2, preds))
   
    
    
    
    
    
    
    
    