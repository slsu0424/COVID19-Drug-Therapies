# Databricks notebook source
# MAGIC %md #Part 3: Model Training and Tracking

# COMMAND ----------

!pip install xgboost
!pip install mlflow
!pip install azureml-core
!pip install azureml-mlflow
!pip install imbalanced-learn

# COMMAND ----------

# import libraries needed
import pandas as pd # for data analysis
import numpy as np # for numeric calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns #for data visualization
from statistics import mode
import scipy as sp

# Print Full Numpy Array
# https://www.delftstack.com/howto/numpy/python-numpy-print-full-array/
import sys
np.set_printoptions(threshold=sys.maxsize)

# library settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# COMMAND ----------

# MAGIC %md #Load data

# COMMAND ----------

#df = spark.read.csv("/mnt/adls/FAERS_CSteroid_preprocess3.csv", header="true", nullValue = "NA", inferSchema="true")
df = spark.read.csv("/mnt/adls/FAERS_CSteroid_preprocess3_5589.csv", header="true", nullValue = "NA", inferSchema="true")

display(df)

# COMMAND ----------

# how many rows, columns

print((df.count(), len(df.columns)))

# COMMAND ----------

# convert to pandas

df1 = df.toPandas()

# COMMAND ----------

# MAGIC %md #Curate feature sets

# COMMAND ----------

# generate updated list of numerics

# https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas
# df_converted2 = df_converted.select_dtypes(include=np.number).columns.tolist()

df1.select_dtypes(exclude='object').dtypes

# COMMAND ----------

# curate feature set

df2 = df1.select_dtypes(exclude='object') \
                            .drop(['primaryid','caseid','caseversion','event_dt','mfr_dt','init_fda_dt','fda_dt','age','wt', \
                                    'rept_dt','last_case_version','val_vbm','start_dt','end_dt','drug_seq','dsg_drug_seq'], axis=1)

#cols =  df_converted.columns[~df_converted.columns.str.endswith('dt')].tolist()

# COMMAND ----------

display(df2)

# COMMAND ----------

df2.shape

# COMMAND ----------

# MAGIC %md ### Split into (X, y)

# COMMAND ----------

# https://sparkbyexamples.com/pandas/pandas-select-all-columns-except-one-column-in-dataframe

# X = input
X = df2.drop("outc_cod_DE" ,axis= 1)

print(X)

# y = output
y = df2['outc_cod_DE']

# COMMAND ----------

## split the data into training and testing data

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
import seaborn as sns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# show size of each dataset (records, columns)
print("Dataset sizes: \nX_train", X_train.shape," \nX_test", X_test.shape, " \ny_train", y_train.shape, "\ny_test", y_test.shape)

data = {
    "train":{"X": X_train, "y": y_train},        
    "test":{"X": X_test, "y": y_test}
}

print ("Data contains", len(data['train']['X']), "training samples and",len(data['test']['X']), "test samples")

# COMMAND ----------

# columns

X_train.columns

# COMMAND ----------

# MAGIC %md #Select Metric

# COMMAND ----------

# MAGIC %md ###Are the classes balanced?

# COMMAND ----------

# check label ratios
# https://dataaspirant.com/handle-imbalanced-data-machine-learning/

# 0 is majority class
# 1 is minority class
sns.countplot(x='outc_cod_DE', data=df2) # data already looks wildly imbalanced but let us continue

# COMMAND ----------

# MAGIC %md # Build a baseline model

# COMMAND ----------

# MAGIC %md ## Baseline - Naive Classifier

# COMMAND ----------

# dummy classifier - establish baseline performance

# A dummy classifier is a type of classifier which does not generate any insight about the data and classifies the given data using only simple rules. The classifier’s behavior is completely independent of the training data as the trends in the training data are completely ignored and instead uses one of the strategies to predict the class label.  It is used only as a simple baseline for the other classifiers i.e. any other classifier is expected to perform better on the given dataset. It is especially useful for datasets where are sure of a class imbalance. It is based on the philosophy that any analytic approach for a classification problem should be better than a random guessing approach.

# https://www.geeksforgeeks.org/ml-dummy-classifiers-using-sklearn/
# https://machinelearningmastery.com/naive-classifiers-imbalanced-classification-metrics
# https://medium.com/@mamonu/what-is-the-scikit-learn-dummy-classifier-95549d9cd44
# http://subramgo.github.io/2017/01/02/AutoGen_BaseClassifier/
# https://docs.w3cub.com/scikit_learn/modules/generated/sklearn.dummy.dummyclassifier.html

# COMMAND ----------

strategies = ['most_frequent', 'stratified', 'uniform', 'constant']

test_scores = []
for s in strategies:
	if s =='constant':
		dclf = DummyClassifier(strategy = s, random_state = 0, constant = 1)
	else:
		dclf = DummyClassifier(strategy = s, random_state = 0)
	dclf.fit(X_train, y_train)
	score = dclf.score(X_test, y_test)
	test_scores.append(score)

# COMMAND ----------

print(test_scores)

# COMMAND ----------

ax = sns.stripplot(strategies, test_scores);
ax.set(xlabel ='Strategy', ylabel ='Test Score')
plt.show()

# constant strategy (predicting minority class) --> most closely approximates F-1 measure

# COMMAND ----------

# MAGIC %md ## Multiple Model Training

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

from xgboost import XGBClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, classification_report

# COMMAND ----------

# https://ataspinar.com/2017/05/26/classification-with-scikit-learn/
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
    #"QDA": QuadraticDiscriminantAnalysis(),
    "Gaussian Process": GaussianProcessClassifier() #http://www.ideal.ece.utexas.edu/seminar/GP-austin.pdf
}

# COMMAND ----------

def batch_classify(X_train, y_train, X_test, y_test, no_classifiers = 10, verbose = True):
    """
    This method, takes as input the X, Y matrices of the Train and Test set.
    And fits them on all of the Classifiers specified in the dict_classifier.
    The trained models, and accuracies are saved in a dictionary. The reason to use a dictionary
    is because it is very easy to save the whole dictionary with the pickle module.
    
    Usually, the SVM, Random Forest and Gradient Boosting Classifier take quiet some time to train. 
    So it is best to train them on a smaller dataset first and 
    decide whether you want to comment them out or not based on the test accuracy score.
    """

# https://datascience.stackexchange.com/questions/28426/train-accuracy-vs-test-accuracy-vs-confusion-matrix

    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.process_time()
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        t_end = time.process_time()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, y_train) # training accuracy
        test_score = classifier.score(X_test, y_test) # test accuracy
        precision = precision_score(y_test, predictions) # fraction of positive predictions that were correct. TP / (TP + FP)
        recall = recall_score(y_test, predictions) # fraction of positive predictions that were correctly identified.  TP / (TP + FN)
        f1 = f1_score(y_test, predictions) # avg of precision + recall ratios
        fbeta = fbeta_score(y_test, predictions, beta=0.5)
        class_report = classification_report(y_test, predictions)
        
        dict_models[classifier_name] = {'model': classifier, \
                                        'train_score': train_score, \
                                        'test_score': test_score, \
                                        'precision': precision_score, \
                                        'recall': recall_score, \
                                        'f1': f1, \
                                        'fbeta': fbeta, \
                                        'train_time': t_diff, 
                                        'class_report': class_report}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models



def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    precision_s = [dict_models[key]['precision'] for key in cls]
    recall_s = [dict_models[key]['recall'] for key in cls]
    f1_s = [dict_models[key]['f1'] for key in cls]
    fbeta_s = [dict_models[key]['fbeta'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    report = [dict_models[key]['class_report'] for key in cls]
    
    #df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),9)), columns = ['classifier', 'train_score', 'test_score', 'precision', 'recall', 'f1', 'fbeta', 'train_time', 'class_report'])
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),7)), columns = ['classifier', 'train_score', 'test_score', 'f1', 'fbeta', 'class_report', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'precision'] = precision_s[ii]
        df_.loc[ii, 'recall'] = recall_s[ii]
        df_.loc[ii, 'f1'] = f1_s[ii]
        df_.loc[ii, 'fbeta'] = fbeta_s[ii]
        df_.loc[ii, 'class_report'] = report[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    display(df_.sort_values(by=sort_by, ascending=False))

# COMMAND ----------

dict_models = batch_classify(X_train, y_train, X_test, y_test, no_classifiers = 10)

display_dict_models(dict_models)

# COMMAND ----------

# MAGIC %md # Evaluate top models

# COMMAND ----------

# MAGIC %md ### 1. Gaussian Process

# COMMAND ----------

model = tree.DecisionTreeClassifier()
#model = GaussianProcessClassifier()

# train the model
model.fit(X_train, y_train)

# make predictions based on the trained model
predictions = model.predict(X_test)

# COMMAND ----------

# test set Outcomes (DE)
print(y_test)

# COMMAND ----------

# predictions - did the algorithm guess the outcome correctly?
print(predictions)

# COMMAND ----------

# some labels in y_test don't appear in predictions

# https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi

set(y_test) - set(predictions) # Specifically in this case, label '1' is never predicted

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Confusion Matrix
# MAGIC 
# MAGIC The confusion matrix is an N x N table (where N is the number of classes) that contains the number of correct and incorrect predictions of the classification model.
# MAGIC 
# MAGIC The rows of the matrix represent the real classes, while the columns represent the predicted classes.
# MAGIC 
# MAGIC The matrix returns four possible prediction outcomes:  
# MAGIC   
# MAGIC   
# MAGIC * True Positives (TP): The model predicted positive, and the real value is positive.
# MAGIC * False Positives (FP): The model predicted positive, but the real value is negative.
# MAGIC * False Negatives (FN): The model predictive negative, but the real value is positive.
# MAGIC * True Negatives (TN): The model predicted negative, and the real value is negative. 
# MAGIC 
# MAGIC These metrics are generally tabulated for the test set and shown together as a confusion matrix, which takes the following form:
# MAGIC   
# MAGIC   
# MAGIC <table style="border: 1px solid black;">
# MAGIC     <tr style="border: 1px solid black;">
# MAGIC         <td style="border: 1px solid black;color: black;" bgcolor="lightgray">TN</td><td style="border: 1px solid black;color: black;" bgcolor="white">FP</td>
# MAGIC     </tr>
# MAGIC     <tr style="border: 1px solid black;">
# MAGIC         <td style="border: 1px solid black;color: black;" bgcolor="white">FN</td><td style="border: 1px solid black;color: black;" bgcolor="lightgray">TP</td>
# MAGIC     </tr>
# MAGIC </table>
# MAGIC 
# MAGIC Note that the correct (true) predictions form a diagonal line from top left to bottom right - these figures should be significantly higher than the false predictions if the model performs well.
# MAGIC 
# MAGIC https://medium.com/swlh/confusion-matrix-and-classification-report-88105288d48f

# COMMAND ----------

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, predictions), annot = True, fmt ='d', cmap='Blues')

plt.ylabel('Actual Death')
plt.xlabel('Predicted Death')
plt.title('Confusion Matrix')

# save as image
plt.savefig("confusion-matrix.png")

# COMMAND ----------

# MAGIC %md %md
# MAGIC #### Confusion Matrix: Summary
# MAGIC 
# MAGIC The numbers in the matrix tell us the following:
# MAGIC * True Positive (TP) = # of patients that were predicted to not die (as a result of being given corticosteroids), and actually did not die.  The algorithm predicted it correctly.
# MAGIC * False Positive (FP) = # of patients were predicted to not die, but actually did die.  
# MAGIC * False Negative (FN) = # of patients were predicted to die, but actually did not die.
# MAGIC * True Negative (TN) = # of patients were predicted to die, and actually did die.
# MAGIC 
# MAGIC The goal is to keep TN and TP high.
# MAGIC 
# MAGIC ##### _Which metric is more costly?_  
# MAGIC 
# MAGIC (FP) If the model predicted that >100 patients would not die (as a result of being given corticosteroids), but actually did die, this could be very costly to a life sciences company.  They would want to know the risk factors that caused a patient to die when considering new drug development.
# MAGIC 
# MAGIC (FN) If the model predicted that >100 patients would die but actually did not die, this could equally be important for a life sciences company when considering new drug development.
# MAGIC 
# MAGIC We can use the data of the confusion matrix to compute additional metrics to quantify the model's performance.
# MAGIC 
# MAGIC https://towardsdatascience.com/understanding-confusion-matrix-precision-recall-and-f1-score-8061c9270011

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Classification Report  
# MAGIC 
# MAGIC For a quick summary of model performance, we can use sklearn to generate a classification report.  This table helps answer the following questions:
# MAGIC  * Precision: Of the predictons the model made for this class, what proportion were correct?
# MAGIC  * Recall: Out of all of the instances of this class in the test dataset, how many did the model identify?
# MAGIC  * F1-Score: An average metric that takes both precision and recall into account
# MAGIC  * Support: How many instances of this class are there in the test dataset?
# MAGIC 
# MAGIC The classification report also includes averages for these metrics, including a weighted average that allows for the imbalance in the number of cases of each class.

# COMMAND ----------

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

# COMMAND ----------

# MAGIC %md #### Accuracy, Precision, Recall, and F1-score

# COMMAND ----------

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average=None) # returns precision for each class; default is pos class (1)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", accuracy) 
print("Precision:", precision)
print("Recall:", recall)
print("F1_score:", f1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Precision-Recall Curves

# COMMAND ----------

# example of a precision-recall curve for a predictive model

# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
# https://www.codespeedy.com/predict_proba-for-classification-problem-in-python/

#from sklearn.metrics import precision_recall_curve
#from matplotlib import pyplot

# predict probabilities of the occurrence of each target
yhat = model.predict_proba(X_test)

# print probabilities
print(yhat)

# COMMAND ----------

# retrieve just the probabilities for the positive class (outcome of Death = 1)
pos_probs = yhat[:, 1]

pos_probs

# COMMAND ----------

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot

# calculate the no skill line as the proportion of the positive class
no_skill = len(y[y==1]) / len(y)

# plot the no skill precision-recall curve
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, pos_probs)

# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='Classifier')

# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()

# COMMAND ----------

# MAGIC %md ### 2. Decision Tree

# COMMAND ----------

model = tree.DecisionTreeClassifier()

# train the model
model.fit(X_train, y_train)

# make predictions based on the trained model
predictions = model.predict(X_test)

# COMMAND ----------

# test set outcomes
y_test

# COMMAND ----------

# test set outcomes
predictions

# COMMAND ----------

# some labels in y_test don't appear in predictions

# https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi

set(y_test) - set(predictions) # Specifically in this case, label '1' is never predicted

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Confusion Matrix

# COMMAND ----------

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, predictions), annot = True, fmt ='d', cmap='Blues')

plt.ylabel('Actual Death')
plt.xlabel('Predicted Death')
plt.title('Confusion Matrix')

# save as image
plt.savefig("confusion-matrix.png")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Classification Report

# COMMAND ----------

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

# COMMAND ----------

# MAGIC %md #### Accuracy, Precision, Recall, and F1-score

# COMMAND ----------

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average=None) # returns precision for each class; default is pos class (1)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", accuracy) 
print("Precision:", precision)
print("Recall:", recall)
print("F1_score:", f1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Precision-Recall Curves

# COMMAND ----------

# example of a precision-recall curve for a predictive model

# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
# https://www.codespeedy.com/predict_proba-for-classification-problem-in-python/

#from sklearn.metrics import precision_recall_curve
#from matplotlib import pyplot

# predict probabilities of the occurrence of each target
yhat = model.predict_proba(X_test)

# print probabilities
print(yhat)

# COMMAND ----------

# retrieve just the probabilities for the positive class (outcome of Death = 1)
pos_probs = yhat[:, 1]

pos_probs

# COMMAND ----------

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot

# calculate the no skill line as the proportion of the positive class
no_skill = len(y[y==1]) / len(y)

# plot the no skill precision-recall curve
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, pos_probs)

# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='Classifier')

# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()

# COMMAND ----------

# MAGIC %md # Improve metric

# COMMAND ----------

# https://www.mvorganizing.org/what-is-a-bad-f1-score/
# https://datascience.stackexchange.com/questions/65094/whats-a-good-f1-score-in-not-extremely-imbalanced-dataset
  
#StandardScaler()
#GridSearchCV for Hyperparameter Tuning.
#Recursive Feature Elimination(for feature selection)
#SMOTE(the dataset is imbalanced so I used SMOTE to create new examples from existing examples)

# COMMAND ----------

# MAGIC %md ## SMOTE

# COMMAND ----------

# Data balancing applied using SMOTE

from imblearn.over_sampling import SMOTE
from collections import Counter

print('Original dataset shape {}'.format(Counter(y)))

sm = SMOTE(random_state=20)

X_resample, y_resample = sm.fit_resample(X, y)

print('New dataset shape {}'.format(Counter(y_resample)))

# COMMAND ----------

# MAGIC %md # Re-train Models

# COMMAND ----------

dict_models = batch_classify(X_resample, y_resample, X_test, y_test, no_classifiers = 10)

display_dict_models(dict_models)

# COMMAND ----------

# MAGIC %md # Evaluate Top Models

# COMMAND ----------

# MAGIC %md ### 1. Gaussian Process

# COMMAND ----------

# test logistic regression algorithm

# https://www.aboutdatablog.com/post/a-quick-overview-of-5-scikit-learn-classification-algorithms

#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier

# instantiate a new logistic regression object
#model = LogisticRegression(fit_intercept=True, penalty='l1', solver='liblinear')
model = tree.DecisionTreeClassifier()
#model = XGBClassifier()
#model = GaussianProcessClassifier()

# train the model
#model.fit(X_train, y_train)
model.fit(X_resample, y_resample)

# make predictions based on the trained model
predictions = model.predict(X_test)

# COMMAND ----------

# test set Outcomes (DE)
print(y_test)

# COMMAND ----------

# predictions - did the algorithm guess the outcome correctly?
print(predictions)

# COMMAND ----------

# some labels in y_test don't appear in predictions

# https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi

set(y_test) - set(predictions) # Specifically in this case, label '1' is never predicted

# COMMAND ----------

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, predictions), annot = True, fmt ='d', cmap='Blues')

plt.ylabel('Actual Death')
plt.xlabel('Predicted Death')
plt.title('Confusion Matrix')

# save as image
plt.savefig("confusion-matrix.png")

# COMMAND ----------

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Accuracy, Precision, Recall, and F1-score

# COMMAND ----------

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average=None) # returns precision for each class; default is pos class (1)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", accuracy) 
print("Precision:", precision)
print("Recall:", recall)
print("F1_score:", f1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Precision-Recall Curves

# COMMAND ----------

# example of a precision-recall curve for a predictive model

# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
# https://www.codespeedy.com/predict_proba-for-classification-problem-in-python/

#from sklearn.metrics import precision_recall_curve
#from matplotlib import pyplot

# predict probabilities of the occurrence of each target
yhat = model.predict_proba(X_test)

# print probabilities
print(yhat)

# COMMAND ----------

# retrieve just the probabilities for the positive class (outcome of Death = 1)
pos_probs = yhat[:, 1]

pos_probs

# COMMAND ----------

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot

# calculate the no skill line as the proportion of the positive class
no_skill = len(y[y==1]) / len(y)

# plot the no skill precision-recall curve
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, pos_probs)

# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='Classifier')

# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()

# COMMAND ----------

# Create list of top most features based on importance
feature_names = X_resample.columns
feature_imports = model.feature_importances_
most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(20, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)
print(most_imp_features)
plt.figure(figsize=(10,6))
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Most important features - Decision Tree (entropy function, complex model)')
plt.show()

# COMMAND ----------

model = RandomForestClassifier(n_estimators=1000)

# train the model
model.fit(X_train, y_train)
  
# make predictions based on the trained model
predictions = model.predict(X_test)

# COMMAND ----------

# test set outcomes
y_test

# COMMAND ----------

# predictions - did the algorithm guess the outcome correctly?
predictions

# COMMAND ----------

# some labels in y_test don't appear in predictions

# https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi

set(y_test) - set(predictions) # Specifically in this case, label '1' is never predicted

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Confusion Matrix

# COMMAND ----------

from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, predictions), annot = True, fmt ='d', cmap='Blues')

plt.ylabel('Actual Death')
plt.xlabel('Predicted Death')
plt.title('Confusion Matrix')

# save as image
plt.savefig("confusion-matrix.png")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Classification Report

# COMMAND ----------

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Accuracy, Precision, Recall, and F1-score

# COMMAND ----------

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", accuracy) 
print("Precision:", precision)
print("Recall:", recall)
print("F1_score:", f1)

# COMMAND ----------

# example of a precision-recall curve for a predictive model

# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
# https://www.codespeedy.com/predict_proba-for-classification-problem-in-python/

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot

# predict probabilities of the occurrence of each target
yhat = model.predict_proba(X_test)

# print probabilities
print(yhat)

# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]

# calculate the no skill line as the proportion of the positive class
no_skill = len(y[y==1]) / len(y)

# plot the no skill precision-recall curve
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, pos_probs)

# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='Classifier')

# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()

# COMMAND ----------

# Create list of top most features based on importance
feature_names = X_train.columns
feature_imports = model.feature_importances_
most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(20, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)
print(most_imp_features)
plt.figure(figsize=(10,6))
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Most important features - Decision Tree (entropy function, complex model)')
plt.show()

# COMMAND ----------

# MAGIC %md # Create or load an Azure ML Workspace  
# MAGIC 
# MAGIC Before models can be deployed to Azure ML, you must create or obtain an Azure ML Workspace. The `azureml.core.Workspace.create()` function will load a workspace of a specified name or create one if it does not already exist. 
# MAGIC 
# MAGIC For more information about creating an Azure ML Workspace, see the [Azure ML Workspace management documentation](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace).

# COMMAND ----------

# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-azure-databricks

import mlflow
import mlflow.azureml
import azureml.mlflow
import azureml.core

from azureml.core import Workspace

subscription_id = '9edd9e25-815c-4cdb-9bc8-d2ba127ec752'

# Azure Machine Learning resource group NOT the managed resource group
resource_group = '2021-06-22_SDUD' 

#Azure Machine Learning workspace name, NOT Azure Databricks workspace
workspace_name = 'SDUDaml'  

# Instantiate Azure Machine Learning workspace
ws = Workspace.get(name=workspace_name,
                   subscription_id=subscription_id,
                   resource_group=resource_group)

#Set MLflow experiment. 
#experimentName = "/Users/sansu@microsoft.com/Drug_Utilization/Notebook 3_FAERS" 
experimentName = "test2" 
mlflow.set_experiment(experimentName)

# COMMAND ----------

uri = ws.get_mlflow_tracking_uri()

print(uri)

mlflow.set_tracking_uri(uri)

# COMMAND ----------

# MAGIC %md ### MLFlow: Log Metrics and Artifacts in Azure Machine Learning Workspace

# COMMAND ----------

# start a new MLflow training run

with mlflow.start_run():
#import mlflow

  # log metrics
  mlflow.log_metric("Accuracy", accuracy)
  mlflow.log_metric("Precision", precision)
  mlflow.log_metric("Recall", recall)
  mlflow.log_metric("F1-score", f1)
  #mlflow.log_metric("ROC AUC", auc)
  
  # log artifacts
  mlflow.log_artifact("confusion-matrix.png")
  #mlflow.log_artifact("roc-curve.png")
  
  # save the model to the outputs directory
  mlflow.sklearn.log_model(model, "model") # https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model
  
  # run id is logged as Azure ML Experiments Run ID
  print("Runid is:", mlflow.active_run().info.run_uuid)
  
  # end run
  mlflow.end_run()

# COMMAND ----------

# start a new MLflow training run

#with mlflow.start_run():
import mlflow

  # log metrics
mlflow.log_metric("Accuracy", accuracy)
mlflow.log_metric("Precision", precision)
mlflow.log_metric("Recall", recall)
mlflow.log_metric("F1-score", f1)
  #mlflow.log_metric("ROC AUC", auc)
  
  # log artifacts
mlflow.log_artifact("confusion-matrix.png")
  #mlflow.log_artifact("roc-curve.png")
  
  # save the model to the outputs directory
mlflow.sklearn.log_model(model, "model") # https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model
  
  # run id is logged as Azure ML Experiments Run ID
print("Runid is:", mlflow.active_run().info.run_uuid)
  
  # end run
  #mlflow.end_run()

# COMMAND ----------

# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

# Generate and plot a synthetic imbalanced classification dataset

from collections import Counter # https://www.guru99.com/python-counter-collections-example.html
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
