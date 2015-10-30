
# coding: utf-8

# # Separating Flowers
# This notebook explores a classic Machine Learning Dataset: the Iris flower dataset
# 
# ## Tutorial goals
# 1. Explore the dataset
# 2. Build a simple predictive modeling
# 3. Iterate and improve your score
# 

# How to follow along:
# 
#     git clone https://github.com/dataweekends/pyladies_intro_to_data_science
# 
#     cd pyladies_intro_to_data_science
#     
#     ipython notebook

# We start by importing the necessary libraries:

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')


# ### 1) Explore the dataset

# #### Numerical exploration
# 
# - Load the csv file into memory using Pandas
# - Describe each attribute
#     - is it discrete?
#     - is it continuous?
#     - is it a number?
# - Identify the target
# - Check if any values are missing
# 

# Load the csv file into memory using Pandas

# In[ ]:

df = pd.read_csv('iris-2-classes.csv')


# What's the content of ```df``` ?

# In[ ]:

df.head(3)


# Describe each attribute (is it discrete? is it continuous? is it a number? is it text?)

# In[ ]:

df.info()


# Are the features continuous or discrete?

# In[ ]:

df.describe()


# #### Identify the target
# What are we trying to predict?

# ah, yes... the type of Iris flower!

# In[ ]:

df['iris_type'].value_counts()


# Check if any values are missing

# In[ ]:

df.info()


# #### Mental notes so far:
# 
# - Dataset contains 100 entries
# - 1 Target column (```iris_type```)
# - 4 Numerical Features
# - No missing values

# #### Visual exploration

# - plot the distribution of the Sepal Length feature
# - check the influence of Sepal Length on the target

# Plot the distribution of Sepal Length

# In[ ]:

df['sepal_length_cm'].plot(kind='hist', figsize=(10,6))
plt.title('Distribution of Sepal Length', size = '20')
plt.xlabel('Sepal Length (cm)', size = '20')
plt.ylabel('Number of flowers', size = '20')


# check the influence of Sepal Length

# In[ ]:

df[df['iris_type']=='virginica']['sepal_length_cm'].plot(kind='hist', bins = 10, range = (4,7),
                                                      figsize=(10,6), alpha = 0.3, color = 'b')
df[df['iris_type']=='versicolor']['sepal_length_cm'].plot(kind='hist', bins = 10, range = (4,7),
                                                          figsize=(10,6), alpha = 0.3, color = 'g')
plt.title('Distribution of Sepal Length', size = '20')
plt.xlabel('Sepal Length (cm)', size = '20')
plt.ylabel('Number of flowers', size = '20')
plt.legend(['Virginica', 'Versicolor'])
plt.show()


# Check the influence of two features of combined

# In[ ]:

plt.scatter(df[df['iris_type']== 'virginica']['petal_length_cm'].values,
            df[df['iris_type']== 'virginica']['sepal_length_cm'].values, label = 'Virginica', c = 'b')
plt.scatter(df[df['iris_type']== 'versicolor']['petal_length_cm'].values,
            df[df['iris_type']== 'versicolor']['sepal_length_cm'].values, label = 'Versicolor', c = 'r')
plt.legend(['virginica', 'versicolor'], loc = 2)
plt.title('Scatter plot', size = '20')
plt.xlabel('Petal Length (cm)', size = '20')
plt.ylabel('Sepal Length (cm)', size = '20')
plt.show()


# Ok, so, the flowers seem to have different characteristics
# 
# Let's build a simple model to test that

# Define a new target column called "target" that is 1 if iris_kind = 'virginica' and 0 otherwise

# In[ ]:

df['target'] = df['iris_type'].map({'virginica': 1, 'versicolor': 0})

print df[['iris_type', 'target']].head(2)
print
print df[['iris_type', 'target']].tail(2)


# Define simplest model as benchmark

# The simplest model is a model that predicts 0 for everybody, i.e. all versicolor.
# 
# How good is it?

# In[ ]:

actual_versicolor = len(df[df['target'] == 0])
total_flowers = len(df)
ratio_of_versicolor = actual_versicolor / float(total_flowers)

print "If I predict every flower is versicolor, I'm correct %0.1f %% of the time" % (100 * ratio_of_versicolor)

df['target'].value_counts()


# We need to do better than that

# Define features (X) and target (y) variables

# In[ ]:

X = df[['sepal_length_cm', 'sepal_width_cm',
        'petal_length_cm', 'petal_width_cm']]
y = df['target']


# Initialize a decision Decision Tree model

# In[ ]:

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=0)
model  


#  Split the features and the target into a Train and a Test subsets.
#  
#  Ratio should be 70/30

# In[ ]:

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size = 0.3, random_state=0)


# Train the model

# In[ ]:

model.fit(X_train, y_train)


# Calculate the model score

# In[ ]:

my_score = model.score(X_test, y_test)

print "Classification Score: %0.2f" % my_score


# Print the confusion matrix for the decision tree model

# In[ ]:

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)

print "\n=======confusion matrix=========="
print confusion_matrix(y_test, y_pred)


# ### 3) Iterate and improve
# 
# Now you have a basic pipeline. How can you improve the score? Try:
# - changing the parameters of the model
#   check the documentation here:
#   http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#   
# - changing the model itself
#   check examples here:
#   http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#   
# - try separating 3 classes of flowers using the ```iris.csv``` dataset provided
