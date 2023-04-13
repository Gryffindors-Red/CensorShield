import nltk
import time


''' 0:hate
    1:offensive
    2:neither'''
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string


# NLP tools
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

# train split and fit models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer

# model selection
from sklearn.metrics import confusion_matrix, accuracy_score

# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# word plot
from wordcloud import WordCloud


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


nltk.download('omw-1.4')
nltk.download('stopwords')

dataset = pd.read_csv(
    'C:/Users/nagip/Desktop/Group_of_projects/Social-Network/network/18_plus_content_decteor/labeled_data.csv')
dataset.head()
