import streamlit as st
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
import nltk
from statistics import mode 
import random
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import preprocessing

nltk.download('stopwords')
from nltk.corpus import stopwords


# Title, instructions, and getting input
st.title('Which candidate would have said this?')

st.write('The purpose of this app is to take text and predict which top DNC candidate (Joe Biden, Bernie Sanders, Amy Klobuchar, Tom Steyer, Elizabeth Warren, Pete Buttigieg) was most likely to have said it during a DNC presidential debate in 2019-2020.')

usr_speech = st.text_area('Enter text below to find out which candidate gave responses most similar to your entry.')


# Initial DF manipulation
data = pd.read_csv('debate_transcripts_v3_2020-02-26.csv') # includes nevada and south carolina

top_candidates = ['Joe Biden', 'Bernie Sanders', 'Amy Klobuchar', 'Tom Steyer', 'Elizabeth Warren', 'Pete Buttigieg', 'Michael Bloomberg']

data = data.loc[data['speaker'].isin(top_candidates)]


# Calculating frequent words and word counts for candidates in top_candidates
freqWords = {}
wordCounts = {}

for candidate in top_candidates:
    candidate_data = data.loc[data['speaker'] == candidate]
    freqWords[candidate] = {}
    words = 0
    for line in candidate_data['speech']:
        for word in line.split():
            words += 1
            word = word.lower().replace(",", "").replace(".", "").replace("?", "").replace("-", "")
            if word in freqWords[candidate]:
                freqWords[candidate][word] += 1
            else:
                freqWords[candidate][word] = 1
    wordCounts[candidate] = words
                
    freqWords[candidate] = sorted(freqWords[candidate].items(), key = lambda kv:(kv[1], kv[0]))
    freqWords[candidate].reverse()

# Calculating word proportions for each candidate
wordProportions = {}

for candidate in top_candidates:
    wordProportions[candidate] = {}
    for word in freqWords[candidate]:
        wordProportions[candidate][word[0]] = word[1]/wordCounts[candidate]

# data['speaker'].value_counts() / len(data)


# Assembling list of all words and corresponding candidate
tmpList = []
tmpList.append([])
tmpList.append([])

for candidate in top_candidates:
    candidate_data = data.loc[data['speaker'] == candidate]
    for line in candidate_data['speech']:
        for word in line.split():
            word = word.lower().replace(",", "").replace(".", "").replace("?", "").replace("-", "").replace("\\", "").replace("!", "")
#             row = pd.DataFrame([[word, candidate]], columns=['word','candidate'])
#             allWords.append(row, ignore_index=True)
            tmpList[0].append(word)
            tmpList[1].append(candidate)

allWords = pd.DataFrame(np.array(tmpList).T,columns=['word','candidate'])


# Removing/adding words that increase accuracy
common_words = stopwords.words('english')

common_words.remove('i')
common_words.remove('whom')
common_words.remove('but')
common_words.remove('while')
common_words.remove('himself')
common_words.remove('his')
common_words.remove('which')

common_words.append('change')
common_words.append('medicare')
common_words.append('can')
common_words.append('would')
common_words.append('say')
common_words.append('going')
common_words.append("we're")

stripped_common_words = []
for word in common_words:
    word = word.replace("'", "")
    stripped_common_words.append(word)
# stripped_common_words


# Train model

tmp = []

for i in range(len(allWords)):
    tmp.append(({'word':allWords['word'][i]},allWords['candidate'][i]))

model_0 = nltk.NaiveBayesClassifier.train(tmp)


# Predict for user data

train_df = data
user_input_dict = {'Speech':[usr_speech], 'Candidate':['user']}
test_df = pd.DataFrame(data=user_input_dict)

other_candidates = test_df['Candidate'].unique()
candidates = ['Joe Biden', 'Bernie Sanders', 'Amy Klobuchar', 'Tom Steyer', 'Elizabeth Warren', 'Pete Buttigieg']

# Get list of words to train on
trList = []
trList.append([])
trList.append([])

teList = []
teList.append([])
teList.append([])

for candidate in candidates:
    candidate_train_data = train_df.loc[data['speaker'] == candidate]
    for line in candidate_train_data['speech']:
        for word in line.split():
            word = word.lower().replace(",", "").replace(".", "").replace("?", "").replace("-", "").replace("\\", "").replace("!", "")
            trList[0].append(word)
            trList[1].append(candidate)

for line in test_df['Speech']:
    for word in line.split():
        word = word.lower().replace(",", "").replace(".", "").replace("?", "").replace("-", "").replace("\\", "").replace("!", "")
        teList[0].append(word)
        teList[1].append(candidate)

trainWords = pd.DataFrame(np.array(trList).T,columns=['word','candidate'])
testWords = pd.DataFrame(np.array(teList).T,columns=['word','candidate'])

train_reformatted = []
test_reformatted = []

# Format into tuple for use in model

for i in range(len(trainWords)):
    train_reformatted.append(({'word':trainWords['word'][i]},trainWords['candidate'][i]))

for i in range(len(testWords)):
    test_reformatted.append(({'word':testWords['word'][i]},testWords['candidate'][i]))

model = nltk.NaiveBayesClassifier.train(train_reformatted)

test_speeches = []

candidate_data = test_df.loc[test_df['Candidate'] == 'user']
for line in candidate_data['Speech']:
    words = []
    for word in line.split():
        word = word.lower().replace(",", "").replace(".", "").replace("?", "").replace("-", "").replace("\"", "")
        words.append(({'word':word}, candidate))
    test_speeches.append(words)

print('test_df: ', test_df)

max_length = 0
tmp_length = 0

# for speech in test_speeches:
speech = test_speeches[0]
word_predictions = []
full_predictions = []
tmp_length = 0
for word in speech:
    if word[0]['word'] not in stripped_common_words and len(word[0]['word']) > 0:
        word_predictions.append(model_0.classify(word[0]))
        full_predictions.append(model_0.classify(word[0]))
    else:
        full_predictions.append("COMMON")
# print(word_predictions)
# print(speech)
# print("Predicted: ", speech_prediction, " Actual: ", speech[0][1])

if len(word_predictions) > 0:
    speech_prediction = mode(word_predictions)

    st.write('# **', speech_prediction, '**')
    st.write('The candidate whose responses most resemble the text above is **', speech_prediction, '**')

    str_result = ''

    split_input = usr_speech.split()

    for i in range(len(full_predictions)):
        if full_predictions[i] == speech_prediction:
            str_result += '**'
            str_result += split_input[i]
            str_result += '**'
        elif full_predictions[i] == 'COMMON':
            str_result += '~~'
            str_result += split_input[i]
            str_result += '~~'
        else:
            str_result += split_input[i]
        
        if (i != len(full_predictions) - 1):
            str_result += ' '
        
    st.write(str_result)

    st.write('### Guide')

    st.write('**Bold** words are words that were most likely to have been said by **', speech_prediction, '**. ~~Strikethrough~~ indicates that the word was not assigned a candidate because it was found on a list of common words. For Biden, some **bold** words may have never been said by him and are only predicted to be him because he had the highest word count overall.' )

st.write('### About this app')

st.write('This classifier was trained on all of the words spoken in each of the 2019-2020 DNC debates and predicts the speaker by parsing the response into individual words, removing common words, and making predictions on the remaining words for the candidate most likely to have said each word. The response was then classified as the candidate that with the highest tally. The Naive Bayes classifier allowes training and predicting each of the words and taking into account the proportion of words each candidate spoke (for example, Steyer was predicted less because he had fewer words overall and Biden was predicted most often because he had the most words spoken).')

st.write('Additionally, words that did not appear in the dataset are always predicted to be Biden, simply because he had the largest word count. This leads to a bias towards Biden in situations such as this where he does not necessarily have the largest proportion of responses. However, the accuracy when predicting for 2019-2020 DNC debate responses was nearly 90%.')

st.write('Chase Mortensen')
