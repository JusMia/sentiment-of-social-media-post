import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


def clean_tweets(tweets):
    # remove twitter Return handles (RT @xxx:)
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:")

    # remove twitter handles (@xxx)
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")

    # remove URL links (httpxxx)
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")

    # remove special characters, numbers, punctuations (except for #)
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")



    return tweets

def sentiment_analyzer_scores(text):
    score = analyzer.polarity_scores(text)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1

#file_name ="danerdt.csv"
#file_name ="dane_justintrudeau.csv"
#file_name ="dane_theresamay.csv"
#file_name ="dane_oprah.csv"
#file_name ="dane_joannakrupa.csv"
#file_name ="hm.csv"
#file_name ="Microsoft.csv"
#file_name ="ikea_usa.csv"
#file_name ="LeagueOfLegends.csv"
#file_name ="TerriIrwin.csv"
#file_name ="DrLindseyFitz.csv"
#file_name ="wendymoore99.csv"
file_name = "Castrofied.csv"
df = pd.read_csv(file_name, encoding='latin')
df.head
# Have a look at the top 5 results.
df['text'] = clean_tweets(df['text'])
print(df['text'])
scores = []
# Declare variables for scores
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
for i in range(df['text'].shape[0]):
    # print(analyser.polarity_scores(sentiments_pd['text'][i]))
    compound = analyzer.polarity_scores(df['text'][i])["compound"]
    pos = analyzer.polarity_scores(df['text'][i])["pos"]
    neu = analyzer.polarity_scores(df['text'][i])["neu"]
    neg = analyzer.polarity_scores(df['text'][i])["neg"]

    scores.append({"Compound": compound,
                   "Positive": pos,
                   "Negative": neg,
                   "Neutral": neu
                   })
sentiments_score = pd.DataFrame.from_dict(scores)
df = df.join(sentiments_score)
print(df)
score_table = df.pivot_table(index='text',  values="Compound", aggfunc = np.mean)
print("print score table")
print(score_table)
print("print df")
print(df)

pos_count = 0
neg_count = 0
neutral_count = 0
for i in range(df.shape[0]):
    lb = df['Compound']
    if lb[i] >= 0.05:
        pos_count = pos_count + 1
    elif (lb[i] > -0.05) and (lb[i] < 0.05):
        neg_count = neg_count + 1
    else:
        neutral_count = neutral_count + 1
#sentiment_analyzer_scores(df['text'])

results = ("scores : "
      "neutral ", neutral_count,
      "negative ", neg_count,
      "positive ",pos_count)


#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/danerdtVader.csv')
#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/dane_justintrudeauVader.csv')
#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/dane_theresamayVader.csv')
#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/dane_oprahVader.csv')
#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/dane_joannakrupavader.csv')
#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/hmvade.csv')
#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/MicrosoftVader.csv')
#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/ikea_usaVader.csv')
#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/LeagueOfLegendsVader.csv')
#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/TerriIrwinVader.csv')
#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/DrLindseyFitzVader.csv')
#df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/wendymoore99Vader.csv')
df.to_csv('C:/Users/justi/Desktop/Doktorat/Sentiment analysis czerwiec/CastrofiedVader.csv')
