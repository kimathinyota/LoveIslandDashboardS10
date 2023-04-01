from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pandas as pd

# Finds sentiments for each of the inputted comments 
def calculate_sentiments(comment_df):
    sia = SIA()
    sentiments_df = comment_df.body.apply(sia.polarity_scores).apply(lambda dic: pd.Series(dic, index=list(dic.keys())))
    return sentiments_df

