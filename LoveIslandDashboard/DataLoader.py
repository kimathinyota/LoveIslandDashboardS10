import pandas as pd
import numpy as np
import streamlit as st
import LoveIslandAnalysis.Mentions as lm
from LoveIslandAnalysis.Islanders import fetch_and_clean_islanders_from_wikipedia
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


def load_comments():
    comment_df = pd.read_csv('LoveIslandDashboard/Data/Comment_inc_sentiments_2023-06-09_2023-06-28.csv')
    comment_df["date"] = pd.to_datetime(comment_df.createdDate, unit='s')
    comment_df['day'] = (comment_df["date"] - pd.to_datetime('2023-6-10')).dt.days + 1
    return comment_df

def load_posts():
    return pd.read_csv('LoveIslandDashboard/Data/Post_2023-06-10_2023-06-27.csv')

def load_islanders():
    return pd.read_csv('LoveIslandDashboard/Data/Islander_s10.csv')

def load_summaries():
    return pd.read_csv('LoveIslandDashboard/Data/EventSummaries_s10.csv')

def load_raw_data():
    comment_df, post_df, islanders, summaries = load_comments(), load_posts(), load_islanders(), load_summaries()
    return comment_df, post_df, islanders, summaries

def load_nicknames():
    nicknames = pd.read_csv("LoveIslandDashboard/Data/Islander_s10_nicknames.csv")
    nicknames_melt = nicknames.melt("Islander", value_name="Nickname")[["Islander", "Nickname"]]
    nicknames_dict = nicknames_melt.dropna().set_index("Nickname").Islander.to_dict()
    return nicknames_dict