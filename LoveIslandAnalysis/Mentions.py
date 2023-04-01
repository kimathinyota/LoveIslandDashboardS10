import pandas as pd
import numpy as np

import itertools

# For each text in the input text series, it will return the number of (first-name) matches for the input name series
# For search_name_series of [a: Tom is cool, b: Ronald and James are bad, c: James and Tom are good] 
# and search_name_series = [T: Tom, R: Ronald, J: James]
# find_names(..) will return ...
#  id | T | R | J
#   a | 1 | 0 | 0
#   b | 0 | 1 | 1
#   c | 1 | 0 | 1
def find_names(search_name_series, text_series):
    all_words = text_series.str.strip().str.split(expand=True).stack().reset_index(level=1, drop=True)
    df = all_words.to_frame('body')
    for index in search_name_series.index:
        df[index] = 0
        name = search_name_series.loc[index]
        df.loc[all_words == name, index] = 1
    mentions_over_index = df.reset_index(names="id").groupby(by='id').sum(numeric_only=True)
    return mentions_over_index
    
# Will transform an input series such that its data becomes the new index and exxtracts the first word (of each data entry) for the transformed data
def generate_search_name_series(islanders_df):
    name_series = pd.Series(
        data = islanders_df.Islander.str.split(' ').str.get(0).to_list(),
        index = islanders_df.Islander.to_list()
    )
    return name_series

# will find the quantity of first-name matches for each islander in each input comment  
def mentions_data(islanders_df, comment_df):
    name_series = generate_search_name_series(islanders_df)
    mentions_over_time = find_names(name_series, comment_df.body)
    mentions_over_time['day'] = comment_df['day']
    return name_series, mentions_over_time

# Transform input series into a list of its index ordered by its corresponding values
# [T: 3, R: 0, J: 1] ---> [T, J]
# [T: 3, R: 2, J: 1] ---> [T, R, J]
# [T: 3, R: 2, J: 1, A: 11] ---> [A, T, R, J]
def cooccur(series):
    s = series[:-2]
    if s.size > 0:
        return s.loc[s > 0].sort_values()[::-1].index.to_list()
    return pd.NA


# Columns of mentions_over_time take the form of: CommentID, Islander1, Islndaer2, Islander3, ....
# Will reduce mentions_over_time dataframe into a series of mentioned islanders list in descending order of mentions frequency
def cooccurrent_data(mentions_over_time):
    return mentions_over_time.apply(cooccur, axis=1)

# give you a set of all the permutations of two different islanders 
def get_islander_pairs(islanders):
    all_islander_combos = set(itertools.permutations(islanders, 2))
    return all_islander_combos

# Given a series of co-mentioned islanders list for each comment, it will create the following dataframe:
# CommentID | (Islander1, Islander2)       | (Islander1, Islander3)        | .... | (IslanderN, islanderM)
#     1     | 1 if both mentioned else NaN | 1 if both mentioned else NaN  | .... |  1 if both mentioned else NaN            
def full_cocurrent_matrix(cooccurrent_data, islanders):
    all_islander_combos = get_islander_pairs(islanders)
    full_co_current_matrix = cooccurrent_data.loc[cooccurrent_data.str.len()>1].apply(
        lambda series: {u:1 for u in get_islander_pairs(series)}
        ).apply(
        lambda series: pd.Series(series, index=all_islander_combos)
        )
    return full_co_current_matrix

def full_cocurrent_matrix_from_mentions(mentions_over_time, islanders):
    data = cooccurrent_data(mentions_over_time)
    return full_cocurrent_matrix(data, islanders)

