import pandas as pd
import numpy as np

import itertools

# For each text in the input text series, it will return the number of (first-name) matches for the input search_name_key_dic
# For text_series of [a: Tom is cool, b: Ronald and James are bad, c: James and Tom and John are good] 
# and search_name_key_dic = [Tom: R, Ronald: R, James: J, John: J]
# find_names(..) will return ...
#  id | T | R | J
#   a | 1 | 0 | 0
#   b | 0 | 1 | 1
#   c | 1 | 0 | 2
def find_names(search_name_key_dic, text_series):
    all_words = text_series.str.strip().str.split(expand=True).stack().reset_index(level=1, drop=True)
    df = all_words.to_frame('body')
    for name in search_name_key_dic:
        key = search_name_key_dic[name]
        if key not in df:
            df[key] = 0
        df.loc[all_words == name, key] += 1
    mentions_over_index = df.reset_index(names="id").groupby(by='id').sum(numeric_only=True)
    return mentions_over_index
    
# Given a series of islander full names it will extract the first names and return them (as dic) with their full names
# nicknames_dict is a dictionary of alternative names for islanders e.g. {"Ro": "Ron Hall", "Ronald": "Ron Hall"}
def generate_search_name_key_dic(islander_full_names_series, nicknames_dict=None):
    name_dict = pd.Series(
        index = islander_full_names_series.str.split(' ').str.get(0).to_list(),
        data = islander_full_names_series.to_list()
    ).to_dict()
    if nicknames_dict is not None:
        name_dict.update(nicknames_dict)
    return name_dict

# will find the quantity of first-name matches for each islander in each input comment  
def mentions_data(islanders_df, comment_df, nicknames_dict=None):
    name_dict = generate_search_name_key_dic(islanders_df.Islander, nicknames_dict)
    mentions_over_time = find_names(name_dict, comment_df.body)
    mentions_over_time['day'] = comment_df['day']
    return name_dict, mentions_over_time

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

