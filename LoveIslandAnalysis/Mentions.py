import pandas as pd
import numpy as np

import itertools


def find_names(search_name_series, text_series):
    all_words = text_series.str.strip().str.split(expand=True).stack().reset_index(level=1, drop=True)
    df = all_words.to_frame('body')
    for index in search_name_series.index:
        df[index] = 0
        name = search_name_series.loc[index]
        df.loc[all_words == name, index] = 1
    mentions_over_index = df.reset_index(names="id").groupby(by='id').sum(numeric_only=True)
    return mentions_over_index
    


def generate_search_name_series(islanders_df):
    name_series = pd.Series(
        data = islanders_df.Islander.str.split(' ').str.get(0).to_list(),
        index = islanders_df.Islander.to_list()
    )
    return name_series


def mentions_data(islanders_df, comment_df):
    name_series = generate_search_name_series(islanders_df)
    mentions_over_time = find_names(name_series, comment_df.body)
    mentions_over_time['day'] = comment_df['day']
    return name_series, mentions_over_time

def cooccur(series):
    s = series[:-2]
    if s.size > 0:
        return s.loc[s > 0].sort_values()[::-1].index.to_list()
    return pd.NA


def cooccurrent_data(mentions_over_time):
    return mentions_over_time.apply(cooccur, axis=1)


def get_islander_pairs(islanders):
    all_islander_combos = set(itertools.permutations(islanders, 2))
    return all_islander_combos

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

