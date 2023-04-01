# Script to determine the most mentioned islander on the sub-reddit at a given time frame
import mysql.connector
import pandas as pd

from LoveIslandAnalysis.PostFetcher import search_subreddit
from datetime import datetime, timedelta
import numpy as np
from markdown_plain_text.extention import convert_to_plain_text


# Day vs number of mentions on reddit discussion threads
# Points of analysis (sense-checking):
#  -> Is there a boost in islander discussion after dumping?

def is_mentioned(name, text):
    return text.find(name)

# Sentiment analysis
# most common words co-occurring with islander (represent as a wordcloud)
    # who is mentioned alongside islanders
# https://towardsdatascience.com/a-beginners-guide-to-sentiment-analysis-in-python-95e354ea84f6
# Day vs number of positive mentions on reddit discussion threads
# Day vs number of negative mentions on reddit discussion threads



# Islanders: pulled from https://en.wikipedia.org/wiki/List_of_Love_Island_(2015_TV_series)_contestants
# IslanderMentionedComments: commentID, islanderID, matched, matchedSimiliarity

import re

def get_islander_mentions(comments_body_series, names_series, chars=['.', ',']):
    # comments_body_df = comments_body_series.to_frame(name='body')
    matched_df = None
    emoji_series = None
    for index in names_series.index:
        names = names_series.loc[index]
        matched_series = comments_body_series.apply(lambda body: IslanderMatcher.find_associated_words_given_list(names, body, chars))
        emoji_series = matched_series.str[2] if emoji_series is None else emoji_series
        matched_df1 = pd.DataFrame(list(matched_series.str[:2]), index=matched_series.index,
                                   columns=pd.MultiIndex.from_tuples([(index, k) for k in ["NumOfMentions", "AssociatedWords"]]))

        matched_df = matched_df1 if matched_df is None else pd.concat( (matched_df, matched_df1), axis=1)
    matched_df["EmojiMap"] = emoji_series
    return matched_df



def create_influence_matrix(mention_df, islander_to_first):
    assoc_words = mention_df.xs(level=1, key='AssociatedWords', axis=1)
    islanders = islander_to_first.index
    first_to_islander = islander_to_first.reset_index().set_index("first")
    def filter_words(dic, chars_nums=np.array([0, 1]), islanders=first_to_islander.index):
        ks = filter(lambda k: k in islanders and len(k) > 1 and np.all(dic[k][0] <= chars_nums), dic.keys())
        return {k: dic[k][1] for k in ks}

    better_assoc_words = assoc_words.apply(lambda series: series.apply(filter_words))
    islanderToSum = {}
    for islander in islanders:
        word_mask = better_assoc_words[islander].str.len() > 0
        islanderToSum[islander] = better_assoc_words[islander].loc[word_mask].apply(pd.Series).sum()
    mentioned_together = pd.DataFrame(islanderToSum)

    mentioned_together["islander"] = first_to_islander
    mentioned_together = mentioned_together.set_index("islander")
    return mentioned_together.loc[sorted(mentioned_together.index), sorted(mentioned_together.columns)]











def get_total_mentions(islander_mentions_df):
    return islander_mentions_df.xs(level=1, axis=1, key="NumOfMentions", drop_level=True)


class IslanderMatcher:

    STOPWORDS = {'be', 'your', 'why', 'say', 'not', 'if', 'their', 'it', 'just', 'he', 'is', 'neither', 'are', 'you',
                 'wants', 'may', 'because', 'almost', 'as', 'cannot', 'from', 'than', 'them', 'among', 'this', 'too',
                 'off', 'then', 'its', 'nor', 'of', 'either', 'only', 'after', 'should', 'they', 'which', 'an', 'like',
                 'says', 'some', 'ever', 'these', 'a', 'rather', 'who', 'will', 'how', 'said', 'us', 'does', 'him',
                 'with', 'twas', 'our', 'were', 'we', 'must', 'or', 'been', 'own', 'when', 'have', 'any', 'she', 'do',
                 'since', 'there', 'whom', 'often', 'has', 'the', 'could', 'me', 'most', 'and', 'dear', 'her', 'that',
                 'about', 'hers', 'am', 'get', 'for', 'least', 'else', 'tis', 'but', 'on', 'can', 'where', 'into',
                 'however', 'his', 'no', 'was', 'had', 'by', 'every', 'did', 'i', 'to', 'able', 'my', 'what', 'yet',
                 'in', 'at', 'might', 'other', 'all', 'would', 'let', 'likely', 'so', 'across', 'got', 'while', 'also'}


    @staticmethod
    def replace_emoji_md(word):
        emojies = [m.group(0) for m in re.finditer(':[a-zA-Z]+:', word)]
        emojiMap = {}
        for n in range(len(emojies)):
            e = emojies[n]
            k = "EMJ" + str(n)
            emojiMap[k] = e
            word = re.sub(e, ' ' + k + ' ', word)
        return word, emojiMap

    @staticmethod
    def pad_special_chars_with_space(word):
        # first remove all apostrophes
        word = IslanderMatcher.merge_apostrophed_words(word)
        p = re.compile(r"([^a-zA-Z0-9]+)")
        return p.sub(' \\1 ', word)

    @staticmethod
    def merge_apostrophed_words(word):
        p = re.compile(r"([a-zA-z])'([a-zA-z])")
        return p.sub("\\1\\2", word)

    @staticmethod
    def merge_double_barreled_words(word):
        p = re.compile(r"([a-zA-z])-([a-zA-z])")
        return p.sub("\\1\\2", word)

    @staticmethod
    def pre_process_markdown(markdown, stopwords=STOPWORDS, emojiMap=None):
        plain_text = convert_to_plain_text(markdown)
        # plain_text = plain_text.lower()
        plain_text = IslanderMatcher.merge_double_barreled_words(plain_text)
        plain_text, emojiMap = IslanderMatcher.replace_emoji_md(plain_text) if emojiMap is None else (plain_text, emojiMap)
        plain_text = IslanderMatcher.pad_special_chars_with_space(plain_text)
        # remove stop words:
        plain_text = ' '.join([word for word in plain_text.split(' ') if word not in stopwords])
        return plain_text, emojiMap



    @staticmethod
    def find_associated_words_given_list(names_list, markdown, chars=['.'], stopwords=STOPWORDS):
        # Pre-processing for markdown
        plain_text, emojiMap = IslanderMatcher.pre_process_markdown(markdown, stopwords)
        stList = np.array(plain_text.split(' '))
        chars = np.array(chars)
        fullStopMask = stList == chars[:, None]

        # Merge the given entries
        num_names, wordToDistances = 0, {}
        for name in names_list:
            name_count, associated_word_to_distances, _ = IslanderMatcher.find_associated_words(name, markdown, chars, stopwords, plain_text, emojiMap, fullStopMask, stList)
            num_names += name_count
            for w in associated_word_to_distances:
                if w not in wordToDistances:
                    wordToDistances[w] = associated_word_to_distances[w]
                else:
                    distance_array, associated_word_count = associated_word_to_distances[w]
                    dArr, awc = wordToDistances[w]
                    dArr = np.where(distance_array < dArr, distance_array, dArr)
                    wordToDistances[w] = [dArr, awc]
        for name in names_list:
            if name in wordToDistances:
                wordToDistances.pop(name)
        return num_names, wordToDistances, emojiMap

    @staticmethod
    def find_associated_words(name, markdown, chars=['.'], stopwords=STOPWORDS, plain_text=None, emojiMap=None, fullStopMask=None, stList=None):
        plain_text, emojiMap = IslanderMatcher.pre_process_markdown(markdown, stopwords=stopwords, emojiMap=emojiMap) if plain_text is None else (plain_text, emojiMap)
        processed_name, _ = IslanderMatcher.pre_process_markdown(name)
        stList = np.array(plain_text.split(' ')) if stList is None else stList
        nameMask = stList == processed_name
        num_names = np.sum(nameMask)
        chars = np.array(chars)
        fullStopMask = stList == chars[:, None] if fullStopMask is None else fullStopMask
        wordToDistances = {}

        for word in stList[~nameMask]:
            # ignore:
            is_word = re.search("\w", word) is not None
            if is_word and word not in wordToDistances:
                wordToDistances[word] = [IslanderMatcher.full_stop_distance(processed_name, word, stList, maskA=nameMask,
                                                                          maskS=fullStopMask), 1]
            elif is_word:
                wordToDistances[word][1] += 1

        return num_names, wordToDistances, emojiMap

    @staticmethod
    def full_stop_distance(wordA, wordB, strList, chars=['.'], maskA=None, maskB=None, maskS=None):
        # Purpose: count number of full stops between two words in an input string
        strList = np.array(strList)
        maskA = strList == wordA if maskA is None else maskA
        maskB = strList == wordB if maskB is None else maskB
        maskS = strList == chars[:, None] if maskS is None else maskS

        # Algorithm:
        # indexAs = indexes in maskA where val is True
        # indexBs = indexes in maskB where val is True
        # For (indexA, indexB) ordered where index of indexA < indexB:
        #      full_point_distance = numOfTrues(maskS[indexA + 1: indexB])

        indexAs = np.nonzero(maskA)
        indexBs = np.nonzero(maskB)
        indexes = np.array(np.meshgrid(indexAs, indexBs)).T.reshape(-1, 2)
        min_distances = np.full(maskS.shape[0], float("inf"))

        for index in indexes:
            a = min(index)
            b = max(index)
            counts = np.sum(maskS[:, a + 1: b], axis=1)
            # full_stop_count = np.sum(maskS[:, a + 1: b], axis=1)
            min_distances = np.where(counts < min_distances, counts, min_distances)

            # full_stop_distance = min(full_stop_distance, full_stop_count)

        return min_distances


