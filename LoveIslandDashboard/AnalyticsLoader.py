import LoveIslandAnalysis.Mentions as lm
from os.path import exists
import pandas as pd

# There are over 30,000 comments and over 30 islanders
# Calculating total mentions of each islander in each comment alone involves ~ 1 million data points
# This is fast (because of c) but doing this constantly is tedius, especially when this data is fairly static
# Goals of this file
#  * Calculate and store necessary intermediate data for analytics for quicker retrieval
#  * Allow recalculation and re-storage of this data in case of corruptions or new data influx


class AnalyticsLoader:

    MENTIONS_DATA_PATH = "LoveIslandDashboard/Data/Analytics/mentions_data.csv"
    COMENTIONS_DATA_PATH = "LoveIslandDashboard/Data/Analytics/comentions_data.csv"

    def does_mentions_data_exist(self):
        return exists(AnalyticsLoader.MENTIONS_DATA_PATH)

    def does_comentions_data_exist(self):
        return exists(AnalyticsLoader.COMENTIONS_DATA_PATH)


    def replace_mentions_data(self):
        name_dict, mentions_over_time = lm.mentions_data(self.islanders_df, self.comments_df, self.nicknames_dict)
        mentions_over_time.to_csv(AnalyticsLoader.MENTIONS_DATA_PATH, index=True)
        return mentions_over_time


    def load_mentions_data(self, force_load=False, use_cache=False):
        # use_cache means it keeps this instance reference to the loaded data (for even faster retrieval)
        if use_cache and (self.mentions_data_cache is not None):
            return self.mentions_data_cache
        if force_load or (not self.does_mentions_data_exist()):
            temp = self.replace_mentions_data()
        else:
            temp = pd.read_csv(AnalyticsLoader.MENTIONS_DATA_PATH).set_index('id')
        if use_cache:
            self.mentions_data_cache = temp
        return temp


    def encode_co_mentions_data(self, matrix):
        matrix.columns = self.encoder.encode_pairs_list(matrix.columns.to_flat_index().to_list())
        return matrix
    

    def decode_co_mentions_data(self, encoded_matrix):
        to_int_tuples = lambda str_tuples: pd.Series(str_tuples).str.extract(r'([\d]+)\D+([\d]+)').astype(int).apply(tuple, axis=1).to_list()
        encoded_matrix.set_index('id', inplace=True)
        encoded_matrix.columns = pd.MultiIndex.from_tuples(self.encoder.decode_pairs_list(to_int_tuples(encoded_matrix.columns)))
        return encoded_matrix


    def replace_comentions_data(self, force_load_mentions=False, use_cache=False):
        mentions_over_time = self.load_mentions_data(force_load_mentions, use_cache)
        matrix =  lm.full_cocurrent_matrix_from_mentions(mentions_over_time, self.islanders_df.Islander.to_list())
        self.encode_co_mentions_data(matrix.copy()).to_csv(AnalyticsLoader.COMENTIONS_DATA_PATH, index=True)
        return matrix

    
    def load_comentions_data(self, force_load_all=False, force_load_just_me=False, use_cache=False):
        # use_cache means it keeps this instance reference to the loaded data (for even faster retrieval)
        if use_cache and (self.comentions_data_cache is not None):
            return self.comentions_data_cache
        if force_load_all or force_load_just_me or (not self.does_comentions_data_exist()):
            temp = self.replace_comentions_data(force_load_all, use_cache)
        else:
            temp = self.decode_co_mentions_data(pd.read_csv(AnalyticsLoader.COMENTIONS_DATA_PATH))
        if use_cache:
            self.comentions_data_cache = temp
        return temp


    def __init__(self, islanders_df, comments_df, nicknames_dict=None):
        self.islanders_df = islanders_df
        self.comments_df = comments_df
        self.mentions_data_cache = None
        self.comentions_data_cache = None
        self.encoder = IslanderEncoder(islanders_df)
        self.nicknames_dict=nicknames_dict


class IslanderEncoder:

    @staticmethod
    def unique_islander_code(islanders_df):
        #print(islanders_df)
        islanders_df_rem_dup = islanders_df.groupby("Islander").agg(lambda ser: ser.iloc[0]).reset_index()
        # assign based on position it entered
        first_names = islanders_df_rem_dup.Islander.str.split(' ').str.get(0).str.replace(r'[^a-zA-Z0-9]', '').to_frame('first')
        first_names['Islander'] = islanders_df_rem_dup.Islander
        first_names['ShowEntryDay'] = islanders_df.ShowEntryDay
        islander_keys = first_names.sort_values(by=['ShowEntryDay', 'Islander']).Islander.reset_index().Islander.reset_index().set_index('Islander')
        return islander_keys['index']
    
    def decode(self, islander_as_code):
        return self.decoder.loc[islander_as_code]
    
    def encode(self, islander):
        return self.encoder.loc[islander]
    
    def encode_pairs_list(self, islander_pairs):
        return list(map(lambda pair: (self.encode(pair[0]), self.encode(pair[1])), islander_pairs))

    def decode_pairs_list(self, code_pairs):
        return list(map(lambda pair: (self.decode(pair[0]), self.decode(pair[1])), code_pairs))
    
    def __init__(self, islanders_df):
        self.encoder = IslanderEncoder.unique_islander_code(islanders_df)
        self.decoder = self.encoder.to_frame().reset_index().set_index('index').Islander
        


