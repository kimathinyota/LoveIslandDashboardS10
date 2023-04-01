import pandas as pd
import numpy as np

import altair as alt
from altair import datum

from LoveIslandDashboard.GeneralCharts import createGeneralisedBarAndLineGraphs, createGeneralisedErrorChart



def createSentimentErrorChart(sentiments_mean_std):
    return createGeneralisedErrorChart(sentiments_mean_std, "Sentiment")
    


def createSentimentBarAndLineGraphs(daily_mentions_including_average, picked_islanders, picked_days, should_include_average, return_data=False):
    chart, data = createGeneralisedBarAndLineGraphs(daily_mentions_including_average, 'MeanSentiment', 'mean', picked_islanders, picked_days, should_include_average, 'Sentiment')
    if not return_data:
        return chart
    return chart, data





