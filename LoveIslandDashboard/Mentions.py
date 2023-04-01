import LoveIslandAnalysis.Mentions as lm
import pandas as pd
import numpy as np

import altair as alt
from altair import datum

from LoveIslandDashboard.GeneralCharts import createGeneralisedBarAndLineGraphs, createGeneralisedErrorChart


def createTotalMentionsErrorChart(mentions_mean_std, return_data=False):
    return createGeneralisedErrorChart(mentions_mean_std, "Total Mentions")
    


def createTotalMentionsBarAndLineGraphs(daily_mentions_including_average, picked_islanders, picked_days, should_include_average, return_data=False):
    chart, data = createGeneralisedBarAndLineGraphs(daily_mentions_including_average, 'Total Mentions', 'sum', picked_islanders, picked_days, should_include_average, "Total Mentions")
    if not return_data:
        return chart
    return chart, data

    

        