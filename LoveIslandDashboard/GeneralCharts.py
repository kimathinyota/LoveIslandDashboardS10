import pandas as pd
import numpy as np

import altair as alt
from altair import datum


def createGeneralisedErrorChart(general_mean_std_by_day, y_value_title, slider_min=0.05, slider_max=10, slider_step=0.1, slider_name='multiple of std: ', day_name='day'):
    general_std_slider = alt.binding_range(min=slider_min, max=slider_max, step=slider_step, name=slider_name)
    general_std_selector = alt.selection_point(fields=['cutoff'],
                                bind=general_std_slider, value=[{'cutoff': 2}])


    general_error_base = alt.Chart(general_mean_std_by_day.reset_index()).transform_calculate(
        lower=datum.mean - general_std_selector.cutoff*datum.std,
        upper=datum.mean + general_std_selector.cutoff*datum.std,
    ).add_params(
        general_std_selector
    )

    general_error_band = general_error_base.mark_errorband().encode(
        x=day_name,
        y=alt.Y('lower:Q', title=y_value_title),
        y2='upper:Q'
    )

    general_mean_base = general_error_base.mark_line().encode(
        x=day_name,
        y=alt.Y('mean', title=y_value_title),
        color='Islanders:N'
    )

    mentions_mean_line = general_error_band + general_mean_base
    return mentions_mean_line
    

import re

def createGeneralisedBarAndLineGraphsFromMelt(daily_x_including_average_melted, value_name, agg_func_str, picked_days, should_include_average, value_title=None, var_name='day'):
    islander_selection = alt.selection_point(fields=['Islanders'])

    # initial_chart = alt.Chart(addm).transform_fold(
    #     picked_islanders + ['Average'],
    #     as_=['Islanders', value_name]
    # ).transform_filter(
    #     (datum.day >= picked_days[0]) &
    #     (datum.day <= picked_days[1])
    # )

    addm_melt = daily_x_including_average_melted

    # initial_chart = alt.Chart(addm_melt).transform_filter(
    #     (datum.day >= picked_days[0]) &
    #     (datum.day <= picked_days[1])
    # )

    initial_chart = alt.Chart(addm_melt)

    if not should_include_average:
        initial_chart = initial_chart.transform_filter(
            datum.Islanders != 'Average'
        )
    
    mentions_totals_bar_chart = initial_chart.mark_bar().encode(
        x= agg_func_str + "(" + value_name + "):Q",
        y='Islanders:N',
        color='Islanders:N',
        opacity = alt.condition(islander_selection, alt.value(1.0), alt.value(0.3))
    ).add_params(
        islander_selection
    ).interactive()

    value_title = value_name if value_title is None else value_title

    mentions_line_graph = initial_chart.encode(x=var_name + ':Q').mark_line().encode(
        y= alt.Y(value_name + ':Q', title=value_name),
        color='Islanders:N'
    ).transform_filter(
        islander_selection
    ).interactive()



    return mentions_totals_bar_chart, mentions_line_graph


def createGeneralisedBarAndLineGraphs(daily_x_including_average, value_name, agg_func_str, picked_islanders, picked_days, should_include_average, value_title=None, day_name='day'):
    addm = daily_x_including_average.loc[
        (daily_x_including_average[day_name] >= picked_days[0]) &
        (daily_x_including_average[day_name] <= picked_days[1])
    ]
    addm_melt = addm.melt(id_vars=['day'], value_name=value_name,  var_name='Islanders')
    return createGeneralisedBarAndLineGraphsFromMelt(addm_melt, value_name, agg_func_str, picked_days, should_include_average, value_title), addm_melt


def createMainEventsRuleChart(summaries_df, picked_days, selected_types):
    type_day = summaries_df.groupby(['Type', 'Day', 'Summary']).count().reset_index().rename(columns={'Day': 'day'})

    rule_chart = alt.Chart(type_day.loc[ (type_day['day'] >= picked_days[0]) 
                                        & (type_day['day'] <= picked_days[1])
                                        & (type_day['Type'].isin(selected_types))]).mark_rule(
        opacity=0.5,
        tooltip=True
    ).encode(
        x='day',
        strokeDash='Type',
        stroke='Type',
        tooltip=['day', 'Type', 'Summary']
    )
    return rule_chart