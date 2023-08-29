# I went to interrogate the questions of: 
# 1) who does r/LoveIslandTV consider to be the most impactful islanders?
# 2) what were the biggest moments?

# what does it mean to be impactful?
#   * mentioned frequently 
#   * mentioned consistently
#   * inspired emotions (positive or negative sentiments)
#   * who dominates discussion in the biggest reddit moments?
#   * who casts the longest shaddow after a dumping?

# Splitting and expanding these out into categories:
# Mentions:
#   islander mention stats: mean, frequency (including and excluding upvotes)
#   how mentions change over time?

# Sentiment
#   islander sentiment stats: mean, frequency (including and excluding upvotes)
#   islander word bubbles
#   how mentions change over time?

# Moments
#  * episode title/day/description to: total number of comments/upvotes
#  * episode title/day/description to: who dominated the discussion?
#  * islanders: (min, maximum) episdoes

# Curious connections: 
#  * who were mentioned together most frequently?
#  * islander to most frequent mentions?


# Special-mentions:
# who casts the longest shaddow after a dumping?



import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import networkx as nx
import nx_altair as nxa
import LoveIslandAnalysis.Mentions as lm
from LoveIslandAnalysis.Islanders import fetch_and_clean_islanders_from_wikipedia
import LoveIslandDashboard.DataLoader as LIDataLoader
from altair import datum
from LoveIslandDashboard.AnalyticsLoader import AnalyticsLoader



#************************************
#********** PAGE TITLE ************** 
#************************************ 
st.header('Love Island: Ask the redditors')
st.caption('Analysis of Season 10\'s Discussion Threads in r/LoveIslandTV')
#************************************
#********** LOADING DATA ************ 
#************************************ 

@st.cache_data
def load_raw_data():
    comment_df, post_df, islanders_df, summaries_df = LIDataLoader.load_raw_data()
    # Adding the reunion to summaries_df
    discuss_mask = post_df.title.str.contains('Daily Discussion')
    post_mask = post_df.title.str.lower().str.contains('post')
    post_df['type'] = 'PRE'
    post_df.loc[post_mask, 'type'] = 'POST'
    post_df.loc[discuss_mask, 'type'] = 'MID'
    post_df['type'].astype(dtype='category')
    nicknames_dict = LIDataLoader.load_nicknames()
    
    return comment_df, post_df, islanders_df, summaries_df, nicknames_dict

data_load_state = st.empty()
data_load_state.text('Loading data...')
comment_df, post_df, islanders_df, summaries_df, nicknames_dict = load_raw_data()


analyticsLoader = AnalyticsLoader(islanders_df, comment_df, nicknames_dict)

data_load_state.text('Loading data...done!')
data_load_state.empty()


#************************************
#************** TABS ****************
#************************************ 

tabs = ['Moments', 'Mentions', 'Sentiments', 'Curious connections', 'Post Villa', 'Raws']
moments_tab, mentions_tab, sentiments_tab, connections_tab, special_tab, raws_tab = st.tabs(tabs)



#************************************
#*********** SIDE BAR *************** 
#************************************ 

islanders = list(islanders_df.Islander.unique())
not_dumped_mask = pd.isna(islanders_df.ShowLeaveDay)
not_dumped = islanders_df.Islander.loc[not_dumped_mask]


from datetime import datetime

with st.sidebar:

    format_func=lambda option: option.split(' ')[0]

    not_casa_mask = islanders_df.ShowEntryDay == islanders_df.MainVillaEntryDay
    casa_mask = ~not_casa_mask
    ogs_mask = islanders_df.ShowEntryDay == 1
    bombshell_mask = ~ogs_mask

    final_mask = ~(islanders_df.OnLeaveStatus == 'Dumped')

    find_islanders = lambda mask: islanders_df.Islander.loc[mask].to_list()

    icasa, iogs, ibombs, ifinals = ( find_islanders(mask) for mask in [casa_mask, ogs_mask, bombshell_mask, final_mask])
    
    st.markdown("## Islanders")

    pick_all = st.checkbox('Select all islanders', help='Undo to bring up islander picker')

    picked_all = islanders

    min_number = 8
    # default_islanders = islanders_df.sort_values(['ShowLeaveDay'])[::-1].Islander.iloc[:min_number].to_list()
    default_islanders = ifinals

    if not pick_all:

        strat_picker = st.checkbox('Stratify picker', help='Click to pick by islander type. Note: will reset your current choices')

        if strat_picker:
            picked_casa = st.multiselect("Pick from Casa islanders", options=icasa, default=[], format_func=format_func)
            picked_ogs = st.multiselect("Pick from OG islanders", options=iogs, default=[], format_func=format_func)
            picked_bombs = st.multiselect("Pick from Bombshell islanders", options=ibombs, default=[], format_func=format_func)
            picked_finals = st.multiselect("Pick from Finalists", options=ifinals, default=ifinals, format_func=format_func)
            picked_all = list(set(picked_casa + picked_ogs + picked_bombs + picked_finals))
        else:
            picked_all = st.multiselect("Pick from all islanders", options=islanders, default=default_islanders, format_func=format_func)


    picked_islanders = list(set(picked_all))

    st.markdown("## Day Range")


    last_day = (datetime.fromtimestamp(comment_df.createdDate.max()) - datetime(2023, 6, 5)).days + 1

    picked_days = (1, last_day)
    

    filter_days = st.checkbox('Filter day range', help='Opens day range slider when clicked')

    if filter_days:
        picked_days = st.slider( 'Select the days', 1, last_day, (1, last_day), help='Note: finale day is day 59')


    st.markdown("## Your Selections")


    if st.checkbox('View picked islanders'):
        st.write(islanders_df.loc[islanders_df.Islander.isin(picked_islanders)])

    if st.checkbox('View picked days'):
        uniques = lambda series: list(set(series))
        fsummaries = summaries_df.loc[:, ['Day', 'Type', 'Summary']]
        missing_days = set(np.arange(1, last_day + 1)) - set(summaries_df.Day.to_list())
        missing_entries = pd.DataFrame(data=[ [d, pd.NA, pd.NA] for d in missing_days],
                                columns=['Day', 'Type', 'Summary'])
        fsummaries = fsummaries.groupby('Day').agg({'Type': uniques, 'Summary':  lambda series: "\n".join(uniques(series))})
        fsummaries = pd.concat([fsummaries, missing_entries.set_index('Day')]).sort_index()
        st.write(fsummaries.loc[picked_days[0]:picked_days[1]])
    st.markdown("## Advanced")
    exclude_nan_sentiment = st.checkbox("Only allow comments where islander has been mentioned to contribute to mean sentiment", True)
 

picked_days_list = list(range(picked_days[0], picked_days[1] + 1, 1))

#************************************
#************ RAWS TAB ************** 
#************************************ 


# raws_tab.write(post_df[['createdDate', 'title', 'type', 'selfText', 'score', 'AuthorName', 'permalink', 'insertionTimestamp']])

apply_filters = raws_tab.checkbox('Apply filters')



display_post = post_df[['type', 'createdDate', 'title', 'selfText', 'score', 'AuthorName', 'permalink', 'insertionTimestamp']]
display_post.columns =['Type of Discussion Thread', 'Date', 'Title', 'Text', 'Score', 'Author\'s username', 'Permalink', 'Date Inserted (to db)']

display_post['Date'] = pd.to_datetime(display_post['Date'], unit='s')
display_post['Day'] = (display_post["Date"] - pd.to_datetime('2023-6-05')).dt.days + 1
display_post['Date Inserted (to db)'] = pd.to_datetime(display_post['Date Inserted (to db)'], unit='s')
display_post = display_post.loc[:,['Day', 'Type of Discussion Thread', 'Date', 'Title', 'Text', 'Score', 'Author\'s username', 'Permalink', 'Date Inserted (to db)']]

display_comment = comment_df[['day', 'createdDate', 'body', 'score', 'compound', 'AuthorName', 'permalink', 'insertionTimestamp']]
display_comment.columns=['Day', 'Date', 'Text', 'Score', 'Sentiment Score', 'Author\'s username', 'Permalink', 'Date Inserted (to db)']


display_comment['Date'] = pd.to_datetime(display_comment['Date'], unit='s')
display_comment['Date Inserted (to db)'] = pd.to_datetime(display_comment['Date Inserted (to db)'], unit='s')

display_islanders = islanders_df
if apply_filters:
    display_islanders = display_islanders.loc[islanders_df.Islander.isin(picked_islanders)]
    display_post = display_post.loc[display_post.Day.isin(picked_days_list)]
    display_comment = display_comment.loc[display_comment.Day.isin(picked_days_list)]
    display_summaries_df = summaries_df.loc[summaries_df.Islanders.isin(picked_islanders) & summaries_df.Day.isin(picked_days_list)].groupby(['Type', 'Summary', 'Day']).agg({'Islanders': list}).reset_index().loc[:, ['Day', 'Type', 'Summary', 'Islanders']].sort_values('Day').reset_index()
else:
    display_summaries_df = summaries_df.groupby(['Type', 'Summary', 'Day']).agg({'Islanders': list}).reset_index().loc[:, ['Day', 'Type', 'Summary', 'Islanders']].sort_values('Day').reset_index()



raws_tab.subheader('Islanders')
raws_tab.write(display_islanders)
raws_tab.caption('Data extracted from https://en.wikipedia.org/wiki/Love_Island_(2015_TV_series,_series_10)')

raws_tab.subheader('Posts')
raws_tab.write(display_post)
raws_tab.caption('Data fetched from r/LoveIslandTV via [PRAW](https://praw.readthedocs.io/en/stable/)')

raws_tab.subheader('Comments')
raws_tab.write(display_comment)
raws_tab.caption('Data fetched from r/LoveIslandTV via [PRAW](https://praw.readthedocs.io/en/stable/)')

raws_tab.subheader('Events')
raws_tab.write(display_summaries_df)
raws_tab.caption('Data extracted from https://en.wikipedia.org/wiki/Love_Island_(2015_TV_series,_series_10)')





#************************************
#********** MENTIONS TAB ************ 
#************************************ 

from LoveIslandDashboard.Mentions import createTotalMentionsErrorChart, createTotalMentionsBarAndLineGraphs


@st.cache_data
def mentions_data():
    return analyticsLoader.load_mentions_data(use_cache=True)

mentions_over_time = mentions_data()


@st.cache_data
def day_by_day_mentions(mentions_over_time):
    return mentions_over_time.groupby(by=['day']).sum()

all_day_by_day_mentions = day_by_day_mentions(mentions_over_time)
addm = all_day_by_day_mentions.loc[:,picked_islanders]


mentions_mean_std = all_day_by_day_mentions.loc[picked_days[0]:picked_days[1]+1].agg(['mean', 'std'], axis=1)
mentions_mean_std['Islanders'] = 'Average'
addm['Average'] = mentions_mean_std['mean']
addm = addm.reset_index()


def generate_about_for_interactive_bar_line_graph(bar_x_value, line_y_value, bar_y_value="chosen islanders", line_x_value="chosen day range"):
    bar_graph_description = "* Interactive bar graph showing the {0} for each of the {1} across all of the discussion threads within the chosen day range".format(bar_x_value, bar_y_value)
    line_graph_description = "* Interactive line graph showing how the {0} vary over the {1} for each islander".format(line_y_value, line_x_value)
    interactivity_about = "* Interactivity Rules:\na) Click each bar to isolate its corresponding data on the two graph\nb) Hold shift to select multiple bars\nc)Tooltips will show when you hover over the data\nd)Interactive scale zooming supported"
    return "\n".join([ "Graphs", bar_graph_description, line_graph_description, interactivity_about])



addm_melt = None

type_day = summaries_df.groupby(['Type', 'Day']).count().reset_index()[['Type', 'Day']].rename(columns={'Day': 'day'})


from LoveIslandDashboard.GeneralCharts import createMainEventsRuleChart




# mentions_tab.altair_chart(rule_chart, True)

if len(picked_islanders) > 0:

    
    with mentions_tab.expander("About"):
        st.markdown(generate_about_for_interactive_bar_line_graph("total mentions", "total mentions"))

    with mentions_tab.expander('Options'):
        mentions_type_selector = st.multiselect("Select all Main Events", type_day['Type'].unique(), ['Entrances', 'Exits'], key='mentions-events-selector')
        include_mentions_average = st.checkbox("Include Average Islander", key='mentions_average')
        
    mentions_rule_chart = createMainEventsRuleChart(summaries_df, picked_days, mentions_type_selector)
    mentions_mean_line = createTotalMentionsErrorChart(mentions_mean_std)

    (mentions_totals_bar_chart, mentions_line_graph), data = createTotalMentionsBarAndLineGraphs(addm, picked_islanders=picked_islanders, picked_days=picked_days,
                                        should_include_average=include_mentions_average, return_data=True)
    addm_melt = data

 

    if include_mentions_average:
        combined_chart = alt.vconcat(mentions_totals_bar_chart, mentions_rule_chart + mentions_mean_line + mentions_line_graph)
    else:
        combined_chart = alt.vconcat(mentions_totals_bar_chart, mentions_rule_chart + mentions_line_graph)

    mentions_tab.altair_chart(combined_chart, True)
else:
    mentions_tab.error("Please select at least 1 islander")



#************************************
#******** CO-MENTIONS TAB *********** 
#************************************ 





from LoveIslandDashboard.CoMentions import create_interactive_influence_graph, standardise_edges_dataframe

@st.cache_data
def full_cocurrent_matrix_from_mentions():
    return analyticsLoader.load_comentions_data(use_cache=True)

full_cocurrent_matrix = full_cocurrent_matrix_from_mentions()
full_cocurrent_matrix['day'] = mentions_over_time['day']

@st.cache_data
def edge_matrix_by_day(full_cocurrent_matrix):
    edge_matrix_by_day = full_cocurrent_matrix.groupby('day').sum().unstack().reset_index()
    edge_matrix_by_day.columns = ['x', 'y', 'day', 'weight']
    return edge_matrix_by_day
     
@st.cache_data
def calculate_centrality(edges_dataframe):
    G = nx.from_pandas_edgelist(edges_dataframe, 'x', 'y', ['weight'])
    b = nx.eigenvector_centrality(G, weight = 'weight')
    return b

def to_edge_frame(edge_matrix_by_day):
    return edge_matrix_by_day.groupby(by=['x', 'y'])[ ['x','y','weight']].sum(numeric_only=True).reset_index()

@st.cache_data
def filter_edge_matrix_by_days(edge_matrix_by_day, start_day, end_day):
    return edge_matrix_by_day.loc[ (edge_matrix_by_day.day >= start_day) &
                                   (edge_matrix_by_day.day <= end_day)]


full_edge_matrix_by_day = edge_matrix_by_day(full_cocurrent_matrix)

filtered_edge_matrix_by_day = filter_edge_matrix_by_days(full_edge_matrix_by_day, picked_days[0], picked_days[1])
filtered_edge_df_by_day = to_edge_frame(filtered_edge_matrix_by_day)
full_centrality = calculate_centrality(filtered_edge_df_by_day)


with connections_tab.expander('About'):
    st.markdown('Below is an interactive graph that displays the total co-mentions (i.e. mentioned together in the same comment) for each pair of chosen (in the sidebar) islanders in the chosen day range. The the thickness of the edge connecting two denotes the weight of the connection.')
    st.markdown('By default the most central islander is isolated (i.e. brightest coloured node on the graph). ')
    st.markdown('The Pie graph shows the co-mention breakdown for a given node (islander), where the partner legend is sorted in descending order of total co-mentions. By default, it shows the breakdown for the most central islander in the graph.')
    st.markdown('Interactivity rules: \n1. Hover over the nodes (coloured circle) for a toolbar displaying the name, centrality and the closest neighbour for the islander it represents \n2. Click a node in the graph to see the Pie-chart breakdown of its co-mentions \n3. Click outside the graph nodes to exit selection mode--all the nodes will regain their brightness and the pie chart will dissapear. \n4. Hover over a sector in the Pie chart to see the name, partner and weight (i.e. total co-mention count) of their connection for the selected node.\n5. Interactive scale zooming supported')
    



if len(picked_islanders) > 1:

    if connections_tab.checkbox('Generate full Matrix', help='It will show a colour map of all of the co-mentions for every single islander irrespective of your day or islander selections'):
        c = alt.Chart(standardise_edges_dataframe(filtered_edge_df_by_day).reset_index()).mark_bar().encode(
        x='name',
        y='partner',
        color='weight:Q')
        connections_tab.altair_chart(c)
    
    interactive_influence_graph_and_donut = create_interactive_influence_graph(filtered_edge_df_by_day, full_centrality, picked_islanders)
    if interactive_influence_graph_and_donut is not None:
        connections_tab.altair_chart(interactive_influence_graph_and_donut, True)
    else:
        connections_tab.error("None of the chosen islanders are mentioned alongside each other")
else:
    connections_tab.error("Please select at least 2 islanders")


#************************************
#******** SENTIMENTS TAB ************ 
#************************************ 

@st.cache_data
def day_by_day_sentiments(sentiments_over_time):
    return sentiments_over_time.groupby(by=['day']).mean()


def labelled_data(compound_series, neg_limit, pos_limit):
    d = compound_series.to_frame()
    d['label'] = 0
    d.loc[d.compound <= neg_limit, 'label'] = -1
    d.loc[d.compound >= pos_limit, 'label'] = 1
    return d



sentiments_over_time = mentions_over_time.iloc[:, :-1]


sentiments_over_time.iloc[:, :-1] = np.where(sentiments_over_time.iloc[:, :-1] > 0, True, np.nan if exclude_nan_sentiment else False)

sentiments_over_time = sentiments_over_time.mul(comment_df.compound, axis=0)
sentiments_over_time['day'] = mentions_over_time.day



all_day_by_day_sentiments = day_by_day_sentiments(sentiments_over_time).fillna(0)

sentiment_mean_std = all_day_by_day_sentiments.loc[picked_days[0]:picked_days[1]].agg(['mean', 'std'], axis=1)
sentiment_mean_std['Islanders'] = 'Average'
# sentiment_mean_std = get_means_and_stds_by_day_df(sentiments_over_time)

adds = all_day_by_day_sentiments.loc[:,picked_islanders]
adds = adds.reset_index()
adds['Average'] = sentiment_mean_std['mean']

from LoveIslandDashboard.Sentiments import createSentimentBarAndLineGraphs, createSentimentErrorChart



adds_melt = None

if len(picked_islanders) > 0:

    with sentiments_tab.expander("About"):
        st.markdown('Note: A sentiment is a score from -1 to 1 denoting how `positive` a body of text is, where -1 is extremely negative, 0 is neutral and 1 is extremely positive.')
        st.markdown('Sentiments have been calculated for each comment using the Vader model (Hutto, C.J. & Gilbert, E.E. (2014)). If an islander is found in a comment, regardless of how many occurrences of this and other islanders alongside it, the sentiment of that comment contributes only once to the aggregated sentiment for that islander. {0} comments where islanders {1} mentioned contribute{2} to the mean sentiment for said islanders.'.format("Only" if exclude_nan_sentiment else "Even", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                "are" if exclude_nan_sentiment else "aren't",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                "" if exclude_nan_sentiment else "(only 0)"))
        st.markdown(generate_about_for_interactive_bar_line_graph("mean sentiment", "mean sentiments"))

    sentiment_mean_line = createSentimentErrorChart(sentiment_mean_std)

    with sentiments_tab.expander("Options"):
        sentiments_type_selector = st.multiselect("Select all Main Events", type_day['Type'].unique(), ['Entrances', 'Exits'], key='sentiments-events-selector')
        include_sentiment_average = st.checkbox("Include Average Islander", key='sentiment_average')

    (totals_sentiment_bar_chart, sentiment_line_graph), data = createSentimentBarAndLineGraphs(adds, picked_islanders, picked_days, include_sentiment_average, return_data=True)
    adds_melt = data

    sentiments_rule_chart = createMainEventsRuleChart(summaries_df, picked_days, sentiments_type_selector)

    if include_sentiment_average:
        combined_chart = alt.vconcat(totals_sentiment_bar_chart, sentiments_rule_chart + sentiment_mean_line + sentiment_line_graph)
    else:
        combined_chart = alt.vconcat(totals_sentiment_bar_chart, sentiments_rule_chart + sentiment_line_graph)

    sentiments_tab.altair_chart(combined_chart, True)
else:
    sentiments_tab.error("Please select at least 1 islander")



#************************************
#********** MOMENTS TAB ************* 
#************************************ 


# Highest sentiment days across all islanders


filtered_islanders = islanders_df.Islander.to_list()

bottom_sentiments = all_day_by_day_sentiments.loc[picked_days[0]:picked_days[1], filtered_islanders].mean(axis=1).sort_values().reset_index(name='Mean').reset_index()


reduced_summaries_df = summaries_df.groupby(['Type', 'Summary', 'Day'])['Islanders'].agg(list).reset_index()


filtered_day_by_day_mentions = all_day_by_day_mentions.loc[picked_days[0]:picked_days[1], picked_islanders]
filtered_day_by_day_sentiments = all_day_by_day_sentiments.loc[picked_days[0]:picked_days[1], picked_islanders]


bottom_mentions = filtered_day_by_day_mentions.sum(axis=1).sort_values().reset_index(name='Mean').reset_index()
bottom_sentiments = filtered_day_by_day_sentiments.mean(axis=1).sort_values().reset_index(name='Mean').reset_index()
reduced_summaries_df2 = reduced_summaries_df.merge(bottom_mentions, how='right', left_on='Day', right_on='day')


# moments_options = st.expander('Options')

max_bins = len(bottom_mentions)

with moments_tab.expander('About'):
    st.markdown('Types of main events: Entrances; Exits: e.g. dumpings, the finale, etc.; Dates; Challenges e.g. Movie Night, Snogging etc.; Coupling; Reunion')
    st.markdown('The Highs are the top three day block (as chosen in options) that generated the highest total mentions (or mean sentiment, if selected in options) for the chosen (in sidebar) islanders and date range. By contrast, the Lows are the bottom three ones.')
    


with moments_tab.expander('Options'):
    granuality = st.slider("Maximum size of day block: e.g. 7 for weekly (approx.) blocks", 1, int(max_bins/3), 3, 1)
    is_sentiments = st.checkbox('What about the vibes tho?', help='Switch to showing highest and lowest sentiment days')






#['binned'] = pd.cut(bottom_mentions['day'], np.arange(sB-0.5, eB, granuality))

bottom_mentions = bottom_mentions.sort_values('day').reset_index(drop=True).reset_index(names='Order')

sB, eB = 0, len(bottom_mentions) + 1
bottom_mentions['binned'] = pd.cut(bottom_mentions['Order'], np.arange(sB-0.5, eB + granuality, granuality))
bottom_sentiments['binned'] = pd.cut(bottom_sentiments['day'], int(max_bins/granuality), duplicates='drop', ordered=True)

limited_reduced = reduced_summaries_df2[['Type', 'Summary', 'day', 'Mean', 'Islanders']]

limited_reduced = limited_reduced.sort_values(by='day')


# moments_tab.write(limited_reduced)

groups = bottom_mentions.groupby(['binned']).agg({'day': list, 'Mean': 'sum'}).reset_index()[['day', 'Mean']].sort_values(by='Mean')
groups = groups.loc[groups.day.str.len() > 0]
sgroups = bottom_sentiments.groupby(['binned']).agg({'day': list, 'Mean': 'mean'}).reset_index()[['day', 'Mean']].sort_values(by='Mean')
sgroups = sgroups.loc[sgroups.day.str.len() > 0]




moment_records = lambda day_mean_series : limited_reduced.loc[limited_reduced.day.isin(day_mean_series.day)]


def to_display_text(moments_records):
    moments_records = moments_records.sort_values(by=['day'])
    main_events = set(moments_records.loc[moments_records['Type'].notna(),'Type'].to_list())

    texts = list(set(moments_records.loc[moments_records['Summary'].notna(), 'Summary'].to_list()))
    sumaries = "* " + "\n* ".join(texts) if len(texts) > 0 else "No main events happened"
    list_of_islanders_list = moments_records.loc[moments_records.Islanders.notna(), 'Islanders'].to_list()
    islanders_involved_in_main_events = list(set([item for sublist in list_of_islanders_list for item in sublist]))

    days_list = list(set(moments_records['day'].to_list()))
    day_range = 'Day ' + str(days_list[0]) if len(days_list) == 1 else "Days " + str(min(days_list)) + " to " + str(max(days_list))
    return main_events, sumaries, islanders_involved_in_main_events, day_range





select_group = lambda is_sentiment: sgroups if is_sentiment else groups

if len(select_group(is_sentiments).index) >= 3:
    moments_tab.subheader('The Highs...')

    hleft, hmiddle, hright = moments_tab.columns(3)

    first, second, third = select_group(is_sentiments).iloc[-1], select_group(is_sentiments).iloc[-2], select_group(is_sentiments).iloc[-3]
    first_records, second_records, third_records = (moment_records(x) for x in [first, second, third])

    stat_title = "Average Sentiment: " if is_sentiments else "Total Mentions: "

    with hmiddle:
        main_events, sumaries, islanders_involved_in_main_events, day_range = to_display_text(first_records)
        emoji = ":smile: " if is_sentiments else ":bomb: "
        st.subheader(emoji + day_range)
        st.caption(stat_title + str(round(first.Mean, 2)))
        st.markdown("Main Events: " + ", ".join(main_events))
        with st.expander("Summary"):
            st.markdown(sumaries)
        

    with hleft:
        main_events, sumaries, islanders_involved_in_main_events, day_range = to_display_text(second_records)
        st.subheader(":two: " + day_range)
        st.caption(stat_title + str(round(second.Mean, 2)))
        st.markdown("Main Events: " + ", ".join(main_events))
        with st.expander("Summary"):
            st.markdown(sumaries)


    with hright:
        main_events, sumaries, islanders_involved_in_main_events, day_range = to_display_text(third_records)
        st.subheader(":three: " + day_range)
        st.caption(stat_title + " " + str(round(third.Mean, 2)))
        st.markdown("Main Events: " + ", ".join(main_events))
        with st.expander("Summary"):
            st.markdown(sumaries)

    # moments_tab.write(pd.cut(reduced_summaries_df2.Day, 20))

    moments_tab.subheader('The Lows...')

    lleft, lmiddle, lright = moments_tab.columns(3)

    first, second, third = select_group(is_sentiments).iloc[0], select_group(is_sentiments).iloc[1], select_group(is_sentiments).iloc[2]
    first_records, second_records, third_records = (moment_records(x) for x in [first, second, third])

    with lmiddle:
        main_events, sumaries, islanders_involved_in_main_events, day_range = to_display_text(first_records)
        emoji = ":rage: " if is_sentiments else ":zzz: "
        st.subheader(emoji + day_range)
        st.caption(stat_title + " " + str(round(first.Mean, 2)))
        st.markdown("Main Events: " + ", ".join(main_events))
        with st.expander("Summary"):
            st.markdown(sumaries)
        

    with lleft:
        main_events, sumaries, islanders_involved_in_main_events, day_range = to_display_text(second_records)
        st.subheader(":two: " + day_range)
        st.caption(stat_title + " " + str(round(second.Mean, 2)))
        st.markdown("Main Events: " + ", ".join(main_events))
        with st.expander("Summary"):
            st.markdown(sumaries)


    with lright:
        main_events, sumaries, islanders_involved_in_main_events, day_range = to_display_text(third_records)
        st.subheader(":three: " + day_range)
        st.caption(stat_title + " " + str(round(third.Mean, 2)))
        st.markdown("Main Events: " + ", ".join(main_events))
        with st.expander("Summary"):
            st.markdown(sumaries)
else:
    moments_tab.error('Not enough data to display. Try lowering the max day block size or increasing the day range ')






# moments_tab.write(pd.concat([comment_df, mentions_over_time])) 

#************************************
#********** SPECIAL TAB ************* 
#************************************ 






melt_filtered_day_by_day_mentions = filtered_day_by_day_mentions.reset_index().melt(id_vars=['day'], value_name='Total Mentions', var_name='Islanders')



# The Molly Marsh exemption: she left the show twice, below will only consider the last 

not_molly_mask = islanders_df.Islander != 'Molly Marsh'
exclude_wrong_molly_mask = not_molly_mask | (~not_molly_mask & (islanders_df.ShowLeaveDay > 40))

#with_exit_days = melt_filtered_day_by_day_mentions.merge(right=islanders_df[['Islander', 'ShowLeaveDay']], how='left', left_on='Islanders', right_on='Islander')


with_exit_days = melt_filtered_day_by_day_mentions.merge(right=islanders_df.loc[exclude_wrong_molly_mask, ['Islander', 'ShowLeaveDay']], how='left', left_on='Islanders', right_on='Islander')


post_villa_day_title = "Number of days post villa"
with_exit_days[post_villa_day_title] = with_exit_days['day'] - with_exit_days['ShowLeaveDay']



melt_day_by_day_mentions = all_day_by_day_mentions.loc[picked_days[0]:picked_days[1]+1].reset_index().melt(id_vars=['day'], value_name='Total Mentions', var_name='Islanders')



melt_day_by_day_mentions_with_exit_days = melt_day_by_day_mentions.merge(right=islanders_df.loc[exclude_wrong_molly_mask, ['Islander', 'ShowLeaveDay']], how='left', left_on='Islanders', right_on='Islander')
melt_day_by_day_mentions_with_exit_days[post_villa_day_title] = melt_day_by_day_mentions_with_exit_days['day'] - melt_day_by_day_mentions_with_exit_days['ShowLeaveDay']
post_villa_mentions_mean_std = melt_day_by_day_mentions_with_exit_days.groupby(post_villa_day_title).agg({'Total Mentions': ['mean', 'std']})
post_villa_mentions_mean_std.columns = ['mean', 'std']
post_villa_mentions_mean_std['Islanders'] = 'Average'

with special_tab.expander('About'):
    st.markdown(generate_about_for_interactive_bar_line_graph('total post-villa mentions', "total mentions", line_x_value="post-villa days"))

with special_tab.expander('Options'):
    max_post_villa_days = st.slider("Max number of days post villa considered:", 1, 63, 7, 1)
    include_post_average = st.checkbox('Include Average Islander')


with_exit_days = melt_day_by_day_mentions_with_exit_days.loc[melt_day_by_day_mentions_with_exit_days.Islanders.isin(picked_islanders) &
                                                             (melt_day_by_day_mentions_with_exit_days[post_villa_day_title] >= 1) &
                                                             (melt_day_by_day_mentions_with_exit_days[post_villa_day_title] <= max_post_villa_days),
                                                             [post_villa_day_title, 'Islanders', 'Total Mentions']]
with_exit_days_with_average = pd.concat([with_exit_days, post_villa_mentions_mean_std.loc[1:max_post_villa_days].rename(columns={'mean': 'Total Mentions'})])


from LoveIslandDashboard.GeneralCharts import createGeneralisedBarAndLineGraphsFromMelt, createGeneralisedErrorChart


if len(with_exit_days) > 0:

    chart = createGeneralisedErrorChart(post_villa_mentions_mean_std.loc[1:max_post_villa_days], "Total Mentions", day_name=post_villa_day_title)
    mentions_totals_bar_chart, mentions_line_graph = createGeneralisedBarAndLineGraphsFromMelt(with_exit_days_with_average, 'Total Mentions', 'sum', picked_days, include_post_average, var_name=post_villa_day_title)
    if not include_post_average:
        special_tab.altair_chart(alt.vconcat(mentions_totals_bar_chart, mentions_line_graph), True)
    else:
        special_tab.altair_chart(alt.vconcat(mentions_totals_bar_chart, mentions_line_graph + chart), True)

else:
    special_tab.error('No data found within your selections')

from datetime import time

# What's left?
# Need to credit wikipedia for sources of data (link to it ideally)
# Maybe let people download the data

# Deployment stuff
# Put on github: write a few entries talking about it
# Extension: write a medium article explaining the project