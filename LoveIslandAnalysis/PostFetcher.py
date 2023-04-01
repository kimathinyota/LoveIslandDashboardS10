
from datetime import datetime, timedelta
from praw.models import MoreComments
import pandas as pd

# Code to manage fetching data from reddit api via PRAW into pandas dataframes for analysis

# Praw class instances don't always have the attributes they can have as stated in the docs
# they are lazy e.g. if a post isn't a Poll it won't return None for post.poll_data, ...
# ... post.poll_data won't exist as an attribute
# below is the method designed to ensure post.poll_data = val | None at all times
def value(obj, attr):
    if hasattr(obj, attr):
        return obj.__getattribute__(attr)
    return None

# SCHEMA format:
# Schema is only defined once in this whole file
#   FieldName -> (Praw class attribute name, MySQL type)
# If PRAW decides to change an attribute name e.g. Submission.id becomes Submission.sid, we can change it once here
# and it will ensure the dataframe indexes are consistent
# MySQL type is also included for love_island_database
def get_schemas():
    COMMENT_SCHEMA = {
        "CommentID": ("id", "VARCHAR(255)"), "PostID": (None, "VARCHAR(255)"),
        "permalink": ("permalink", "VARCHAR(255)"),
        "AuthorID": (None, "VARCHAR(255)"), "AuthorName": (None, "VARCHAR(255)"),
        "createdDate": ("created_utc", "INT(11)"), "body": ("body", "TEXT"),
        "score": ("score", "INTEGER"), "parentID": ("parent_id", "VARCHAR(255)"),
        "insertionTimestamp": (None, "INT(11)")
    }
    POST_SCHEMA = {
        "PostID": ("id", "VARCHAR(255)"), "name": ("name", "VARCHAR(255)"), "title": ("title", "TEXT"),
        "permalink": ("permalink", "VARCHAR(255)"), "createdDate": ("created_utc", "INT(11)"),
        "AuthorID": (None, "VARCHAR(255)"), "AuthorName": (None, "VARCHAR(255)"),
        "isStickied": ("stickied", "BOOLEAN"), "linkFlairText": ("link_flair_text", "TEXT"),
        "linkUrl": ("url", "VARCHAR(255)"), "isLocked": ("locked", "BOOLEAN"), "edited": ("edited", "INT(11)"),
        "numOfComments": ("num_comments", "INTEGER"), "selfText": ("selftext", "TEXT"), "score": ("score", "INTEGER"),
        "upvoteRatio": ("upvote_ratio", "INTEGER"), "insertionTimestamp": (None, "INT(11)")}
    POLL_SCHEMA = {
        "PostID": (None, "VARCHAR(255)"), "totalVoteCount": ("total_vote_count", "INTEGER"),
        "votingEndTimestamp": ("voting_end_timestamp", "INT(11)"),
        "insertionTimestamp": (None, "INT(11)")}
    POLL_OPTIONS_SCHEMA = {
        "PollOptionID": ("id", "VARCHAR(255)"), "text": ("text", "VARCHAR(255)"),
        "voteCount": ("vote_count", "INTEGER"), "insertionTimestamp": (None, "INT(11)")}

    return COMMENT_SCHEMA, POST_SCHEMA, POLL_SCHEMA, POLL_OPTIONS_SCHEMA


COMMENT_SCHEMA, POST_SCHEMA, POLL_SCHEMA, POLL_OPTIONS_SCHEMA = get_schemas()


def schema_value(instance, attr, overwrite_atrr=None, overwrite_variable_map=None):
    can_overwrite = overwrite_atrr is not None and overwrite_variable_map is not None
    if can_overwrite and overwrite_atrr in overwrite_variable_map:
        return overwrite_variable_map[overwrite_atrr]
    return value(instance, attr)


def schema_to_data(instance, schema, overwrite_variable_map=None):
    data = {k: [schema_value(instance, attr, k, overwrite_variable_map)] for k, (attr,_) in schema.items()}
    return data


def determine_time_filter(start_date):
    # Reddit lets you filter posts by certain time limits
    # Possible boundaries: limit posts to last hour, day, ..., all
    times = ["hour", "day", "week", "month", "year"]
    time_filters = {"day": timedelta(days=1), "hour": timedelta(hours=1),
                    "month": timedelta(days=31), "week": timedelta(weeks=1),
                    "year": timedelta(days=365)}
    curr = 0
    now = datetime.now()
    # Find the index of the first time limit in times that the start date isn't within:
    # e.g. if start date is within last 5 minutes, then its 0 ("hour")
    while curr < len(times) and start_date < (now - time_filters[times[curr]]) :
        curr += 1
    # if the start date isn't within a year, then you have to fetch all of the posts
    filter = "all" if curr >= len(times) else times[curr]
    return filter


# Schema:
# Comments: CommentID, body, permalink, author, edited, createdDate, score, upvoteRatio, insertionTimestamp, mentionedIslandersIDs
# Posts: PostID, author, authorFlairText, isPoll, isStickied, linkFlairText, linkUrl, isLocked, name, title, permalink, isEdited, createdDate, numOfComments, score, upvoteRatio, insertionTimestamp
# Polls: PollID, totalVoteCount, votingEndTimestamp, insertionTimestamp,
# PollOptions: PollOptionID, PollID, text, current_vote_count, insertionTimestamp, mentionedIslandersIDs
# Islanders: pulled from https://en.wikipedia.org/wiki/List_of_Love_Island_(2015_TV_series)_contestants
# IslanderMentionedComments: commentID, islanderID, matched, matchedSimiliarity,
# ID: seriesAgeName

# Comments: CommentID, body, permalink, author, edited, createdDate, score, insertionTimestamp
def to_comment_data(comment_instance, post_ID, now=datetime.now()):
    cI = comment_instance
    author = value(cI, "author")
    a_name, a_id = None, None
    if author:
        a_name, a_id = value(author, "name"), value(author, "id")
    c_overwrite = {"PostID": post_ID, "AuthorID": a_id, "AuthorName": a_name, "insertionTimestamp": round(now.timestamp())}
    data = schema_to_data(instance=cI, schema=COMMENT_SCHEMA, overwrite_variable_map=c_overwrite)
    return data


def get_comments_from_post_dataframe(submission_instance, now=datetime.now()):
    post_id = submission_instance.id
    df = None
    for top_level_comment in submission_instance.comments:
        if isinstance(top_level_comment, MoreComments):
            # MoreComments objects: represent the “load more comments”, and “continue this thread” links
            # Skip them
            continue
        data = to_comment_data(top_level_comment, post_id, now)
        df2 = pd.DataFrame(data)
        df = df2 if df is None else pd.concat([df, df2])
    return df


def to_poll_options_data(poll_option_instance, postID, now=datetime.now()):
    poI = poll_option_instance
    p_overwrite = {"PostID": postID, "insertionTimestamp": round(now.timestamp())}
    data = schema_to_data(instance=poI, schema=POLL_OPTIONS_SCHEMA, overwrite_variable_map=p_overwrite)
    return data

def to_poll_dataframes(poll_data_instance, postID, now=datetime.now()):
    poll_options_df = None
    for option in poll_data_instance.options:
        data = to_poll_options_data(option, postID, now)
        op_df = pd.DataFrame(data)
        poll_options_df = op_df if poll_options_df is None else pd.concat([poll_options_df, op_df])
    pdI = poll_data_instance
    p_overwrite = {"PostID": postID, "insertionTimestamp": round(now.timestamp())}
    poll_data = schema_to_data(instance=pdI, schema=POLL_SCHEMA, overwrite_variable_map=p_overwrite)
    poll_df = pd.DataFrame(poll_data)
    return poll_df, poll_options_df


# Posts: PostID, name, title, permalink, createdDate, author, authorFlairText, isPoll, isStickied, linkFlairText, linkUrl, isLocked, isEdited, numOfComments, score, upvoteRatio, insertionTimestamp
# Polls: PollID, totalVoteCount, votingEndTimestamp, insertionTimestamp,
# PollOptions: PollOptionID, PollID, text, current_vote_count, insertionTimestamp, mentionedIslandersIDs
def to_post_dataframes(submission_instance, now=datetime.now()):
    poll_df, poll_options_df = None, None
    sI = submission_instance
    author = sI.author
    a_name = author.name
    a_id = author.id

    is_poll = hasattr(sI, 'poll_data')
    postID = sI.id

    p_overwrite = {"PostID": postID, "AuthorID": a_id, "AuthorName": a_name, "insertionTimestamp": round(now.timestamp()),
                   "isPoll": [is_poll]}

    p_data = schema_to_data(instance=sI, schema=POST_SCHEMA, overwrite_variable_map=p_overwrite)
    if is_poll:
        poll_data = sI.poll_data
        poll_df, poll_options_df = to_poll_dataframes(poll_data, postID, now)
    post_df = pd.DataFrame(p_data)
    return post_df, poll_df, poll_options_df


def get_comments_and_posts_dataframes(subreddit_instance, start_date, now=datetime.now()):
    filter = determine_time_filter(start_date)
    posts = subreddit_instance.top(time_filter=filter, limit=None)
    res = []
    # Inefficient part: posts are fetched in upvote order (not date) so...
    # ...we have to manually check each post is created after start_date

    post_df, poll_df, poll_options_df, comment_df = None, None, None, None
    for post in posts:
        if start_date <= datetime.fromtimestamp(post.created_utc):
            c_df = get_comments_from_post_dataframe(post, now)
            post_df2, poll_df2, poll_options_df2 = to_post_dataframes(post, now)

            post_df = post_df2 if post_df is None else pd.concat([post_df, post_df2])
            poll_df = poll_df2 if poll_df is None else pd.concat([poll_df, poll_df2])
            poll_options_df = poll_options_df2 if poll_options_df is None else pd.concat([poll_options_df, poll_options_df2])
            comment_df = c_df if comment_df is None else pd.concat([comment_df, c_df])

    return post_df, poll_df, poll_options_df, comment_df

def search_subreddit_iterator(subreddit_instance, start_date, query="*", sort='relevance', syntax='lucene', now=datetime.now()):
    filt = determine_time_filter(start_date)
    print("Searching posts in last", filt)
    posts = subreddit_instance.search(query=query, sort=sort, syntax=syntax,
                                      time_filter=filt)
    return posts


def fetch_by_url(reddit_instance, post_url, include_comments=True):
    submission = reddit_instance.submission(url=post_url)
    return get_all_post_data(submission, include_comments=include_comments)

def fetch_by_id(reddit_instance, post_id, include_comments=True):
    submission = reddit_instance.submission(id=post_id)
    return get_all_post_data(submission, include_comments=include_comments)


def get_all_post_data(post, now=datetime.now(), include_comments=True):
    c_df = get_comments_from_post_dataframe(post, now) if include_comments else None
    post_df2, poll_df2, poll_options_df2 = to_post_dataframes(post, now)
    return c_df, post_df2, poll_df2, poll_options_df2


def search_subreddit(subreddit_instance, start_date, query="*", sort='relevance', syntax='lucene', now=datetime.now()):
    filter = determine_time_filter(start_date)
    print("Time filter", filter)
    posts = search_subreddit_iterator(subreddit_instance, start_date, query, sort, syntax, now)
    res = []
    # Inefficient part: posts are fetched in upvote order (not date) so...
    # ...we have to manually check each post is created after start_date
    post_df, poll_df, poll_options_df, comment_df = None, None, None, None
    for post in posts:
        if start_date <= datetime.fromtimestamp(post.created_utc):
            c_df = get_comments_from_post_dataframe(post, now)
            post_df2, poll_df2, poll_options_df2 = to_post_dataframes(post, now)

            post_df = post_df2 if post_df is None else pd.concat([post_df, post_df2])
            poll_df = poll_df2 if poll_df is None else pd.concat([poll_df, poll_df2])
            poll_options_df = poll_options_df2 if poll_options_df is None else pd.concat([poll_options_df, poll_options_df2])
            comment_df = c_df if comment_df is None else pd.concat([comment_df, c_df])
    return post_df, poll_df, poll_options_df, comment_df

