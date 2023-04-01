import praw

def authenticate(secret, client_id, user_agent):
    reddit = praw.Reddit(client_id=client_id, client_secret=secret, user_agent=user_agent)
    return reddit

def authenticate_love_island(secret, client_id, user_agent):
    sub_name = "LoveIslandTV"
    reddit = authenticate(secret, client_id, user_agent)
    return reddit.subreddit(sub_name)
