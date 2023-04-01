import praw

def authenticate():
    secret = '8VSVPtHnLg0COf7ifgHVn09g50GjMw'
    client_id = 'mPdfG3HufOS03lbx4cYmgw'
    user_agent = 'LoveIslandScraping'
    reddit = praw.Reddit(client_id=client_id, client_secret=secret, user_agent=user_agent)
    return reddit

def authenticate_love_island():
    sub_name = "LoveIslandTV"
    reddit = authenticate()
    return reddit.subreddit(sub_name)
