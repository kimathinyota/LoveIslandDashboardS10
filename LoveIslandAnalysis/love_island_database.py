import mysql.connector
from LoveIslandAnalysis.PostFetcher import get_schemas
COMMENT_SCHEMA, POST_SCHEMA, POLL_SCHEMA, POLL_OPTIONS_SCHEMA = get_schemas()
import datetime

# Purpose:
#   * Manage connections to db
#   * Functionality to CREATE, INSERT(or update), DELETE records for the following tables:
#       * Comments[CommentID, PostID, permalink, AuthorID, AuthorName, createdDate, body, score, insertionTimestamp]
#       * Posts[PostID, name, title, permalink, createdDate, AuthorID, AuthorName, authorFlairText, isPoll, isStickied,
#         linkFlairText, linkUrl, isLocked, isEdited, numOfComments, selfText, score, upvoteRatio, insertionTimestamp]
#       * Polls[PostID, totalVoteCount, votingEndTimestamp, insertionTimestamp]
#       * PollOptions[PollOptionID, PostID, text, voteCount, insertionTimestamp]
#   * Write code to fetch all of the LoveIsland discussion posts and load it into the database


class LIDatabaseManager:

    # query a list of queries in the form: [(sql, values),...]
    def query_all(self, queries, print_full=True):
        if len(queries) == 0:
            return None
        if not self.is_connected:
            print("Connection to db:" + "succeeded" if self.connect_to_db() else "failed")
        failed, succ = [], []
        for query in queries:
            sql, val = query
            try:
                if val is not None:
                    self.mycursor.execute(sql, val)
                else:
                    self.mycursor.execute(sql)
                succ += [query]
            except mysql.connector.errors.Error as e:
                if print_full:
                    print(e)
                failed += [query]

        self.mydb.commit()
        self.close()
        print("Executed:", str(len(succ)) + "/" + str(len(queries)))
        if print_full:
            print("Finished executing queries:", succ)
            print("Failed executing queries:", failed)

    def execute_query(self, sql):
        if not self.is_connected:
            print("Connecting to db: " + "succeeded" if self.connect_to_db() else "failed")
        self.mycursor.execute(sql)
        result = self.mycursor.fetchall()
        self.close()
        return result

    def close(self):
        if self.is_connected:
            self.is_connected = False
            self.mycursor.close()
            self.mydb.close()

    def create_table_sql(self, schema, table, primary_key):
        schema = schema.copy()
        schema[primary_key] = schema[primary_key] + ("NOT NULL",)
        sql = ",".join([" ".join([key] + list(value_tuples[1:])) for key, value_tuples in schema.items()])
        sql = "CREATE TABLE IF NOT EXISTS " + table + " (" + sql + ",PRIMARY KEY (" + primary_key + ")" ");"
        return sql

    # Insert (and update) into each table
    def replace_into_table_queries(self, schema, table, dataframe):
        keys = list(schema.keys())
        a = "(" + ",".join([key for key in keys]) + ")"
        v = "(" + ",".join(["%s" for _ in keys]) + ")"
        sql = "REPLACE INTO " + table + " " + a + " VALUES " + v
        ordered_rows = dataframe.loc[:, keys]
        queries = [(sql, tuple(row)) for row in ordered_rows.to_numpy()]
        return queries

    def replace_into_post_table(self, post_df, print_full=False):
        queries = self.replace_into_table_queries(POST_SCHEMA, "Post", post_df)
        self.query_all(queries, print_full)

    def replace_into_comments_table(self, comment_df, print_full=False):
        queries = self.replace_into_table_queries(COMMENT_SCHEMA, "Comment", comment_df)
        self.query_all(queries, print_full)

    def replace_into_poll_table(self, poll_df, print_full=False):
        queries = self.replace_into_table_queries(POLL_SCHEMA, "Poll", poll_df)
        self.query_all(queries, print_full)

    def replace_into_poll_option_table(self, poll_option_df, print_full=False):
        queries = self.replace_into_table_queries(POLL_OPTIONS_SCHEMA, "PollOption", poll_option_df)
        self.query_all(queries, print_full)

    def get_all_post_ids(self):
        sql = 'SELECT PostID FROM Post'
        res = [a[0] for a in self.execute_query(sql)]
        return res

    def get_last_post_insertion_time(self):
        sql = "SELECT insertionTimestamp FROM Post ORDER BY insertionTimestamp DESC LIMIT 1"
        res = self.execute_query(sql)[0][0]
        return res

    def connect_to_db(self):
        self.is_connected = True
        self.mydb = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            passwd="root",
            database="love_island_db",
            auth_plugin='mysql_native_password',
            port=3307
        )
        self.mycursor = self.mydb.cursor()
        return self.mydb.is_connected()

    # Creating Tables:
    def create_all_tables(self, print_full=False):
        queries = [(self.create_post_table_sql(), None), (self.create_comment_sql(), None),
                    (self.create_poll_table_sql(), None), (self.create_poll_option_sql(), None)]
        self.query_all(queries, print_full)

    def create_post_table_sql(self):
        return self.create_table_sql(POST_SCHEMA, "Post", "PostID")

    def create_poll_table_sql(self):
        return self.create_table_sql(POLL_SCHEMA, "Poll", "PostID")

    def create_poll_option_sql(self):
        return self.create_table_sql(POLL_OPTIONS_SCHEMA, "PollOption", "PollOptionID")

    def create_comment_sql(self):
        return self.create_table_sql(COMMENT_SCHEMA, "Comment", "CommentID")

    # Delete all Tables
    def drop_all_tables(self, print_full=False):
        tables = ["Post", "Comment", "Poll", "PollOption"]
        queries = [ (self.drop_table_sql(table), None) for table in tables]
        self.query_all(queries, print_full)

    def drop_table_sql(self, table):
        return "DROP TABLE " + table + ";"

    def fetch_all_posts(self, start_date, end_date):
        cols = list(POST_SCHEMA.keys())
        columns = ",".join(cols)
        ts1 = start_date.timestamp()
        ts2 = None if end_date is None else end_date.timestamp()
        sql = "SELECT " + columns + " FROM Post "
        sql += "WHERE createdDate > " + str(ts1)
        if ts2 is not None:
            sql += " AND createdDate < " + str(ts2)
        return self.execute_query(sql)
    
    def fetch_all_posts_from(self, start_date):
        return self.fetch_all_posts(start_date,)
    
    def fetch_posts(self):
        cols = list(POST_SCHEMA.keys())
        columns = ",".join(cols)
        sql = "SELECT " + columns + " FROM Post "
        return self.execute_query(sql), cols

    def fetch_all_comments(self):
        cols = list(COMMENT_SCHEMA.keys())
        columns = ",".join(cols)
        sql = "SELECT " + columns + " FROM Comment "
        return self.execute_query(sql), cols
    
    def fetch_all_comments_by_date(self, start_date, end_date):
        ts1 = start_date.timestamp()
        ts2 = None if end_date is None else end_date.timestamp()
        cols = list(COMMENT_SCHEMA.keys())
        columns = ",".join(cols)
        sql = "SELECT " + columns + " FROM Comment "
        sql += "WHERE createdDate > " + str(ts1) 
        if ts2 is not None:
            sql += " AND createdDate < " + str(ts2)
        return self.execute_query(sql), cols
    
    def fetch_all_comments_from(self, start_date):
        return self.fetch_all_comments_by_date(start_date, None)

    def fetch_all_comments_for_posts(self, post_ids):
        ids = "(" + ",".join(post_ids) + ")"
        cols = list(COMMENT_SCHEMA.keys())
        columns = ",".join(cols)
        sql = "SELECT " + columns + " FROM Comment "
        sql += "WHERE PostID IN " + ids
        return self.execute_query(sql)

    def __init__(self):
        self.is_connected = False
        self.mydb = None
        self.mycursor = None


