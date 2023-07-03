
# Use to fetch latest Islander information from updating Wikipedia table at:
# Source: "https://en.wikipedia.org/wiki/Love_Island_(2015_TV_series,_series_10)"

# requests for handling HTTP requests
import requests
# BeautifulSoup for organising data into a form that pandas can accept
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd



# Wikipedia keeps an updated table of all the islanders
# Below will fetch that data and transform it into a pandas dataframe
def fetch_islanders_from_wikipedia(link="https://en.wikipedia.org/wiki/Love_Island_(2015_TV_series,_series_10)"):
    # Data is in a table
    # Inspected table element and found its tag info...
    # ... <table_class="wikitable sortable jquery-tablesorter">

    # Lets make a GET request
    table_class = "wikitable sortable jquery-tablesorter"
    response = requests.get(link)
    # Response will (if success e.g. status code is 200) will be the HTML of the page at URL
    print(response.status_code)
    if response.status_code != 200:
        return None
    # Remove all the unneccessary html stuff. Just keep the table part
    soup = BeautifulSoup(response.text, 'html.parser')
    indiatable = soup.find('table', {'class': "wikitable"})
    # Use PANDAS to turn fetched HTML table (as string) into a dataframe
    contestants = pd.read_html(str(indiatable))
    contestants = pd.DataFrame(contestants[0])
    return contestants

# This method may need to be updated depending on wikipedia edit
# https://en.wikipedia.org/wiki/Love_Island_(2015_TV_series,_series_10)
def clean_islanders_dataframe_from_wikipedia(contestants):
    # Remove nulls and only keep necessary fields
    contestants_clean = contestants.loc[pd.notna(contestants.Islander),
                                        ['Islander', 'Age', 'Hometown', 'Entered', 'Status']].copy()
    # Sort out types
    contestants_clean[['Islander', 'Hometown', 'Entered', 'Status']] = contestants_clean[
        ['Islander', 'Hometown', 'Entered', 'Status']].astype(str)
    contestants_clean['Age'] = contestants_clean['Age'].astype(int)

    # As it stands: Entered is in the form: Day #
    # Need to extract just the numeric part
    contestants_clean["ShowEntryDay"] = contestants_clean.Entered.str.extract(r'([0-9]+)').astype('Int64')

    # The Molly Exemption: molly marsh renters as a casa amor contestant on day 27, but it is scraped from wikipedia as 271 
    contestants_clean["ShowEntryDay"].loc[contestants_clean.ShowEntryDay>200] = contestants_clean["ShowEntryDay"].loc[contestants_clean.ShowEntryDay>200].astype(str).str[:2].astype('Int64')


    # As it stands: Status is in the form: Participating | Dumped (Day #)
    # Need to extract just the numeric part (but allow for nulls)
    # 'Int64' allows for nulls
    contestants_clean['ShowLeaveDay'] = contestants_clean.Status.str.extract(r'([0-9]+)').astype('Int64')

    # Below field for storing explicitely whether an islander has been dumped
    # a little uneccessary, but better for information representation.
    dumped_mask = pd.notna(contestants_clean.ShowLeaveDay)
    contestants_clean["OnLeaveStatus"] = pd.NA
    contestants_clean.OnLeaveStatus[dumped_mask] = contestants_clean.Status[dumped_mask].str.split(' ').str.get(0)


    # storing when islanders entered the mainVilla
    # Casa Amor happens from day 26 to day 30, so any islanders who arrive between these days are Casa Amor islanders
    casa_mask = (contestants_clean.ShowEntryDay > 25) & (contestants_clean.ShowEntryDay < 30)
    contestants_clean["MainVillaEntryDay"] = contestants_clean["ShowEntryDay"].copy()
    contestants_clean["MainVillaEntryDay"].loc[casa_mask] = pd.NA


    # Recoupling date happens on day 32
    # The casa contestants not already dumped by day 32 will enter the Main villa on day 32
    not_dumped_casa = casa_mask & pd.isna(contestants_clean.MainVillaEntryDay)
    #contestants_clean["MainVillaEntryDay"].loc[not_dumped_casa] = 32

    # Entered and Status fields are now redundant
    contestants_clean.drop(columns=['Entered', 'Status'], inplace=True)

    return contestants_clean


def fetch_and_clean_islanders_from_wikipedia(link="https://en.wikipedia.org/wiki/Love_Island_(2015_TV_series,_series_10)",
                                             cleaning_function=lambda df: clean_islanders_dataframe_from_wikipedia(df)):
    return cleaning_function(fetch_islanders_from_wikipedia(link))

