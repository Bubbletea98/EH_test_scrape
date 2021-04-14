import pandas as pd
import snscrape.modules.twitter as sntwitter

# Below are two ways of scraping using the Python Wrapper.
# Comment or uncomment as you need. If you currently run the script as is it will scrape both queries
# then output two different csv files.


# Query by text search
# Setting variables to be used below
maxTweets = 1

# Creating list to append tweet data to
tweets_list2 = []

# Using TwitterSearchScraper to scrape data and append tweets to list


#since:2021-01-01

for i,tweet in enumerate(sntwitter.TwitterSearchScraper('EDC until:2021-03-31').get_items()):
    if i>maxTweets: # This sets the max tweets number
        break
    if i%1000==0:
        print('EDC')
        print('Finished: ',i)
    m_users = []
    if tweet.mentionedUsers != None: # If there are mentioned users, write them in the list
        for i in range(len(tweet.mentionedUsers)):
            m_users.append(tweet.mentionedUsers[i].username)

    tweets_list2.append([tweet.id, # Tweet ID for indexing purposes
                         tweet.date, # Date the Tweet was posted
                         tweet.content, # Content of the Tweet
                         tweet.user.username, # Username of the author of the Tweet
                         tweet.user.displayname, # Display of the author of the Tweet
                         tweet.user.verified, # Verified status of Tweet author
                         tweet.user.location, # Location of the Tweet author
                         tweet.retweetCount, # Count of retweets of the Tweet at the moment of the scrape
                         m_users]) # Mentioned users in the Tweet

# Creating a dataframe from the tweets list above


tweets_df2 = pd.DataFrame(tweets_list2, columns=['Tweet_id',
                                                 'Datetime',
                                                 'Tweet_content',
                                                 'Username',
                                                 'Display_name',
                                                 'Verified',
                                                 'User_location',
                                                 'Retweet_count',
                                                 'Mentioned_users'])

# Display first 5 entries from dataframe
tweets_df2.head()

# Export dataframe into a CSV
tweets_df2.to_csv('EH-RawTweets_EDC_until2021-03-31.csv', sep=',', index=False)



# SNSCRAPE - covid1
# Imports
import pandas as pd
import snscrape.modules.twitter as sntwitter

# Below are two ways of scraping using the Python Wrapper.
# Comment or uncomment as you need. If you currently run the script as is it will scrape both queries
# then output two different csv files.

# Query by text search
# Setting variables to be used below
maxTweets = 10000

# Creating list to append tweet data to
tweets_list2 = []

# Using TwitterSearchScraper to scrape data and append tweets to list


#since:2021-01-01

for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Allianz until:2021-03-31').get_items()):
    if i>maxTweets: # This sets the max tweets number
        break
    if i%1000==0:
        print('Allianz')
        print('Finished: ',i)
    m_users = []
    if tweet.mentionedUsers != None: # If there are mentioned users, write them in the list
        for i in range(len(tweet.mentionedUsers)):
            m_users.append(tweet.mentionedUsers[i].username)

    tweets_list2.append([tweet.id, # Tweet ID for indexing purposes
                         tweet.date, # Date the Tweet was posted
                         tweet.content, # Content of the Tweet
                         tweet.user.username, # Username of the author of the Tweet
                         tweet.user.displayname, # Display of the author of the Tweet
                         tweet.user.verified, # Verified status of Tweet author
                         tweet.user.location, # Location of the Tweet author
                         tweet.retweetCount, # Count of retweets of the Tweet at the moment of the scrape
                         m_users]) # Mentioned users in the Tweet

# Creating a dataframe from the tweets list above


tweets_df2 = pd.DataFrame(tweets_list2, columns=['Tweet_id',
                                                 'Datetime',
                                                 'Tweet_content',
                                                 'Username',
                                                 'Display_name',
                                                 'Verified',
                                                 'User_location',
                                                 'Retweet_count',
                                                 'Mentioned_users'])

# Display first 5 entries from dataframe
tweets_df2.head()

# Export dataframe into a CSV
tweets_df2.to_csv('EH-RawTweets_Allianz_until2021-03-31TEST6.csv', sep=',', index=False)
