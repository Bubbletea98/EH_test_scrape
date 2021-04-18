#!/usr/bin/env python
# coding: utf-8

# In[1]:


import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import time
import datetime


# In[2]:


list_name=['AIGInsurance',
'AmericanInternationalGroup',
'Atradius',
'Coface',
'ExportDevelopmentCanada',
'EDC Insurance',
'EulerHermes']


# In[3]:


tweets_list= {'AIGInsurance':[],
'AmericanInternationalGroup':[],
'Atradius':[],
'Coface':[],
'ExportDevelopmentCanada':[],
'EDC Insurance':[],
'EulerHermes':[]}

for name in list_name:# Setting variables to be used below
    maxTweets =100000
    # Creating list to append tweet data to
    
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('{} since:2019-06-01'.format(name)).get_items()):
        if i%1000==0:
            print("Finished {}: ".format(name),i)
            time.sleep(1)
        if i>maxTweets:
            break
        if tweet.mentionedUsers!=None:
            mention_tweet=",".join([i.username for i in tweet.mentionedUsers])
        else:
            mention_tweet=np.nan
        tweets_list[name].append([tweet.date, tweet.id, tweet.content, tweet.user.username,mention_tweet,tweet.user.followersCount,tweet.user.listedCount,tweet.user.location])
        


# In[4]:


tweets_df1= pd.DataFrame(tweets_list['AIGInsurance'], columns=['Datetime', 'Tweet Id', 'Text', 'Username','user_mention','followersCount','listedCount','location'])
tweets_df2= pd.DataFrame(tweets_list['AmericanInternationalGroup'], columns=['Datetime', 'Tweet Id', 'Text', 'Username','user_mention','followersCount','listedCount','location'])
tweets_df4= pd.DataFrame(tweets_list['Coface'], columns=['Datetime', 'Tweet Id', 'Text', 'Username','user_mention','followersCount','listedCount','location'])
tweets_df5= pd.DataFrame(tweets_list['ExportDevelopmentCanada'], columns=['Datetime', 'Tweet Id', 'Text', 'Username','user_mention','followersCount','listedCount','location'])
tweets_df3= pd.DataFrame(tweets_list['Atradius'], columns=['Datetime', 'Tweet Id', 'Text', 'Username','user_mention','followersCount','listedCount','location'])
tweets_df6= pd.DataFrame(tweets_list['EDC Insurance'], columns=['Datetime', 'Tweet Id', 'Text', 'Username','user_mention','followersCount','listedCount','location'])
tweets_df7= pd.DataFrame(tweets_list['EulerHermes'], columns=['Datetime', 'Tweet Id', 'Text', 'Username','user_mention','followersCount','listedCount','location'])


# In[5]:


tweets_df1.to_csv('AIGInsurance.csv')
tweets_df2.to_csv('AmericanInternationalGroup.csv')
tweets_df4.to_csv('Coface.csv')
tweets_df5.to_csv('ExportDevelopmentCanada.csv')
tweets_df3.to_csv('Atradius.csv')
tweets_df6.to_csv('EDC Insurance.csv')
tweets_df7.to_csv('EulerHermes.csv')
tweets_df1=pd.read_csv('AIGInsurance.csv')
tweets_df2=pd.read_csv('AmericanInternationalGroup.csv')
tweets_df4=pd.read_csv('Coface.csv')
tweets_df5=pd.read_csv('ExportDevelopmentCanada.csv')
tweets_df3=pd.read_csv('Atradius.csv')
tweets_df6=pd.read_csv('EDC Insurance.csv')
tweets_df7=pd.read_csv('EulerHermes.csv')

print('================================Finished Scraping===================================')

# ### Topic Modeling

# In[7]:


# libraries
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import emoji
import plotly.graph_objects as go
import gif
import plotly.offline as pyo
import matplotlib.pyplot as plt


# In[8]:


tweets_df_5_6= pd.concat([tweets_df5,tweets_df6], axis=0)
tweets_df_1_2= pd.concat([tweets_df1,tweets_df2], axis=0)


# In[9]:


list_save_csv=[tweets_df_1_2,tweets_df3,tweets_df4,tweets_df_5_6,tweets_df7]


# In[10]:


list_name=['AIGInsurance',
'Atradius',
'Coface',
'ExportDevelopmentCanada',
'EulerHermes']
count=0
for i in list_save_csv:
    i.to_csv("{}_combine.csv".format(list_name[count]))
    count=count+1


# In[11]:


df_list=[tweets_df_1_2,tweets_df3,tweets_df4,tweets_df_5_6,tweets_df7]


# In[12]:


# Df clean up
my_stopwords = nltk.corpus.stopwords.words('english') +                ['like','would','see','la','de', 'que',"'s", "’s",'e','da']+nltk.corpus.stopwords.words('french')
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@’'

def clean_up(text):
    text = text.lower()
    # remove replied msg
    if 'rt @' in text:
        if len(text.split(": ")) < 2:
            text = text.split(": ")[0]
        else:
            text = text.split(": ")[1]
    # remove the unnecessay URLs
    if "https" in text:
        text = text.split("https")[0]
    if "@" in text:
        text = text.split(" ")
        # remove the strings with symbol @
        text = [j for j in text if '@' not in j]
        text = ' '.join(text)
    # remove hashtag symbols
    text = text.replace("#", "")
    text = text.replace("'s'", "")
    text = text.replace("’s'", "")
    text = re.sub('['+my_punctuation + ']+', ' ', text)# strip punctuation
    text = re.sub('\s+', ' ', text)#remove double spacing
    text = re.sub('([0-9]+)', '', text) # remove numbers
    text = emoji.get_emoji_regexp().sub(u'', text) # remove emoji
    text = ' '.join([word for word in text.split(' ') if word not in my_stopwords]) # remove stopwords
    return text


# In[13]:


for df in df_list:
    df['clean_tweet'] = df['Text'].map(clean_up)
    print('Finished 1 dataset')


# In[14]:


def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


# In[15]:


topic_list=[]
i=1
for df in df_list:

    vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')

    tf = vectorizer.fit_transform(df['clean_tweet']).toarray()

    tf_feature_names = vectorizer.get_feature_names()

    number_of_topics = 5

    model = LatentDirichletAllocation(n_components=number_of_topics, random_state=101)

    model.fit(tf)
    no_top_words = 10

    topcis = display_topics(model, tf_feature_names, no_top_words)
    print('Finished: ',i)
    i=i+1
    topic_list.append(topcis)
    


# In[16]:


list_name=['AIGInsurance',
'Atradius',
'Coface',
'ExportDevelopmentCanada',
'EulerHermes']
count=0
for i in topic_list:
    i.to_csv("{}topic_modling.csv".format(list_name[count]))
    count=count+1

print('====================================Finished Topic Modelling=======================================')


for df in df_list:
    df['location'].value_counts().head(25)
    df.location.replace({
        'SLC UT': 'United States',
        'London': 'United Kingdom',
        'London, England':'United Kingdom',
        'London, UK':'United Kingdom',
        'UK': 'United Kingdom',
        'New York, NY':'United States',
        'Global':'Worldwide',
        'Tucson, AZ':'United States',
        'USA':'United States',
        'Washington, DC':'United States',
        'San isidro, Peru': 'Peru',
        'San Francisco, CA':'Canada',
        'Hong Kong':'China',
        'Offices worldwide': 'Worldwide',

        'Boston, MA': 'United States',
        }, inplace=True)

    df.loc[~df.location.isin(['United States','United Kingdom','Worldwide','Peru','Canada','China']), 'location'] = 'Other'



count=1
area_topic_list=[]
for df in df_list[:4]:
    areas = list(df['location'].unique())
    print(areas)
    topics_areas = pd.DataFrame()
    for i in areas:
        print(i)
        if len(df[df['location']==i])>16:
            
            vectorizer = CountVectorizer(max_df=0.9, min_df=10, token_pattern='\w+|\$[\d\.]+|\S+')

            tf = vectorizer.fit_transform(df[df['location']==i]['clean_tweet']).toarray()

            tf_feature_names = vectorizer.get_feature_names()

            number_of_topics2 = 1
            no_top_words2 = 3

            model2 = LatentDirichletAllocation(n_components=number_of_topics2, random_state=101)

            model2.fit(tf)

            topics2 = display_topics(model2, tf_feature_names, no_top_words2)
            topics2['area']=i


            topics_areas = topics_areas.append(topics2)
    print('Finished: ',count)
    count=count+1
    area_topic_list.append(topics_areas)


# In[195]:


areas = list(df_list[4]['location'].unique())

topics_areas = pd.DataFrame()
for i in areas:
    print(i)
    if len(df_list[4][df_list[4]['location']==i])>16:

        vectorizer = CountVectorizer(max_df=0.9, min_df=5, token_pattern='\w+|\$[\d\.]+|\S+')

        tf = vectorizer.fit_transform(df_list[4][df_list[4]['location']==i]['clean_tweet']).toarray()

        tf_feature_names = vectorizer.get_feature_names()

        number_of_topics2 = 1
        no_top_words2 = 3

        model2 = LatentDirichletAllocation(n_components=number_of_topics2, random_state=101)

        model2.fit(tf)

        topics2 = display_topics(model2, tf_feature_names, no_top_words2)
        topics2['area']=i


        topics_areas = topics_areas.append(topics2)
print('Finished: ',count)
count=count+1
area_topic_list.append(topics_areas)


# In[227]:


final_topic_area=[]
for df in area_topic_list:
    df=df.groupby('area')['Topic 0 words'].agg(list)
    df=df.reset_index()
    final_topic_area.append(df)


# In[231]:


final_topic_area[0].to_csv('AIG_Areas_topic.csv')
final_topic_area[1].to_csv('Atridius_Areas_topic.csv')
final_topic_area[2].to_csv('coface_Areas_topic.csv')
final_topic_area[3].to_csv('EDC_Areas_topic.csv')
final_topic_area[4].to_csv('EH_Areas_topic.csv')

print('============================================Finished Topic =================================')

analyzer = SentimentIntensityAnalyzer()
list_plot=['AIGInsurance',
'Atradius',
'Coface',
'EDC',
'EulerHermes']


for df in df_list:
    VADER_score = []
    for i in df['clean_tweet']:
        va = analyzer.polarity_scores(i)['compound']
        VADER_score += [va]
    df['sentiment_score'] = VADER_score

print('===========================================Calculated sentiment score==============================')
# In[355]:



sa_loc = df_list[1].groupby('location')['sentiment_score'].mean()
sa_loc=sa_loc.reset_index()
sa_loc=sa_loc.sort_values('location')


# In[356]:


location_list=sa_loc['location']


# In[357]:


list(location_list)


# In[358]:


list_plot


# In[396]:


# sentiment analysis for each date
i=0
plt.figure(figsize=(20, 7))
list_sa_loc=[]
for df in df_list:
    sa_loc = df.groupby('location')['sentiment_score'].mean()
    sa_loc=sa_loc.reset_index()
    sa_loc=sa_loc.sort_values('location')

    list_sa_loc.append(sa_loc)


#     sa_loc = sa_loc.to_frame().reset_index()


# In[377]:


from functools import reduce
df_plot_area_sa = reduce(lambda df1,df2: pd.merge(df1,df2,on='location',how='outer'), list_sa_loc)


# In[379]:


df_plot_area_sa.columns=['location','AIGInsurance', 'Atradius', 'Coface', 'EDC', 'EulerHermes']


# In[384]:


b, c = df_plot_area_sa.iloc[0], df_plot_area_sa.iloc[1]

temp = df_plot_area_sa.iloc[0].copy()
df_plot_area_sa.iloc[0] = c
df_plot_area_sa.iloc[1] = temp



df_plot_area_sa.to_csv('sentiment_countrywide.csv')
print('================================finished sentiment by countries================================')




# In[416]:






df=df_list[1]
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.info()
df['Date'] = df['Datetime'].dt.date 
df['Date'] = pd.to_datetime(df['Date']).dt.to_period('m')
dates = df.Date.unique()
topics_dates2 = pd.DataFrame()

for i in dates:
    vectorizer = CountVectorizer(max_df=0.9, min_df=15, token_pattern='\w+|\$[\d\.]+|\S+')

    tf = vectorizer.fit_transform(df[df['Date']==i]['clean_tweet']).toarray()

    tf_feature_names = vectorizer.get_feature_names()

    number_of_topics2 = 1
    no_top_words2 = 1

    model2 = LatentDirichletAllocation(n_components=number_of_topics2, random_state=101)

    model2.fit(tf)


    topics2 = display_topics(model2, tf_feature_names, no_top_words2)
    topics2['Date']=i

    topics_dates2 = topics_dates.append(topics2)



topics_dates2.to_csv('Atradius_topic_modeling_across_time.csv')

topics_dates1.to_csv('AIG_topic_modeling_across_time.csv')

topics_dates.to_csv('EH_topic_modeling_across_time.csv')

print('=============================Calculated topic modeling across time=======================================')
# In[500]:


count_month={}
list_count=['AIGInsurance',
'Atradius',
'Coface',
'EDC',
'EulerHermes']
count=0
df_list[0]['Datetime'] = pd.to_datetime(df_list[0]['Datetime'])
df_list[0]['Date'] = df_list[0]['Datetime'].dt.date 
df_list[0]['Date'] = pd.to_datetime(df_list[0]['Date']).dt.to_period('m')
df_list[0]=df_list[0].sort_values(by='Date',ascending=False)
date=  df_list[0]['Date'].unique()
for df in df_list:
    
    list_sub=[]
    for i in date:
        
        list_sub.append(len(df[df['Date']==i]))
    
    count_month[list_count[count]]=list_sub
    count=count+1 



df_count=pd.DataFrame(count_month)
df_count['Date']=df['Date'].unique()



df_count=df_count.sort_values(by='Date')


df_count.to_csv('tweet_count_across_date.csv')
print('=========================================================tweet count across date finished==================')


date_x=df_count['Date'].astype(str)


# In[511]:




# In[441]:






sa_date=[]
for df in df_list:
    sa_loc = df.groupby('Date')['sentiment_score'].mean()
    sa_loc=sa_loc.reset_index()
    sa_date.append(sa_loc)


# In[453]:


date_across_sent=sa_date[0]['Date']


# In[514]:


from functools import reduce
df_plot_time_sa = reduce(lambda df1,df2: pd.merge(df1,df2,on='Date',how='outer'), sa_date)


# In[517]:


df_plot_time_sa.columns=['Date','AIGInsurance',
'Atradius',
'Coface',
'EDC',
'EulerHermes']


# In[519]:


df_plot_time_sa.to_csv('sentiment_across_date.csv')

print('=====================================Sentiment score across date finished===============================')
# In[464]:



# In[469]:


date_across_sent=date_across_sent.astype(str)




# Maybe you needed to display plot in jupyter notebook


# ### Lift Value

# In[132]:


text_col=[]
for df in df_list:
    text_col= text_col+df["clean_tweet"].tolist()
    


# In[133]:


index_lkup={'trade credit insurance':['trade','credit','insurance'],'Export Import':['import','export'],'Other Insurance organizations':['bank','amp','icba'],'AIGInsurance':['AIGInsurance','aig','AIG','AmericanInternationalGroup'],'Atradius':['Atradius','atradius'],'Coface':['coface','Coface'],'EDC':['edc','EDC','EDCInsurance','ExportDevelopmentCanada'],'EulerHermes':['eulerhermes','euler','hermes','EH','EulerHermes','Euler','Hermes']}


# In[134]:


from nltk.corpus import stopwords       ## nltk package built-in stopword file
from nltk.stem import PorterStemmer     ## for Stemming (Based on String -Faster)
import preprocessor as p                ## for URLs/Hashtags/Mentions/Reserved Words (RT, FAV)/Emojis/Smileys removal
import re                               ## for regular expression operations

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import LancasterStemmer

from bs4 import BeautifulSoup
import requests
import pandas as pd
from nltk.stem import WordNetLemmatizer
lancaster=LancasterStemmer()
wordnet_lemma = WordNetLemmatizer()
stop_words = stopwords.words("english")+stopwords.words("french")
def my_tokenizer(s):
    """
    input: text string
    output: tokens

    processing:
        1. pre-processing text 
            e.g.    lower-casing, 
                    removing unnecessary poeces (space, punctuation - exept "'")
                    removing stopwords
        2. brand-model lookup and replace
    """
    s = s.lower()   # lower every word letter (later for review)
    s = p.clean(s)  # for URLs/Hashtags/Mentions/Reserved Words (RT, FAV)/Emojis/Smileys removal
    s = lancaster.stem(s) 
    s = re.sub(r"[^\w\d'\s]+",'',s)
    tokens = nltk.tokenize.word_tokenize(s)                     # works better than string.split
    tokens =[wordnet_lemma.lemmatize(t) for t in tokens]
    #tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # return words into their base form to reduce the vocab size
    #tokens = [stemmer.stem(t) for t in tokens]                  # return words into their base form to reduce the vocab size
    tokens = [t for t in tokens if t not in stop_words]         # remove stopwords
    #toekns = [index_lkup[t] for t          in tokens[t]     if t in index_lkup else t]
    test_tokenized_brand  = []
    for t in tokens:
        for key in index_lkup.keys():
            if t in index_lkup[key]: # if not repeated then keep and assign indices
                test_tokenized_brand.append(key)
            
    return test_tokenized_brand


# In[135]:


get_ipython().run_cell_magic('time', '', 'text_tokenized = []\n\nfor i in range(len(text_col)):\n    tokens = my_tokenizer(str(text_col[i]))\n    text_tokenized.append(tokens)')




import numpy as np
### Create columns for top 10 brands 
### If message contain the brand, then brand_i = 1, otherwise 0
lift_bin = pd.DataFrame(0, index=np.arange(len(text_tokenized)), columns=index_lkup.keys())
text_tokenized
for i in range(len(text_tokenized)):
    for j in text_tokenized[i]:
        if j in index_lkup.keys():
            lift_bin.iloc[i,lift_bin.columns.get_loc(j)]=1
            


# In[188]:


### loading pkg for lifting
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[189]:


frequent_itemsets = apriori(lift_bin, 
                            min_support=0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001, 
                            use_colnames=True)


# In[190]:


frequent_itemsets


# In[191]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.to_csv("rules000000000000001.csv",index=False)


# In[192]:


lift=rules[["antecedents","consequents","lift"]]




pair_lift= lift
pair_lift["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[:]).astype("unicode")
pair_lift["consequents"] = rules["consequents"].apply(lambda x: list(x)[:]).astype("unicode")


# In[195]:


pair_lift.to_csv("Lift_value_overall.csv")


# In[196]:


pair_lift[pair_lift['consequents']=="['AIGInsurance']"]


# In[197]:


pair_lift=pair_lift[:16]
pair_lift['antecedents']=[i.replace('[','').replace(']','').replace("'",'') for i in pair_lift['antecedents']]
pair_lift['consequents']=[i.replace('[','').replace(']','').replace("'",'')for i in pair_lift['consequents']]


# In[198]:


pair_lift


# In[199]:


pivot=pair_lift.pivot_table('lift',index='antecedents',columns='consequents')
#pivot=pivot.replace(np.nan,0)

pivot_final=pivot.replace(np.nan,np.inf)


# In[200]:


pivot_final.to_csv("pivot_pair.csv",index=True)


# In[201]:


pivot_scaled=1/pivot


pivot_scaled=pivot_scaled.replace(np.nan,0)
pivot_scaled.to_csv('lift_value_competitors.csv')


# In[203]:


from sklearn.manifold import MDS
import matplotlib.pyplot as plt


# In[204]:
mds = MDS(2,random_state=0,dissimilarity='precomputed')
lift_2D = mds.fit_transform(pivot_scaled)

plt.rcParams['figure.figsize'] = [8, 8]
plt.rc('font', size=17, weight='bold')

x_list = []
y_list = []
label_list = []
fig = plt.figure(figsize=(12, 10))
for i in np.unique(pivot_scaled.columns):
    subset = lift_2D[pivot_scaled.columns == i]

    x = [row[0] for row in subset]
    y = [row[1] for row in subset]

    plt.scatter(x, y, s=300)
for label, x, y in zip(pivot_scaled.columns, lift_2D[:, 0], lift_2D[:, 1]):
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(-20, 20),
        textcoords='offset points'
    )
    label_list.append(label)
    x_list.append(x)
    y_list.append(y)
plt.savefig('mdsplot_copetitors.png', dpi=300)
plt.show()




mdsplot=pd.DataFrame(x_list,columns=['x_coordinates'])
mdsplot['y_coordinates']=y_list
mdsplot['label']=label_list
mdsplot.to_csv('mdsplot_data.csv')


print('=========================================Finshed ALL===================================================')
# In[ ]:




