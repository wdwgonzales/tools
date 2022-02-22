
import pickle, os
import logging, sys, warnings, argparse
import tkinter as tk
warnings.filterwarnings("ignore")
from logging.handlers import TimedRotatingFileHandler
FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "my_app.log"

def get_console_handler():
   console_handler = logging.StreamHandler(sys.stdout)
   console_handler.setFormatter(FORMATTER)
   return console_handler
def get_file_handler():
   file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
   file_handler.setFormatter(FORMATTER)
   return file_handler
def get_logger(logger_name):
   logger = logging.getLogger(logger_name)
   logger.setLevel(logging.DEBUG) # better to have too much log than not enough
   logger.addHandler(get_console_handler())
   logger.addHandler(get_file_handler())
   # with this pattern, it's rarely necessary to propagate the error up to parent
   logger.propagate = False
   return logger

my_logger = get_logger(__name__)

my_logger.info('Initiating.')


from langdetect import detect, detect_langs
import pandas as pd
import snscrape.modules.twitter as sntwitter
import preprocessor as p
import itertools, re
from nnsplit import NNSplit
from alive_progress import alive_bar
import json
from geopy.geocoders import Nominatim
# Initialize Nominatim API
geolocator = Nominatim(user_agent="geoapiExercises")

import spacy
from spacy.lang.en.examples import sentences



working = os.getcwd()
save = working + '/json'
pic = working + '/pics'
resultsfolder = working + '/results'

from m3inference import M3Twitter
import pprint
import argparse
import sys, ast
import configparser
import logging
import subprocess, sys


sample = '''Sandig sa pinal kag validated nga fire incident report sang Iloilo City Social Welfare and Development Office napetsahan Enero 25, 2012 nga ining mga pamilya nga biktima sang sunog ang sa karon nagatener sa Manuel Luis Quezon Elementary School, Rizal Elementary School, A. Bonifacio Elementary School, Malipayon Day Care Center, Tanza Timawa Day Care Center, kag Tanza Timawa Barangay Hall. Ginsiling ni Sarah Palelo sang CSWDO City Proper nga ang 74 ka iban pa nga apektado nga pamilya ang nagatener sa ila mga himata nga nagapuyo sa mga apektado nga barangay ukon kaingod nga kabaranggyan. Ang sunog nagluntad pasado alas singko sang hapon sang Domingo, Enero 22 nga ginpalayag nagsugod sa Barangay Tanza Timawa Zone II kag naglapta sa Tanza Esperanza kag Malipayon kag nagtupok sang 273 ka mga balay kag naghalit sang 23 ka iban pa. Kapin naman sa 1,830 ka mga persona ang naapektuhan suno sa report sang CSWDO. Suno naman sa Iloilo City Public Information Office ang mga evacuee ang ginahatagan sang bulig gikan sa mga donasyon halin sa gobyerno kag pribado nga mga indibidwal kag grupo. Samtang padayon naman nga naga-apelar si Mayor Jed Patrick Mabilog sang tayuyon nga pagpanikasog agud buligan ang mga biktima sang sunog. Ginarekomendar sang CSWDO nga ang mga biktima sang sunog ang pagahatagan sang food assistance sa duha ka semana subong man support services kasubong sang pinansyal para sa temporary shelter assistance kag practical skills training cum livelihood assistance para sa ila rehabilitasyon kag pagpaayo.'''

def custom_split(sepr_list, str_to_split):
    # create regular expression dynamically
    regular_exp = '|'.join(map(re.escape, sepr_list))
    return re.split(regular_exp, str_to_split)



def eng_sentencize(string):
    splitter = NNSplit.load("en")
    splits = splitter.split([string])[0]
    listofsplitted = [str(e) for e in splits]
    return listofsplitted

def sentencize (string, language):
    if language == "en":
        try:
            return eng_sentencize(string)
        except:
            return 'NA'
    else:

        separators = ". ", "! "
        x = string.replace("!", "!! ")
        x = x.replace(". ", ".. ")
        return custom_split(separators, str(x))

def detectlg(str):
    try:
        x = detect(str)
        if x == "tl":
            x == "fil"
            return x
    except:
        return 'NA'


#def nn_detectlg(str):
#    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,max_num_bytes=1000)
#    result = detector.FindLanguage(text=str)
#    return result.language

#def nn_detectlg_prob(str):
#    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,max_num_bytes=1000)
#    result = detector.FindLanguage(text=str)
#    return float(result.probability)

#def nn_detectlg_reliability(str):
#    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,max_num_bytes=1000)
#    result = detector.FindLanguage(text=str)
#    return result.is_reliable

#def nn_detectlg_top3(str):
#    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
#    results = detector.FindTopNMostFreqLangs(text=str, num_langs=2)
#    listt = []
#    for result in results:
#        x = [result.language, result.probability]
#        listt.append(x)
#    return listt


#df_coord['user_location'] =  df_coord['user'].apply(lambda x: x['location'])

def extract(mode, inputt, number):

    column_names = ['date', 'url','id', 'user', 'lang', 'conversationId', 'likeCount', 'mentionedUsers','content']

    if mode == 'term':
        scraped_tweets = sntwitter.TwitterSearchScraper(inputt).get_items()
        sliced_scraped_tweets = itertools.islice(scraped_tweets, number)
        df = pd.DataFrame(sliced_scraped_tweets)[column_names]
        return df

    elif mode == 'user':
        #INPUT  MUST BE A LIST HERE

        df = pd.DataFrame(columns = column_names)

        for n, k in enumerate(inputt):
            scraped_tweets = sntwitter.TwitterSearchScraper('from:{}'.format(inputt[n])).get_items()
            sliced_scraped_tweets = itertools.islice(scraped_tweets, number)
            premerge = pd.DataFrame(sliced_scraped_tweets)[column_names]

            frames = [df, premerge]
            df = pd.concat(frames)
        return df

    elif mode == 'location':
        scraped_tweets = sntwitter.TwitterSearchScraper(f'near:"{inputt}" within:5km' ).get_items()
        sliced_scraped_tweets = itertools.islice(scraped_tweets, number)
        df = pd.DataFrame(sliced_scraped_tweets)[column_names]
        return df

    elif mode == 'geolocation':
        #INPUT must be X coordinate, Y coordinate, Zkm 'loc = '34.052235, -118.243683, 10km'
        #scraped_tweets = sntwitter.TwitterSearchScraper('geocode:"{}"'.format(inputt)).get_items()
        scraped_tweets = sntwitter.TwitterSearchScraper('geocode:"{}"'.format(inputt)).get_items()
        sliced_scraped_tweets = itertools.islice(scraped_tweets, number)
        df = pd.DataFrame(sliced_scraped_tweets)[column_names]

        df['user_location'] =  df['user'].apply(lambda x: x['location'])

        return df


#https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets
#https://medium.com/swlh/how-to-scrape-tweets-by-location-in-python-using-snscrape-8c870fa6ec25





def extract_gen(number = 10, users = 'N/A', near = 'N/A', until = 'N/A',lang = 'N/A', term = 'N/A', geocode = 'N/A', result_type = 'N/A', user = 'N/A'):
    column_names = ['date', 'url','id', 'user', 'lang', 'conversationId', 'likeCount', 'mentionedUsers','content']


    if users == 'N/A' and user == 'N/A':


        fullquery = [geocode, near, lang, result_type, until, user]
        fullquery_str = ['geocode', 'near', 'lang', 'result_type', 'until', 'from']

        queriesformerge = []
        with alive_bar(len(fullquery)) as bar:
            for idx,q in enumerate(fullquery):
                liststring = []
                if q != 'N/A':
                    if fullquery_str[idx] == 'geocode':
                        z = ", ".join(q)
                        query = fullquery_str[idx]+f':"{z}"'
                        queriesformerge.append(query)
                    else:
                        query = fullquery_str[idx]+f':"{q}"'
                        queriesformerge.append(query)
                bar()

        prefinalquery = ", ".join(queriesformerge)

        if term != 'N/A':
            finalquery = term + " " + prefinalquery
            print(finalquery)
        else:
            finalquery = prefinalquery
            print(finalquery)

        scraped_tweets = sntwitter.TwitterSearchScraper(finalquery).get_items()
        sliced_scraped_tweets = itertools.islice(scraped_tweets, number)
        df = pd.DataFrame(sliced_scraped_tweets)[column_names]


    elif users != 'N/A':

        df = pd.DataFrame(columns = column_names)

        for n, k in enumerate(users):
            scraped_tweets = sntwitter.TwitterSearchScraper('from:{}'.format(users[n])).get_items()
            sliced_scraped_tweets = itertools.islice(scraped_tweets, number)
            premerge = pd.DataFrame(sliced_scraped_tweets)[column_names]

            frames = [df, premerge]
            df = pd.concat(frames)



    df['user_location'] =  df['user'].apply(lambda x: x['location'])
    df['real_id'] =  df['user'].apply(lambda x: x['id'])
    df['displayname'] =  df['user'].apply(lambda x: x['displayname'])
    df['profileImageUrl'] =  df['user'].apply(lambda x: x['profileImageUrl'])
    df['username'] =  df['user'].apply(lambda x: x['username'])
    df['description'] =  df['user'].apply(lambda x: x['description'])
    df['content_cleaned'] =  df['content'].apply(lambda x: p.clean(x))
    df['sentencized'] =  df['content_cleaned'].apply(lambda x: sentencize(x,str(df['lang'])))

    if near != 'N/A':
        df['location_searched'] =  near
    else:
            df['location_searched'] =  'N/A'
    if geocode != 'N/A':
        vx = ", ".join(geocode)
        df['location_searched'] = vx
    return df



def sentencify(dataframe):
    compiled = []

    for i, row in dataframe.iterrows(): #for everyrow
        for idx, clause in enumerate(row['sentencized']): #for every clause in row

            taggedlistt = tag_spacy(clause)

            datarow = [row['date'],
            row['id'],
            row['real_id'],
            row['lang'],
            clause,
            row['location_searched'],
            tagornot(pos, taggedlistt[0]),
            tagornot(dep, taggedlistt[1])
            ]
            compiled.append(datarow)




    finaldf = pd.DataFrame(compiled, columns =
    ['date',
    "id",
    "userid",
    "lang.twitter",
    "divided.tweet",
    "location.searched",
    "postag",
    "deptag"
    ])

    #finaldf['lang_reg'] =  finaldf['divided.tweet'].apply(lambda x: detectlg(x))
    #finaldf['lang_nn'] =  finaldf['divided.tweet'].apply(lambda x: nn_detectlg(x))
    #finaldf['lang_nn_probability'] =  finaldf['divided.tweet'].apply(lambda x: nn_detectlg_prob(x))
    #finaldf['lang_nn_reliability'] =  finaldf['divided.tweet'].apply(lambda x: nn_detectlg_reliability(x))
    #finaldf['lang_nn_top3'] =  finaldf['divided.tweet'].apply(lambda x: nn_detectlg_top3(x))

    return finaldf

def to_csv(df, filename):
    os.chdir(resultsfolder)
    df.to_csv(filename, encoding='utf-8', index=False)

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def downloadpics(df):
    os.chdir(pic)
    import urllib.request
    ls = df['profileImageUrl'].to_list()
    for idx, i in enumerate(ls):
        urllib.request.urlretrieve(f"{i}", f"pic{idx}.jpg")

def createjsonl(df):
    os.chdir(working)
    newdf = df[['userid', 'displayname', 'username', 'description', 'lang.twitter', 'imagelink']]
    newdf.columns = ['id', 'name', 'screen_name', 'description','lang', 'img_path']
    newdf.to_json('output.jsonl', orient='records', lines=True)
    #print(data)

#SOCIAL INFORMATION ADD
#https://github.com/euagendas/m3inference
def extractsoc(strid):
    run = subprocess.check_output(["python3", "scripts/m3twitter.py", "--skip-cache", "--id", strid, "--auth", "scripts/auth.txt"])
    dict_str = run.decode("UTF-8")
    my_data = ast.literal_eval(dict_str)
    return my_data


def extract_social(a):
    dc = extractsoc(str(a))['output']
    dictage = dc['age']
    dictsex = dc['gender']
    dictorg = dc['org']
    maxage = max(dictage,key=dictage.get)
    maxsex = max(dictsex,key=dictsex.get)
    maxorg = max(dictorg,key=dictorg.get)
    lis = [maxage, maxsex, maxorg]
    return lis


def getsocialdf(df):
    x = list(set(df['userid'].to_list()))
    list_dataframe = pd.DataFrame(x)
    list_dataframe.columns = ['userid']
    list_dataframe['social'] = list_dataframe['userid'].map(lambda a:extract_social(a))
    #list_dataframe['social'] = list_dataframe['userid'].map(lambda a: max(extractsoc(str(a))['output']['age'], key=my_data.get))
    list_dataframe[['age.pred','sex.pred', 'org.pred']] = pd.DataFrame(list_dataframe.social.tolist(), index= list_dataframe.index)
    return list_dataframe

def getloc(listt):
    lat = listt[0]
    long = listt[1]
    location = geolocator.geocode(lat+","+long)
    return list(location)

def tag_spacy(string):
    doc = nlp(string)
    postagged = []
    deptagged = []
    posdeptagged = []

    for idx, i in enumerate(doc):
        x = i.text + '_' + i.pos_
        y = i.text + '_' + i.dep_
        z = i.text + '_'+ i.pos_ + '_'+ i.dep_
        postagged.append(x)
        deptagged.append(y)
        posdeptagged.append(z)

    str_postagged = " ".join(postagged)
    str_deptagged = " ".join(deptagged)
    str_posdeptagged = " ".join(posdeptagged)

    list_results = [str_postagged, str_deptagged, str_posdeptagged]
    return list_results


def tagornot(option, result):
    if option == 'no':
        return 'None'
    else:
        return result



##GUI
root=tk.Tk()
root.geometry("700x450") # setting the windows size


number_var = tk.StringVar()
users_var = tk.StringVar()
near_var = tk.StringVar()
until_var = tk.StringVar()
lang_var = tk.StringVar()
term_var = tk.StringVar()
geocode_var = tk.StringVar()
result_type_var = tk.StringVar()
social_var = tk.StringVar()
senten_var = tk.StringVar()
pos_var = tk.StringVar()
dep_var = tk.StringVar()
posdep_var = tk.StringVar()
spacylang_var = tk.StringVar()

def submit():
    global number
    global users
    global near
    global until
    global lang
    global term
    global geocode
    global result_type
    global social
    global senten
    global pos
    global dep
    global posdep
    global spacylang

    number = number_var.get()
    users = users_var.get()
    near = near_var.get()
    until = until_var.get()
    lang = lang_var.get()
    term = term_var.get()
    geocode = geocode_var.get()
    result_type = result_type_var.get()
    social = social_var.get()
    senten = senten_var.get()
    pos = pos_var.get()
    dep = dep_var.get()
    posdep = posdep_var.get()
    spacylang = spacylang_var.get()

    root.destroy()
    my_logger.info("\n")
    my_logger.info("------------------------------")
    my_logger.info("INPUT SUMMARY")
    my_logger.info("------------------------------")
    my_logger.info("Number of tweets (per user, if indicated): " + number)
    my_logger.info("Users: " + users)
    my_logger.info("Proximal Location " + near)
    my_logger.info("Date: " +  until)
    my_logger.info("Language: " +  lang)
    my_logger.info("Term: " +  term)
    my_logger.info("Geocode: " +  geocode)
    my_logger.info("Type of tweet: " + result_type)
    my_logger.info("\n")
    my_logger.info("Demographic prediction: " +  social)
    my_logger.info("Sentencize Tweets: " +  senten)
    my_logger.info("POS Tag Tweets: " +  pos)
    my_logger.info("DEP Tag Tweets: " +  dep)
    my_logger.info("POSDEP Tag Tweets: " +  posdep)
    my_logger.info("SpaCy Model Language: " +  spacylang)
    my_logger.info("------------------------------")
    my_logger.info("\n")

root.title('TweetCorp v.1') #set title

number_label = tk.Label(root, text = 'Number of tweets', font=('calibre',10, 'bold'))
number_entry = tk.Entry(root,textvariable = number_var, font=('calibre',10,'normal'), width  = 40)
number_var.set("50000")

users_label = tk.Label(root, text = 'Users (comma-ed)', font=('calibre',10, 'bold'))
users_entry = tk.Entry(root,textvariable = users_var, font=('calibre',10,'normal'), width  = 40)
#users_var.set("barackobama")
#users_var.set("bbcmundo,nytimes")
users_var.set("N/A")

near_label = tk.Label(root, text = 'Proximal location', font=('calibre',10, 'bold'))
near_entry = tk.Entry(root,textvariable = near_var, font=('calibre',10,'normal'), width  = 40)
near_var.set("N/A")

until_label = tk.Label(root, text = 'Date', font=('calibre',10, 'bold'))
until_entry = tk.Entry(root,textvariable = until_var, font=('calibre',10,'normal'), width  = 40)
until_var.set("2021-01-28")

lang_label = tk.Label(root, text = 'Language', font=('calibre',10, 'bold'))
lang_entry = tk.Entry(root,textvariable = lang_var, font=('calibre',10,'normal'), width  = 40)
lang_var.set("en")

term_label = tk.Label(root, text = 'Specific term', font=('calibre',10, 'bold'))
term_entry = tk.Entry(root,textvariable = term_var, font=('calibre',10,'normal'), width  = 40)
term_var.set("N/A")

geocode_label = tk.Label(root, text = 'Geocode (lat, long, range in km; comma-ed)', font=('calibre',10, 'bold'))
geocode_entry = tk.Entry(root,textvariable = geocode_var, font=('calibre',10,'normal'), width  = 40)
geocode_var.set("13.7567, 121.0584, 12km")
#geocode_var.set("N/A").

result_type_label = tk.Label(root, text = 'Result type', font=('calibre',10, 'bold'))
result_type_entry = tk.OptionMenu(root, result_type_var, "mixed", "recent", "popular")
result_type_var.set("N/A") #mixed, recent, popular

social_label = tk.Label(root, text = 'Demographic prediction', font=('calibre',10, 'bold'))
social_entry = tk.OptionMenu(root, social_var, "yes", "no")
social_var.set("no")

senten_label = tk.Label(root, text = 'Sentencize tweets?', font=('calibre',10, 'bold'))
senten_entry = tk.OptionMenu(root, senten_var, "yes", "no")
senten_var.set("yes")

senten_label = tk.Label(root, text = 'Sentencize tweets?', font=('calibre',10, 'bold'))
senten_entry = tk.OptionMenu(root, senten_var, "yes", "no")
senten_var.set("yes")

pos_label = tk.Label(root, text = 'POS tag tweets? (For sentencized)', font=('calibre',10, 'bold'))
pos_entry = tk.OptionMenu(root, pos_var, "yes", "no")
pos_var.set("yes")

dep_label = tk.Label(root, text = 'DEP tag tweets? (For sentencized)', font=('calibre',10, 'bold'))
dep_entry = tk.OptionMenu(root, dep_var, "yes", "no")
dep_var.set("yes")

posdep_label = tk.Label(root, text = 'POSDEP tag tweets? (For sentencized)', font=('calibre',10, 'bold'))
posdep_entry = tk.OptionMenu(root, posdep_var, "yes", "no")
posdep_var.set("no")

spacylang_label = tk.Label(root, text = 'Model language', font=('calibre',10, 'bold'))
spacylang_entry = tk.OptionMenu(root, spacylang_var, "N/A", "English", "Tagalog")
spacylang_var.set("English")


sub_btn=tk.Button(root,text = 'Submit', command = submit) # creating a button using the widget button that will call the submit function



number_label.grid(row=0,column=0, sticky = 'W') # placing the label and entry in the required position using grid method
number_entry.grid(row=0,column=1, sticky = 'W')
users_label.grid(row=1,column=0, sticky = 'W')
users_entry.grid(row=1,column=1, sticky = 'W')
near_label.grid(row=2,column=0, sticky = 'W')
near_entry.grid(row=2,column=1, sticky = 'W')
until_label.grid(row=3,column=0, sticky = 'W')
until_entry.grid(row=3,column=1, sticky = 'W')
lang_label.grid(row=4,column=0, sticky = 'W')
lang_entry.grid(row=4,column=1, sticky = 'W')
term_label.grid(row=5,column=0, sticky = 'W')
term_entry.grid(row=5,column=1, sticky = 'W')
geocode_label.grid(row=6,column=0, sticky = 'W')
geocode_entry.grid(row=6,column=1, sticky = 'W')
result_type_label.grid(row=7,column=0, sticky = 'W')
result_type_entry.grid(row=7,column=1, sticky = 'W')
social_label.grid(row=9,column=0, sticky = 'W')
social_entry.grid(row=9,column=1, sticky = 'W')
senten_label.grid(row=10,column=0, sticky = 'W')
senten_entry.grid(row=10,column=1, sticky = 'W')

pos_label.grid(row=11,column=0, sticky = 'W')
pos_entry.grid(row=11,column=1, sticky = 'W')
dep_label.grid(row=12,column=0, sticky = 'W')
dep_entry.grid(row=12,column=1, sticky = 'W')
posdep_label.grid(row=13,column=0, sticky = 'W')
posdep_entry.grid(row=13,column=1, sticky = 'W')
spacylang_label.grid(row=14, column=0, sticky = 'W')
spacylang_entry.grid(row=14,column=1, sticky = 'W')

sub_btn.grid(row=15,column=1, sticky = 'S')

root.mainloop() # performing an infinite loop for the window to display




#def extract_gen(number = 10, users = 'N/A', near = 'N/A', until = 'N/A',lang = 'N/A', term = 'N/A', geocode = 'N/A', result_type = 'N/A', user = 'N/A')
#dataframe = extract_gen(100, 'N/A', 'Manila', '1km')
#dataframe = extract_gen(20)
#dataframe = extract_gen(10, 'N/A', "Manila", "2020-09-19", "en")


#PIPELINE
my_logger.info('Initiating extraction...')

if users != "N/A":
    #users = users.split(", ")
    separators = ", ", ","
    users = custom_split(separators, users)

if geocode != "N/A":
    #users = users.split(", ")
    separators = ", ", ","
    geocode = custom_split(separators, geocode)
    print(geocode)


dataframe = extract_gen(int(number), users, near, until, lang, term, geocode, result_type)

#dataframe = extract_gen(2, ['bbcmundo','nytimes'], "N/A", "N/A", "N/A")

#Language
if spacylang == "English":
    nlp = spacy.load("en_core_web_sm")
else:
    my_logger.info('Other models not available yet.')


if senten == "yes":
    my_logger.info('Initiating sentencizing...')
    dataframe  = sentencify(dataframe)

#downloadpics(dataframe)
#createjsonl(dataframe)
#m3twitter.infer('output.jsonl')


if geocode != 'N/A':
    geocode = getloc(geocode)
else:
    geocode == 'NA'

if social == "yes":

    my_logger.info('Producing pre-prediction data...')
    file_name = f'results_presocial_{until}_{geocode}.csv'
    to_csv(dataframe, file_name)
    my_logger.info(f'First corpus created. Exported as CSV. File name is: {file_name}.')
    my_logger.info('Predicting demographic from profiles...')
    socialdata = getsocialdf(dataframe)
    finaldf = pd.merge(dataframe, socialdata, on="userid")

    file_name = f'results_withsocial_{until}_{geocode}.csv'
    my_logger.info('Exporting to CSV.')
    to_csv(finaldf, file_name)
    my_logger.info(f'Second corpus created. Exported as CSV. File name is: {file_name}.')
else:
    file_name = f'results_nosocial_{until}_{geocode}.csv'
    to_csv(dataframe, file_name)
    my_logger.info(f'Corpus created. Exported as CSV. File name is: {file_name}.')
    #print(getloc(['14.599512', '120.98', '10km']))
