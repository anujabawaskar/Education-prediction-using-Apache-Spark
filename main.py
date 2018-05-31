import bz2
import json
import os
from pyspark.context import SparkContext
from pyspark.accumulators import AccumulatorParam
import numpy as np
from scipy import spatial
import pandas as pd
import re
import operator
import csv

CACHE_DIR = "D:\TwitterDatastream\PYTHONCACHE_SMALL"
EDU_DATA = 'merged.csv'
TRAIN_FEAT_CSV = 'testFeat.csv'
TRAIN_LABS_CSV = 'testLabs.csv'
TRAIN_FEAT_LABS_CSV = 'testFeatLabs.csv'
FEATURE_NAMES_CSV = 'featureNames.csv'
sc = SparkContext('local', 'test')
# location_data = pd.read_csv('new_merged.csv')

class WordsSetAccumulatorParam(AccumulatorParam):
    def zero(self, v):
        return set()
    def addInPlace(self, acc1, acc2):
        return acc1.union(acc2)

# An accumulator used to build the word vocabulary
class WordsDictAccumulatorParam(AccumulatorParam):
    def zero(self, v):
        return dict()
    def addInPlace(self, acc1, acc2):
        for key in acc2.keys():
            try:
                acc1[key] += acc2[key]
            except:
                acc1[key] = acc2[key]
        return acc1

# An accumulator used to build the word vocabulary
# vocabulary = sc.accumulator(set(), WordsSetAccumulatorParam())
vocabulary = sc.accumulator(dict(), WordsDictAccumulatorParam())

# load Education census data
location_data = pd.read_csv(EDU_DATA)
area_dict = dict(zip(location_data['city'], location_data[['fips', 'without_hsd','with_hsd', 'somecollege', 'bachelors']].values.tolist()))
county_dict = dict(zip(location_data['county'], location_data[['fips', 'without_hsd','with_hsd', 'somecollege', 'bachelors']].values.tolist()))
coord_dict = {tuple(x[:2]):x[2] for x in location_data[['lat', 'lng', 'county']].values}

# create a KD tree of known county center locations to be used to map a tweet coordinate to a county
latlon = list()
for index, row in location_data.iterrows():
    latlon.append([location_data['lat'][index], location_data['lng'][index]])

latlon = np.array(latlon)
latlonKDT = spatial.KDTree(latlon)

# function to map place, location or coordinate data from a tweet to a FIPS code of the county and the education
# level distribution of that county
def mapToCounty(place, location, coordinates):
    # coordr_dict = {tuple(x[:2]):x[2] for x in location_data[['lat_r', 'lng_r', 'county']].values}
    if place:
        place = (place.split(",")[0]).lower()
        # country = (place.split(",")[1]).lower()
        try:
            if area_dict[place]: return area_dict[place]
        except: None
    if location:
        location = (location.split(",")[0]).lower()
        try:
            if area_dict[location]: return area_dict[location]
        except: None
    if coordinates:
        closestLoc = spatial.KDTree(latlon).query(coordinates, k=1, distance_upper_bound=9)[1]
        try:
            closest = latlon[closestLoc]
        except:
            return None
        # closest = spatial.KDTree(latlon).query(coordinates, k=1, distance_upper_bound=9)
        # if closest[0] != float('inf') and latlon[closest[1]][0] != 0. and latlon[closest[1]][1] != 0.:
        #     print(coordinates, closest, latlon[closest[1]])
        # return closest[0], closest[1]
        if coord_dict[closest[0], closest[1]]:
            county_k = coord_dict[(closest[0], closest[1])]
            return county_dict[county_k]

    return None

# Load Tweets from each file (.bz2 or .json)
def load_bz2_json(filename):
    if '.bz2' in filename:
        with bz2.open(filename, 'rt') as f:
            lines = str(f.read()).split('\n')
    else:
        with open(filename) as f:
            lines = str(f.readlines()).split('\\n')
    num_lines = len(lines)
    tweets = []
    for line in lines:
        try:
            if line == "":
                num_lines -= 1
                continue
            tweets.append(json.loads(line))
        except:
            continue
    # print(filename, len(tweets))
    return tweets

# strip each tweet object and keep only whats necessary in a dictonary
def load_tweet(tweet, tweets_saved):
    try:
        # tweet_id = tweet['id']
        tweet_text = tweet['text']
        tweet_user_id = tweet['user']['id']
        tweet_user_location = tweet['user']['location']
        tweet_user_lang = tweet['user']['lang']
        try: tweet_coordinates = tweet['coordinates']['coordinates']
        except: tweet_coordinates = None
        try: tweet_place = tweet['place']['full_name']
        except: tweet_place = None
        map_to_county = mapToCounty(tweet_place, tweet_user_location, tweet_coordinates)
        if map_to_county:
            tweet_county = int(map_to_county[0])
            tweet_education_level = tuple(map_to_county[1:])
        else:
            tweet_county = None
            tweet_education_level = None
            # created_at = tweet['created_at']
    except KeyError:
        return {}, tweets_saved

    data = {'tweet_text': tweet_text,
            # 'tweet_id': tweet_id,
            'tweet_user_id': tweet_user_id,
            # 'tweet_user_location': tweet_user_location,
            'tweet_user_lang': tweet_user_lang,
            # 'tweet_place': tweet_place,
            # 'tweet_coordinates': tweet_coordinates,
            'tweet_county': tweet_county,
            'tweet_education_level': tweet_education_level}
            # 'date_loaded': datetime.datetime.now(),
            # 'tweet_json': json.dumps(tweet)}

    tweets_saved += 1
    return data, tweets_saved

wordPattern = re.compile(r"\b[A-Za-z_.,!\"']+\b", re.IGNORECASE)
httpPattern = re.compile(r"^RT |@\S+|http\S+", re.IGNORECASE)

# Function that uses regular expressions to remove unwanted characters, URLs, etc. and split tweet_text
# into meaningful words
def parseTweetText(tweet):
    text = tweet['tweet_text']
    text = httpPattern.sub(r"", text)
    words = wordPattern.findall(text)
    tweet['tweet_text'] = words #list(zip(words, [1]*len(words)))
    # print(tweet)
    return tweet

# function to combine word lists and count frequency of each word locally
def combineWordLists(x ,y):
    global vocabulary
    if isinstance(x, dict):
        wordDict = x
        xny = y
    else:
        wordDict = dict()
        xny = x + y
    for w in xny:
        # vocabulary +=[w]
        vocabulary += {w: 1}
        try:
            wordDict[w] += 1
        except:
            wordDict[w] = 1

    return wordDict

# function to add words to the vocabulary and count frequency of each word globally
def genVocabulary(x):
    global vocabulary
    arr = x[1]
    if isinstance(arr, dict):
        return x
    else:
        wordDict = dict()
        for w in arr:
            vocabulary += {w: 1}
            try:
                wordDict[w] += 1
            except:
                wordDict[w] = 1
        x = (x[0],wordDict)
        return x

# read tweets from each file and parse them into dictionaries with only relevant data
def handle_file(filename):
    tweets = load_bz2_json(filename)
    tweet_dicts = []
    tweets_saved = 0
    for tweet in tweets:
        tweet_dict, tweets_saved = load_tweet(tweet, tweets_saved)
        if tweet_dict:
            tweet_dicts.append(tweet_dict)

    return tweet_dicts

# filter only tweets that have text, land, education and are written in english
def filterTweets(tweet):
    # location = tweet['tweet_user_location']
    # coordinates = tweet['tweet_place']
    # place = tweet['tweet_coordinates']
    text = tweet['tweet_text']
    lang = tweet['tweet_user_lang']
    education = tweet['tweet_education_level']
    county = tweet['tweet_county']
    # if location or coordinates or place: ret = True
    # else: return False
    if not text or text == []: return False
    if lang != 'en': return False
    if education is None or county is None: return False

    return True

# store all data into CSV files
def storeResults(traindata, vocab):
    columnIdx = {vocab[voc][0]: voc for voc in range(len(vocab))}

    with open(TRAIN_FEAT_CSV, 'wt') as trainFeatFile, open(TRAIN_LABS_CSV, 'wt') as trainLabsFile, open(TRAIN_FEAT_LABS_CSV, 'wt') as trainFeatLabsFile:
        trainFeatwriter = csv.writer(trainFeatFile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        trainLabswriter = csv.writer(trainLabsFile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        trainFeatLabswriter = csv.writer(trainFeatLabsFile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for row in traindata:
            edu = row[0][1]
            featDict = row[1]
            feats = np.zeros(len(columnIdx))
            for key in featDict:
                try:
                    feats[columnIdx[key]] = featDict[key]
                except:
                    continue
            trainFeatwriter.writerow(feats.tolist())
            trainLabswriter.writerow(list(edu))
            combList = list(edu) + feats.tolist()
            trainFeatLabswriter.writerow(combList)

# main function with all the Spark code
def main():
    fileNames = sc.parallelize([])

    # generate a list of all files in the data directory
    for root, dirs, files in os.walk(CACHE_DIR):
        subFileNames = sc.parallelize(files).map(lambda file: os.path.join(root, file))
        fileNames = sc.union([fileNames, subFileNames])
    # load all tweets and filter
    tweetsRdd = fileNames.flatMap(lambda file: handle_file(file)).filter(lambda tweet: filterTweets(tweet))
    # clean, parse and filter tweets and map each to county and education level
    wordsRdd = tweetsRdd.map(lambda tweet: parseTweetText(tweet)).filter(lambda tweet: filterTweets(tweet))
    # set county and education level as the key for each tweet and keep only the text as value
    countyEduRdd = wordsRdd.map(lambda tweet: ((tweet['tweet_county'], tweet['tweet_education_level']), tweet['tweet_text']))
    # aggregate tweets based on county level and generate vocabulary
    countyEduRdd = countyEduRdd.reduceByKey(lambda x, y: combineWordLists(x, y)).map(lambda z: genVocabulary(z))
    tempRes = countyEduRdd.collect()
    # print(tempRes)
    print(len(tempRes))
    vocabRDD = sc.parallelize(vocabulary.value.items())
    # filter out words that only occur once in the entire dataset (mainly noise)
    vocabRDD = vocabRDD.filter(lambda voc: True if voc[1] > 1 else False)
    # print("vocabulary = ", sorted(vocabulary.value.items(), key=operator.itemgetter(1)))
    vocab = sorted(vocabRDD.collect(), key=operator.itemgetter(1), reverse=True)
    # print("vocabulary = ", vocab)
    print("vocabulary size = ", len(vocab))
    storeResults(tempRes, vocab)

if __name__ == "__main__":
    main()