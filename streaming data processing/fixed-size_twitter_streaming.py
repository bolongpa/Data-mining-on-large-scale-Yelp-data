import sys
from pyspark import SparkContext
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import random
import csv

# consumer key, consumer secret, access token, access secret.
ckey = "zkXJntGOetb4jmotOPgDu2RpN"
csecret = "D4nfKr0TJxUVyNh6SQlTbbOHOyxEAEN0WcdoA6Wmin232zJX6K"
atoken = "1275144443847249921-QW7IRMCCf67EuvmHuQDfRK1gwpLJcr"
asecret = "clCryh8cfVjYGxfCi4qfHPseGhCT1XYOud6KJTGbQ1vWe"


class myStreamListener(StreamListener):
    global cnt
    global cnt_tag
    global savedtags
    global sc
    global output_path

    def on_data(self, data):

        def frequenttags(tweettags):

            data = sc.parallelize(tweettags).map(lambda x: (x['text'], 1))\
                .reduceByKey(lambda x, y: x + y).map(lambda x: (x[1], [x[0]]))\
                .reduceByKey(lambda x, y: x + y).sortBy(lambda x: -x[0]).take(3)
            # write into file
            with open(output_path, 'a') as f:
                filewriter = csv.writer(f)
                filewriter.writerow(['The number of tweets with tags from the beginning: ' + str(cnt)])
                for n in data:
                    for t in sorted(n[1]):
                        filewriter.writerow([str(t) + ' : ' + str(n[0])])
                filewriter.writerow([])

        global cnt
        global cnt_tag
        all_data = json.loads(data)
        if 'entities' in all_data.keys():
            if 'hashtags' in all_data['entities'].keys():
                if all_data['entities']['hashtags']:  # tweet hashtags not empty
                    cnt += 1
                    tags = all_data['entities']['hashtags']
                    for t in tags:
                        if cnt_tag < 100:
                            cnt_tag += 1
                            savedtags.append(t)
                            frequenttags(savedtags)
                        else:
                            if random.random() < 100 / cnt_tag:
                                cnt_tag += 1
                                dropindex = random.randrange(100)
                                savedtags.pop(dropindex)
                                savedtags.append(t)
                                frequenttags(savedtags)
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    arg = sys.argv
    if len(arg) != 3:
        print('Please rerun and confirm the parameters.')
    else:
        port_num = int(arg[1])
        output_path = arg[2]
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)

        # build RDD and initialization
        sc = SparkContext()
        cnt = 0  # count the number of tweets with tags
        cnt_tag = 0  # count the number of tags received
        savedtags = []

        twitterStream = Stream(auth, myStreamListener())
        twitterStream.filter(track=['america'], languages=['en'])
