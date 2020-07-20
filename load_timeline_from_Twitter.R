#############################################
#Load data from Twitter and saves it to csv
#Author @Tomasz Hachaj, @Justyna Miazga
#############################################

# load the package
library(twitteR)
library(rtweet)


# set the credentials
consumer_key <- "XXX"
consumer_secret <- "XXX"
access_token <- "XXX"
access_secret <- "XXX"

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)

twitterUser_rdt <- getUser("realDonaldTrump")
tweets_rdt <- userTimeline(twitterUser_rdt, includeRts = T,n = 3200)

df <- as.data.frame(tweets_rdt[[1]])
for (a in 2:length(tweets_rdt))
{
  df <- rbind(df, as.data.frame(tweets_rdt[[a]]))
}
write.csv(df,'donald22.csv', row.names=F)
twitterUser_rdt <- getUser("justintrudeau")
tweets_rdt <- userTimeline(twitterUser_rdt, includeRts = T,n = 3200)

df <- as.data.frame(tweets_rdt[[1]])
for (a in 2:length(tweets_rdt))
{
  df <- rbind(df, as.data.frame(tweets_rdt[[a]]))
}
write.csv(df,'justintrudeau22.csv', row.names=F)
twitterUser_rdt <- getUser("theresa_may")
tweets_rdt <- userTimeline(twitterUser_rdt, includeRts = T,n = 3200)

df <- as.data.frame(tweets_rdt[[1]])
for (a in 2:length(tweets_rdt))
{
  df <- rbind(df, as.data.frame(tweets_rdt[[a]]))
}
write.csv(df,'theresa_may22.csv', row.names=F)
twitterUser_rdt <- getUser("joannakrupa")
tweets_rdt <- userTimeline(twitterUser_rdt, includeRts = T,n = 3200)

df <- as.data.frame(tweets_rdt[[1]])
for (a in 2:length(tweets_rdt))
{
  df <- rbind(df, as.data.frame(tweets_rdt[[a]]))
}
write.csv(df,'joannakrupa22.csv', row.names=F)
twitterUser_rdt <- getUser("oprah")
tweets_rdt <- userTimeline(twitterUser_rdt, includeRts = T,n = 3200)

df <- as.data.frame(tweets_rdt[[1]])
for (a in 2:length(tweets_rdt))
{
  df <- rbind(df, as.data.frame(tweets_rdt[[a]]))
}
write.csv(df,'oprah22.csv', row.names=F)
