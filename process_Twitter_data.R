#############################################
#Process Twitter data
#Author @Tomasz Hachaj, @Justyna Miazga
#############################################

#load in the libraries we will need
#install.packages("tidyverse", dependencies=TRUE)
#install.packages("backports")
#install.packages("broom")

library(tidyverse)
library(tidytext)
library(glue)
library(stringr)
library(corrgram)

path.to.files <- "C:\\Users\\justi\\Desktop\\Doktorat\\Sentiment analysis czerwiec\\"
#Load data from CSV, data.twitter - data from Twitter, sentimetn - CNN sentiment
#data.twitter <- read.csv(paste(path.to.files, "dane_joannakrupa.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "dane_joannakrupa_sentiment.csv", sep = ""), header = TRUE)
#data.vader <- read.csv(paste(path.to.files, "dane_joannakrupavader.csv", sep = ""), header = TRUE)

#data.twitter <- read.csv(paste(path.to.files, "dane_justintrudeau.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "dane_justin_sentiment.csv", sep = ""), header = TRUE)
#data.vader <- read.csv(paste(path.to.files, "dane_justintrudeauVader.csv", sep = ""), header = TRUE)

#data.twitter <- read.csv(paste(path.to.files, "dane_oprah.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "dane_oprah_sentiment.csv", sep = ""), header = TRUE)
#data.vader <- read.csv(paste(path.to.files, "dane_oprahVader.csv", sep = ""), header = TRUE)

#data.twitter <- read.csv(paste(path.to.files, "dane_theresamay.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "dane_theresamay_sentiment.csv", sep = ""), header = TRUE)
#data.vader <- read.csv(paste(path.to.files, "dane_theresamayVader.csv", sep = ""), header = TRUE)

#data.twitter <- read.csv(paste(path.to.files, "danerdt.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "danerdt_sentiment.csv", sep = ""), header = FALSE)
#data.vader <- read.csv(paste(path.to.files, "danerdtVader.csv", sep = ""), header = TRUE)

#data.twitter <- read.csv(paste(path.to.files, "hm.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "hm_sentiment.csv", sep = ""), header = TRUE)
#data.vader <- read.csv(paste(path.to.files, "hmvader.csv", sep = ""), header = TRUE)

#data.twitter <- read.csv(paste(path.to.files, "Microsoft.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "Microsoft_sentiment.csv", sep = ""), header = TRUE)
#data.vader <- read.csv(paste(path.to.files, "MicrosoftVader.csv", sep = ""), header = TRUE)

#data.twitter <- read.csv(paste(path.to.files, "ikea_usa.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "ikea_usa_sentiment.csv", sep = ""), header = TRUE)
#data.vader <- read.csv(paste(path.to.files, "ikea_usaVader.csv", sep = ""), header = TRUE)

#data.twitter <- read.csv(paste(path.to.files, "LeagueOfLegends.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "LeagueOfLegends_sentiment.csv", sep = ""), header = TRUE)
#data.vader <- read.csv(paste(path.to.files, "LeagueOfLegendsVader.csv", sep = ""), header = TRUE)

#data.twitter <- read.csv(paste(path.to.files, "TerriIrwin.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "TerriIrwin_sentiment.csv", sep = ""), header = TRUE)
#data.vader <- read.csv(paste(path.to.files, "TerriIrwinVader.csv", sep = ""), header = TRUE)

#data.twitter <- read.csv(paste(path.to.files, "DrLindseyFitz.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "DrLindseyFitz_sentiment.csv", sep = ""), header = TRUE)
#data.vader <- read.csv(paste(path.to.files, "DrLindseyFitzVader.csv", sep = ""), header = TRUE)

#data.twitter <- read.csv(paste(path.to.files, "wendymoore99.csv", sep = ""), stringsAsFactors = FALSE)
#sentimetn <- read.csv(paste(path.to.files, "wendymoore99_sentiment.csv", sep = ""), header = TRUE)
#data.vader <- read.csv(paste(path.to.files, "wendymoore99Vader.csv", sep = ""), header = TRUE)

data.twitter <- read.csv(paste(path.to.files, "Castrofied.csv", sep = ""), stringsAsFactors = FALSE)
sentimetn <- read.csv(paste(path.to.files, "Castrofied_sentiment.csv", sep = ""), header = TRUE)
data.vader <- read.csv(paste(path.to.files, "CastrofiedVader.csv", sep = ""), header = TRUE)

tt <- data.twitter$text

#Sentiment analysis with Bing lexicon
# remove any dollar signs (they're special characters in R)
fileText <- gsub("\\$", "", tt) 
resultsList <- list()
tokensList <- list()
for (a in 1:length(fileText))
{
  print(a)
  oneTweet <- fileText[a]
  # tokenize
  tokens <- data_frame(text = oneTweet) %>% unnest_tokens(word, text)
  # get the sentiment from the first text: 
  tokensList[[a]] <- nrow(tokens)

  df <- data.frame(positive = c(0), negative = c(0))
  dfresult<- tokens %>%
    inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
    count(sentiment) %>% # count the # of positive & negative words
    spread(sentiment, n, fill = 0)# made data wide rather than narrow
    if (!is.null(dfresult$positive[1]))
      df$positive[1] <- dfresult$positive[1]
  
    if (!is.null(dfresult$negative[1]))
      df$negative[1] <- dfresult$negative[1]
  print(df)
  resultsList[[a]] <- df
    #mutate(sentiment = positive - negative) # # of positive words - # of negative owrds
}


#Appending Bing - based sentiment and CNN - based sentiment to Twitter data
data.twitter$positive <- rep(x = 0, length(data.twitter$text))
data.twitter$negative <- rep(x = 0, length(data.twitter$text))
data.twitter$sentiment <- rep(x = 0, length(data.twitter$text))
data.twitter$wcbing <- unlist(tokensList)

for (a in 1:length(fileText))
{
  data.twitter$positive[a] <- resultsList[[a]]$positive
  data.twitter$negative[a] <- resultsList[[a]]$negative
  data.twitter$sentiment[a] <- data.twitter$positive[a] - data.twitter$negative[a]
}
colnames(sentimetn) <- c('Negative', 'Positive', 'WC')

data.twitter$nn.negative <- sentimetn$Negative
data.twitter$nn.positive <- sentimetn$Positive
data.twitter$nn.wc <- sentimetn$WC
data.twitterhelp <- data.twitter


#Standardizing
data.twitterhelp$favoriteCount <- (data.twitterhelp$favoriteCount - mean(data.twitterhelp$favoriteCount)) / sd(data.twitterhelp$favoriteCount)
data.twitterhelp$retweetCount <- (data.twitterhelp$retweetCount - mean(data.twitterhelp$retweetCount)) / sd(data.twitterhelp$retweetCount)



ile <- list()
i <- 1
for (a in -6:7)
{
  ile[[i]] <- nrow(data.twitterhelp[data.twitterhelp$sentiment == a,])
  print(ile[[i]])
  i <- i + 1
}
#This table presents the percentage of tweets that obtained certain lexicon-based score in percent
unlist(ile) / nrow(data.twitterhelp) * 100

#Lexicon - based scoring (2)
plot(data.twitterhelp$sentiment, data.twitterhelp$favoriteCount, 
     main='Donald J. Trump:
     favorite count and sentiment score', 
     xlab="Lexicon Sentiment Score", ylab="Standardized Favorite Count")
plot(data.twitterhelp$sentiment, data.twitterhelp$retweetCount,
     main="Donald J. Trump: 
     retweet count and sentiment score", 
     xlab="Lexicon Sentiment score", ylab="Standardized Retweet Count")


stand.fc <- (data.twitter$favoriteCount - mean(data.twitter$favoriteCount)) / sd(data.twitter$favoriteCount)
stand.rc <- (data.twitter$retweetCount - mean(data.twitter$retweetCount)) / sd(data.twitter$retweetCount)

#Lexicon - based scoring (3)
plot(data.twitter$sentiment / data.twitter$wcbing, stand.fc, 
     main='Donald J. Trump:
     favorite count and sentiment score / wc', 
     xlab="Lexicon Sentiment Score / wc", ylab="Standardized Favorite Count")
plot(data.twitter$sentiment / data.twitter$wcbing, stand.rc, 
     main='Donald J. Trump:
     retweet count and sentiment score / wc', 
     xlab="Lexicon Sentiment Score / wc", ylab="Standardized Retweet Count")


#CNN - based scoring
dabefiltered <- data.twitterhelp[data.twitterhelp$nn.wc > 1, ]

plot(dabefiltered$nn.positive, dabefiltered$favoriteCount, 
     main='Donald J. Trump:
     favorite count and sentiment score', 
     xlab="DNN Sentiment Score", ylab="Standardized Favorite Count")
plot(dabefiltered$nn.positive, dabefiltered$retweetCount,
     main="Donald J. Trump: 
     retweet count and sentiment score", 
     xlab="DNN Sentiment score", ylab="Standardized Retweet Count")

#correlation matrix
df.to.check.cor <- data.frame(data.twitterhelp$sentiment, 
                              data.twitterhelp$nn.positive,
                              data.twitter$sentiment / data.twitter$wcbing,
                              data.twitter$wcbing,
                              data.twitterhelp$retweetCount,
                              data.twitterhelp$favoriteCount
                              )
#NN
cor(data.twitterhelp$nn.positive, data.twitterhelp$favoriteCount)
cor(data.twitterhelp$nn.positive, data.twitterhelp$retweetCount)

#bing
cor(data.twitterhelp$sentiment, data.twitterhelp$favoriteCount)
cor(data.twitterhelp$sentiment, data.twitterhelp$retweetCount)

#vader
cor(data.vader$Positive, data.twitterhelp$favoriteCount)
cor(data.vader$Positive, data.twitterhelp$retweetCount)

#cor(data.twitterhelp$nn.positive, stand.fc)
#cor(data.twitterhelp$nn.positive, stand.rc)




cor_m <- formatC(cor(df.to.check.cor), digits = 2, format = "f")

#Shapiro-Wilk normality test:
shapiro.test(data.twitterhelp$sentiment)
shapiro.test(data.twitterhelp$nn.positive)
shapiro.test(data.twitter$sentiment / data.twitter$wcbing)
shapiro.test(data.twitter$wcbing)
shapiro.test(data.twitterhelp$retweetCount)
shapiro.test(data.twitterhelp$favoriteCount)




#CNN error rate during training
train.plot <- read.csv('e:\\Publikacje\\Miazga\\trening_plot.txt', stringsAsFactors = FALSE, header = FALSE)
train.plot <- train.plot$V1 / 100
train.plot <- 1 - train.plot
plot(x = 1:length(train.plot), y = train.plot, 
     main = "Error rate during CNN training",
     xlab = "Epochs",
     ylab = "Error rate")
lines(x = 1:length(train.plot), y = train.plot, col='blue')

############################
#alpha (2)
#JK
8.0 / log(946000)
13.00 / log(946000)

#JT
71.0 /log(4380000)
196.0 / log(4380000)

#OW
90.0 / log(41700000)
148.0 / log(41700000)

#TM
252.0 / log(671000)
406 / log(671000)

#DT
17854 / log(55700000)
76510 / log(55700000)

#IU

4/log(373800)
17 /log(373800)

