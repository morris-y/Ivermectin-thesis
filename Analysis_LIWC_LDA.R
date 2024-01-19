############################## language detection ################################
#install.packages("textcat")
library(textcat)

ivermectin_data <- read_csv("FB_eng_column_fixed.csv")
column_ivermectin <-ivermectin_data%>% 
  filter(
    grepl("Ivermectin", ivermectin_data$message)|
      grepl("Ivermectin", ivermectin_data$description)|
      grepl("ivermectin", ivermectin_data$message)|
      grepl("ivermectin", ivermectin_data$description))

lang <- textcat(column_ivermectin$message)
lang[is.na(lang)]<- "non"

num <- 0
index_list <- {}
for (i in seq.int(1:length(lang))){
  if (lang[[i]] == "english" ){
    index_list <- append(index_list, i)
    num <-  num + 1
  }
}
print(num)

new_eng <- {}
for (j in index_list){
  new_eng <- rbind(new_eng, column_iver[j,])
}

############################# 預處理 ################################


library(tidyverse)
library(quanteda)
library(quanteda.textplots)
library(quanteda.textstats)
#install.packages(("quanteda.textplots"))
#install.packages("compiler")
library(compiler)

#data是 column_fixed

#corpus #合併後只需要分析message欄位
corpus_iver <- corpus(column_fixed, text_field = "message") 

# 概觀
ndoc(corpus_iver) # Number of Documents
nchar(corpus_iver[1:20]) # Number of character for the first 20 documents
ntoken(corpus_iver[1:20]) # Number of tokens for the first 20 documents
ntoken(corpus_iver[1:20], remove_punct = TRUE) # Number of tokens for the first 20 documents after removing punctuation

View(head(docvars(corpus_iver))) #欄位長什麼樣

# tokeniasation Creating DFM 清理掉一些不會用到的分析單位
tokens_iver <- tokens(corpus_iver,
                         remove_punct = TRUE,
                         remove_numbers = TRUE,
                         remove_url = TRUE,
                         verbose = TRUE)
dfm_iver <- dfm(tokens_iver) # 直接有matrix出來

# Inspecting the results 前三十最多的字
topfeatures(dfm_iver, 50)

# What do they say about "vaccine"?
view(kwic(tokens_iver, "vaccine", 3))
write.csv(
  kwic(tokens_iver, "vaccine", 3),"what_do_they_say_about_vaccine.csv"
)

# Plotting a histogram
features <- topfeatures(dfm_remostwo, 20)  # Putting the top 20 words into a new object
data.frame(list(term = names(features), frequency = unname(features))) %>% # Create a data.frame for ggplot
  ggplot(aes(x = reorder(term,-frequency), y = frequency)) + # Plotting with ggplot2
  geom_point() +
  theme_bw() +
  labs(x = "Term", y = "Frequency") +
  theme(axis.text.x=element_text(angle=90, hjust=1))

# Doing it again, removing stop words this time!

# Defining custom stopwords
customstopwords <- c("two", "one","also","like","even")
dfm_remostwo <- dfm_remove(dfm_iver, c(stopwords('english'), customstopwords))

# Inspecting the results again
topfeatures(dfm_remostwo, 100)

# Top words again
features <- topfeatures(dfm_remostwo, 100)  # Putting the top 100 words into a new object
data.frame(list(term = names(features), frequency = unname(features))) %>% # Create a data.frame for ggplot
  ggplot(aes(x = reorder(term,-frequency), y = frequency)) + # Plotting with ggplot2
  geom_point() +
  theme_bw() +
  labs(x = "Term", y = "Frequency") +
  theme(axis.text.x=element_text(angle=90, hjust=1))


#LDA

#install.packages("topicmodels")
#install.packages("quanteda")
#install.packages("tidyverse")
#install.packages("tidytext")
#install.packages("lubridate")
#install.packages("sysfonts")
#install.packages("showtext")
#install.packages("jiebaR")
#install.packages("servr")
#install.packages("ldatuning")
#install.packages("doParallel")
#install.packages("reshape2")
#install.packages("caret")

library(topicmodels)
library(quanteda)
library(tidyverse)
library(lubridate)
library(sysfonts)
#font_add_google("Noto Sans TC", "Noto Sans TC")
library(showtext)
showtext_auto()
library(jiebaR)
library(tidytext)
library(servr)

###################
# dfm_remostwo
# Trimming DFM to reduce training time
docvars(dfm_remostwo, "docname") <- docnames(dfm_remostwo)
dfm_trimmed <- dfm_trim(dfm_remostwo, min_docfreq = 20 , min_count = 50)# 最少出現在20個貼文，最少出現50次
dfm_trimmed
# sparse是很多0的matrix 對於我們沒有幫助 人類的很多話都是很少出現的，表達模式就是如此

# Removing rows that contain all zeros
# 把所有都是0的row刪掉
row_sum <- apply(dfm_trimmed , 1, sum)
dfm_trimmed <- dfm_trimmed[row_sum> 0, ]

# Converting to another format
lda_data <- convert(dfm_trimmed, to = "topicmodels")
lda_data

####################################################################################
## Finding K                                                                      ##
####################################################################################
#220727四萬筆資料，tuning 5小時左右
######################### LDA Tuning ######################### 
library("ldatuning")

ldatuning.result <- FindTopicsNumber(
  lda_data,
  topics = seq(from = 20, to = 120, by = 5),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"), # There are 4 possible metrics: Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"
  method = "Gibbs",
  control = list(seed = 4321),
  verbose = TRUE
)
FindTopicsNumber_plot(ldatuning.result)

######################### Perplexity ######################### 

# Another approach to find K is perplexity: we split the data into 5 parts, 
# train a model using 4 of the 5 parts, then see how well does it predict the held-out one, repeat 5 times.
# Note that this approach can take a long time since we need to train 5 models for each candidate K
# 每次跑資料量計時
# Ivermectin documents: 1957, terms: 3592 用時4142s 69min

library(doParallel)
library(dplyr)
library(reshape2)
library(tidyr)
library(ggplot2)

# Here we do parallelisation to speed up the process.
# 加快速度
cluster <- makeCluster(detectCores(logical = TRUE)-1, outfile = "Log.txt")
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(topicmodels)
})

n <- nrow(lda_data)
burnin <- 1000
iter <- 1000
keep <- 50
folds <- 5
splitfolds <- sample(1:folds, n, replace = TRUE)
candidate_k <- c(10, 20, 30, 40, 50) # candidates for how many topics

clusterExport(cluster, c("lda_data", "burnin", "iter", "keep", "splitfolds", "folds", "candidate_k"))

system.time({
  results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
    k <- candidate_k[j]
    results_1k <- matrix(0, nrow = folds, ncol = 2)
    colnames(results_1k) <- c("k", "perplexity")
    for(i in 1:folds){
      train_set <- lda_data[splitfolds != i , ]
      valid_set <- lda_data[splitfolds == i, ]
      
      fitted <- LDA(train_set, k = k, method = "Gibbs",
                    control = list(burnin = burnin, iter = iter, keep = keep) )
      results_1k[i,] <- c(k, perplexity(fitted, newdata = valid_set))
    }
    print(k)
    return(results_1k)
  }
})
stopCluster(cluster)

# Plotting the results
results_df <- as.data.frame(results)
results_df$istest <- "test"
avg_perplexity <- results_df %>% group_by(k) %>% summarise(perplexity = mean(perplexity))
avg_perplexity$istest <- "avg"
plot_df <- rbind(results_df, avg_perplexity)

ggplot(plot_df, aes(x = k, y = perplexity, group = istest)) +
  geom_point(aes(colour = factor(istest))) +
  geom_line(data = subset(plot_df, istest %in% "avg"), color = "red") +
  ggtitle("5-fold Cross-validation of Topic Modelling") +
  labs(x = "Candidate k", y = "Perplexity") +
  scale_x_discrete(limits = candidate_k) +
  scale_color_discrete(name="Test\\Average",
                       breaks=c("test", "avg"),
                       labels=c("Test", "Average")) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))


####################################################################################
## Fit Optimal Model(s)                                                           ##
####################################################################################
# A good practice would be to fit multiple models from our K search and compare their performance

lda_model_75 <- LDA(lda_data, 75, method="Gibbs")  

get_terms(lda_model_75, k=50)
#k=50有點混淆 它是告訴你出現頻率前50多的字

####################################################################################
##  Visualization                                                                 ##
####################################################################################

######################### Visualize Terms ##############################
library(tidytext)
library(tidyr)

# First we need to extract the beta (probability of a word in a topic)
topics <- tidy(lda_model_75, matrix = "beta")

topics %>%
  group_by(topic) %>%
  top_n(40, beta) %>%
  ungroup() %>%
  arrange(topic, -beta) %>% 
  mutate(term = reorder_within(term, beta, topic)) %>%
  #排列順序 讓beta高的在前面顯示
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()

######################### LDAvis ##############################
library(topicmodels)
library(dplyr)
library(stringi)
library(quanteda)
library(LDAvis)

# LDAvis is an interactive tool to visualise a LDA model
# It is useful for initial exploration, especially the relationship between topics

# There might be a discrepancy between the dfm we feed the function,
# and the dfm actually used for training (some rows might be excluded)
visdfm <- dfm_subset(dfm_trimmed, docname %in% rownames(lda_data))

# A custom function to transform data to json format
topicmodels_json_ldavis <- function(fitted, dfm, dtm){
  # Find required quantities
  phi <- posterior(fitted)$terms %>% as.matrix
  theta <- posterior(fitted)$topics %>% as.matrix
  vocab <- colnames(phi)
  doc_length <- ntoken(dfm)
  
  temp_frequency <- as.matrix(dtm)
  freq_matrix <- data.frame(ST = colnames(temp_frequency),
                            Freq = colSums(temp_frequency))
  rm(temp_frequency)
  # Convert to json
  json_lda <- LDAvis::createJSON(phi = phi, theta = theta,
                                 vocab = vocab,
                                 doc.length = doc_length,
                                 term.frequency = freq_matrix$Freq)
  return(json_lda)
}

#這一行最重要，要選到你要分析的那個model
json_lda <- topicmodels_json_ldavis(lda_model_75, visdfm, lda_data)
serVis(json_lda, out.dir = "LDAvis", open.browser = TRUE)


######################### Topic Proportion per Document ##############################
doc_gamma <- tidy(lda_model_75, matrix = "gamma")

doc_gamma %>%
  filter(document %in% c("text1","text2","text3","text4","text5","text6")) %>% 
  ggplot(aes(factor(topic), gamma, fill = factor(topic))) +
  geom_col() +
  facet_wrap(~ document) +
  labs(x = "topic", y = expression(gamma))

#以上是所有資料的gamma值
#輸出一個檔案備份可以xlookup原始gamma值，當然之後寫個code在r裡面處理完成更好

write.csv(doc_gamma,"gamma.csv")

####################################################################################
##  Validation                                                                    ##
####################################################################################

# Before we do anything, it is a good idea to have a data frame that contains
# all document-level information and the topic for each document
topic_df <- docvars(corpus_iver)
topic_df$doc_name <- docnames(corpus_iver)
topic_df$text <- as.character(corpus_iver)

lda_df <- data.frame(topic = get_topics(lda_model_75), 
                     doc_name = lda_model_75@documents)
topic_df <- left_join(topic_df, lda_df, by = "doc_name")
topic_count <- group_by(topic_df, topic) %>% summarise(topic_count = n())
topic_df <- left_join(topic_df, topic_count, by = "topic")
topic_df$TW_date <- as.Date(topic_df$TW_Time) # 這邊注意date欄位要改名
topic_df$topic <- as.factor(topic_df$topic)

#寫出帶topic的原始檔案
write.csv(topic_df, "data_with_topic.csv")

######################### Accuracy per Topic ##############################

# Accuracy, this is the bare minimum that we should do
validation_df <- topic_df %>% 
  group_by(topic) %>% 
  slice_sample(n = 10, replace = TRUE)
write.csv(validation_df, "topic_validation_random.csv")

# One more step, coding texts into topics and analyse with confusion matrix
validation_sample <- topic_df %>% 
  select(topic, text) %>% 
  group_by(topic) %>% 
  slice_sample(n = 2, replace = TRUE) %>% 
  ungroup()

validation_sample <- validation_sample[sample(1:nrow(validation_sample)), ]
validation_sample$no <- 1:nrow(validation_sample)
validation_coding <- select(validation_sample, no, text)

saveRDS(validation_sample, "validation_sample.rds")
write.csv(validation_sample, "validation_sample.csv", row.names = FALSE)
write.csv(validation_coding, "validation_coding.csv", row.names = FALSE)

# Do this after coding
validation_coded <- read.csv("validation_coding_coded.csv")
validation <- left_join(validation_sample, validation_coded, by = "no")

# Confusion Matrix
library(caret)
validation$code <- factor(validation$code)
validation$topic <- factor(validation$topic, levels = levels(validation$code))
confusionMatrix(validation$topic, validation$code)

############################ LIWC ########################################
library(tidytext)
library(dplyr)
####
#install.packages("devtools")
library(devtools)
#devtools::install_github("quanteda/quanteda.corpora")
#devtools::install_github("kbenoit/quanteda.dictionaries")
library(quanteda.corpora)
library(quanteda.dictionaries)

liwc2015dict <- dictionary(file = "LIWC2015 Dictionary - Internal.dic",
                           format = "LIWC")
liwcanalysis <- liwcalike(corpus_iver, liwc2015dict)
write.csv(liwcanalysis, "liwcanalysis2.csv",row.names = TRUE)

# ttest using result from liwc

topiclab <- read.csv("topiclab.csv")
topiclab$category[1:75]

df_selecvar<-summarize(by_day, 
                       present=mean(`timeorient (time orientation).focuspresent (present focus)`, na.rm = TRUE),
                       past=mean(`timeorient (time orientation).focuspast (past focus)`, na.rm = TRUE),
                       anger=mean(`affect (affect).negemo (negative emotions).anger (anger)`, na.rm = TRUE),
                       certain=mean(`cogproc (cognitive processes).certain (certainty)`, na.rm = TRUE),
                       filler=mean(`cogproc (cognitive processes).certain (certainty)`, na.rm = TRUE),
                       death=mean(`persconc (personal concerns).death (death)`, na.rm = TRUE),
                       money=mean(`persconc (personal concerns).money (money)`, na.rm = TRUE),
                       interaction=mean(interact, na.rm = TRUE),
                       count = n())

ttest <- select(ttest2,
         doc_name,text,topic,TW_date,
         `cogproc (cognitive processes).certain (certainty)`,
        `function (function words).pronoun (pronouns).ipron (impersonal pronouns)`,
        `function (function words).pronoun (pronouns)`,
        `function (function words).adverb (adverbs)`,
        `function (function words).pronoun (pronouns).ppron (personal pronouns).we (we)`,
        `function (function words).pronoun (pronouns).ppron (personal pronouns).they (they)`,
        `affect (affect).negemo (negative emotions).anger (anger)`)

topiclab<-topiclab[c(1:75),]
labeled<-as.list.data.frame(topiclab)
ttest_final <- ttest %>% replace(ttest, list = ttest$topic, values = labeled$category)
ttest_final<-as.data.frame(ttest_final)
