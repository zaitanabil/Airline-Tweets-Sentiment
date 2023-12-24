
# Sentiment Analysis of Airline Tweets: From Data Exploration to Production Implementation

This project focuses on analyzing Twitter data related to major airline companies, gathered from February 2015. The main objectives and processes of the project are:

+ **Data Exploration:** Analyzing the sentiment distribution across airlines, identifying common reasons for negative sentiments for each airline, and exploring the relationship between sentiment confidence and retweet count.
+ **Sentiment Analysis Model:** Building a machine learning model to predict tweet sentiments, involving data preprocessing, model refinement, and evaluation.
+ **Path to Production:** Theoretical discussion on deploying the model in a production environment, covering ongoing training, data quality assurance, performance monitoring, and integration into web services.

## Part 1 - Data Visualisation

### Overview
In this part, we focus on visualizing and analyzing the sentiment distribution within a dataset of tweets related to various airline companies. The following key areas are explored:

- Distribution of Sentiment Across Airlines
- Analysis of Negative Tweet Reasons by Airline
- Relationship Between Sentiment Confidence and Retweet Count

Each analysis is supported by appropriate visualizations and insights derived from the data.

### Distribution of Sentiment Across Airlines

#### Objective
Compare the distribution of sentiments (positive, neutral, negative) across different airlines.

#### Approach
The approach involved aggregating and analyzing a dataset comprising tweets related to different airlines. These tweets were already categorized into three sentiment classes: positive, neutral, and negative. The primary focus was on quantitatively assessing the distribution of these sentiments across various airlines. To achieve this, a count plot was created using data visualization tools, offering a clear and comparative view of sentiment distribution for each airline. This visualization method was chosen for its effectiveness in displaying categorical data and enabling easy comparison across different groups. The plot distinctly marked each sentiment category with different colors, facilitating an immediate visual grasp of the predominant sentiment for each airline and across the sector.

#### Results
![Sentiment Distribution](https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Distribution%20of%20Sentiments%20Across%20Different%20Airlines.png)

The visualization shows the distribution of sentiments (positive, neutral, negative) across different airlines.
1) Negative Sentiment Prevalence: Negative sentiments still dominate for most airlines, indicating a trend where customers are more likely to express dissatisfaction on social platforms.
2) Variation Among Airlines: There's a noticeable variation in the volume and distribution of sentiments across different airlines, suggesting differing levels of customer satisfaction or public perception.
3) Airline-Specific Trends: Specific airlines show different patterns in sentiment distribution, which could be indicative of their service quality, customer experience, or public relations effectiveness.

### Analysis of Negative Tweet Reasons by Airline

#### Objective
Identify the most common reasons for negative sentiments towards each airline.

#### Approach
To conduct this analysis, the dataset, which consisted of tweets related to different airlines and their associated sentiments, was filtered to focus solely on tweets with negative sentiments. The reasons for these negative sentiments were already categorized in the dataset under various labels such as "Customer Service Issue", "Late Flight", "Can't Tell", and others.

A heatmap was chosen as the primary visualization tool for this analysis. This type of visualization effectively displays the frequency of each negative reason across different airlines, enabling an easy comparison. The heatmap assigns varying color intensities to different values, providing a clear visual representation of the most common complaints for each airline. The intensity of the color in the heatmap correlates with the frequency of tweets for each negative reason, allowing for quick identification of the most pressing issues for each airline.

#### Results
![Negative Tweet Reasons](https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Most%20Common%20Reasons%20for%20Negative%20Sentiments%20by%20Airline.png)

The heatmap visualizes the most common reasons for negative sentiments towards each airline.
**Key insights from this analysis:**
1) Reasons for Negative Sentiments: Various reasons like "Customer Service Issue", "Late Flight", "Can't Tell", etc., are common across different airlines. The count of tweets for each reason is indicated by the numbers in the heatmap.
2) Airline-Specific Issues: Each airline has a unique pattern of negative reasons. Some airlines may have higher complaints in certain areas compared to others.
3) Comparative Analysis: By comparing the heatmap across airlines, you can identify which issues are more prevalent for specific airlines. This can guide targeted improvements in customer service or operations.

### Relationship Between Sentiment Confidence and Retweet Count

#### Objective
Examine if tweets with higher sentiment confidence are more likely to be retweeted.

#### Approach
The methodology involved analyzing a dataset of tweets concerning various airlines, each annotated with a sentiment (positive, neutral, negative) and a corresponding confidence level for that sentiment. The key variable of interest was the number of retweets each tweet received.

A scatter plot was employed as the visualization technique for this analysis. This type of plot is adept at revealing correlations or patterns between two quantitative variables. In the plot, each tweet was represented as a point, positioned according to its sentiment confidence and retweet count. Different colors were used to differentiate between the sentiment categories. This visualization approach allowed for an immediate visual appraisal of any potential relationship between sentiment confidence and the propensity of a tweet to be retweeted.

#### Results
![Sentiment Confidence vs Retweet Count](https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Relationship%20Between%20Sentiment%20Confidence%20and%20Retweet%20Count.png)

The scatter plot above illustrates the relationship between airline sentiment confidence and retweet count, with different sentiments (positive, neutral, negative) indicated by different colors.
**Observations:**
1) Spread of Data: The majority of tweets, regardless of sentiment or confidence level, have low retweet counts. This is a typical pattern seen on social media platforms.
2) Sentiment Confidence: The data points are spread across various levels of sentiment confidence, but there doesn't appear to be a clear trend where higher sentiment confidence correlates with a higher number of retweets.
3) Sentiment Type: The plot includes different sentiment types, but there's no distinct pattern to suggest that tweets of a particular sentiment, or those with higher confidence in that sentiment, are more likely to be retweeted.

### Conclusion
[Provide a brief overall conclusion of the data visualization part, summarizing the key insights and their implications.]

### Additional Resources
- [Jupyter Notebook](path/to/jupyter_notebook.ipynb)
- [Data Source](path/to/data_source.csv)

## Part 2 - Sentiment Analysis Model

## Part 3 - Path to Production
