# Sentiment Analysis of Airline Tweets: From Data Exploration to Production Implementation

This project focuses on analyzing Twitter data related to major airline companies, gathered from February 2015. The main objectives and processes of the project are:

+ **Data Visualisation:** Analyzing the sentiment distribution across airlines, identifying common reasons for negative sentiments for each airline, and exploring the relationship between sentiment confidence and retweet count.
+ **Sentiment Analysis Model:** Building a machine learning model to predict tweet sentiments, involving data preprocessing, model refinement, and evaluation.
+ **Path to Production**

## Initial Data Overview

Here is a quick overview of the data structure used in our analysis:
| tweet_id         | airline_sentiment | airline_sentiment_confidence | negativereason | negativereason_confidence | airline        | airline_sentiment_gold | name     | negativereason_gold | retweet_count | text                                                        | tweet_coord | tweet_created                | tweet_location | user_timezone               |
|------------------|-------------------|------------------------------|----------------|--------------------------|----------------|------------------------|----------|---------------------|---------------|------------------------------------------------------------|-------------|------------------------------|----------------|-----------------------------|
| 570306133677760513 | neutral           | 1.0000                       | NaN            | NaN                      | Virgin America | NaN                    | cairdin  | NaN                 | 0             | @VirginAmerica What @dhepburn said.                         | NaN         | 2015-02-24 11:35:52 -0800    | NaN            | Eastern Time (US & Canada) |
| 570301130888122368 | positive          | 0.3486                       | NaN            | 0.0000                   | Virgin America | NaN                    | jnardino | NaN                 | 0             | @VirginAmerica plus you've added commercials t...           | NaN         | 2015-02-24 11:15:59 -0800    | NaN            | Pacific Time (US & Canada) |
| 570301083672813571 | neutral           | 0.6837                       | NaN            | NaN                      | Virgin America | NaN                    | yvonnalynn | NaN               | 0             | @VirginAmerica I didn't today... Must mean I n...           | NaN         | 2015-02-24 11:15:48 -0800    | Lets Play      | Central Time (US & Canada) |
| 570301031407624196 | negative          | 1.0000                       | Bad Flight     | 0.7033                   | Virgin America | NaN                    | jnardino | NaN                 | 0             | @VirginAmerica it's really aggressive to blast...           | NaN         | 2015-02-24 11:15:36 -0800    | NaN            | Pacific Time (US & Canada) |
| 570300817074462722 | negative          | 1.0000                       | Can't Tell     | 1.0000                   | Virgin America | NaN                    | jnardino | NaN                 | 0             | @VirginAmerica and it's a really big bad thing...           | NaN         | 2015-02-24 11:14:45 -0800    | NaN            | Pacific Time (US & Canada) |


## Part 1 - Data Visualisation

In this part, we focus on visualizing and analyzing the sentiment distribution related to various airline companies. The following key areas are explored:

- Distribution of Sentiment Across Airlines
- Analysis of Negative Tweet Reasons by Airline
- Relationship Between Sentiment Confidence and Retweet Count

Each analysis is supported by appropriate visualizations and insights derived from the data.

### Sentiment Distribution Overview
<div align="center">
    <img src="https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Distribution%20of%20Sentiments%20Across%20Different%20Airlines.png" width="500">
</div>

This illustration show us the prevalence of positive, neutral, and negative sentiments in tweets about different airlines. A visual plot highlights the sentiment distribution, revealing a predominant presence of negative sentiments. Variations among airlines and specific trends are evident, indicating different customer satisfaction levels.

### Analyzing Negative Tweet Causes
<div align="center">
    <img src="https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Most%20Common%20Reasons%20for%20Negative%20Sentiments%20by%20Airline.png" width="500">
</div>

Focusing on negative tweets, this analysis identifies primary reasons for dissatisfaction, such as service issues or flight delays. A heatmap illustrates these factors across airlines, revealing unique patterns and frequent complaints.

### Sentiment Confidence vs. Retweet Analysis
<div align="center">
    <img src="https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Relationship%20Between%20Sentiment%20Confidence%20and%20Retweet%20Count.png" width="500">
</div>

The relationship between sentiment confidence in tweets and their retweet frequency is explored. A scatter plot is used to analyze this correlation, showing that while sentiment confidence varies, it doesn't necessarily predict retweet likelihood.

### Conclusion
The analysis underscores the predominance of negative sentiments in airline-related tweets, stressing the importance of customer service. It highlights the need for targeted improvements based on specific complaints. Additionally, the lack of a clear link between sentiment strength and social media engagement suggests the influence of other factors in online interactions. This research emphasizes the role of social media in understanding customer experiences and shaping airline strategies.

## Part 2 - Sentiment Analysis Model

In this section, I delve into the development and optimization of machine learning models for predicting sentiments in airline-related tweets. This part encompasses data preprocessing, model development, iterative enhancement, and performance assessment.

### 1. Model Implementation

#### Objective
Craft a machine learning model to determine the sentiment of a tweet (positive, neutral, negative) based on its content.

#### Methodology
My approach involved evaluating various machine learning algorithms to find the best fit. I considered:

- Multinomial Naive Bayes: Excellent for text with discrete features.
- Logistic Regression: Effective for binary classification tasks.
- Support Vector Classifier (SVC): Suitable for high-dimensional text data.
- Random Forest Classifier: Great for avoiding overfitting.

These models were chosen for their established effectiveness in text-based sentiment analysis. I used a combination of scikit-learn, NLTK, and Pandas for data processing and model tuning.

### 2. Data Preprocessing

Data preparation is crucial for machine learning. My focus was on refining text data for optimal model learning.

#### Preprocessing Steps
- **Step 1: Column Deletion Based on Missing Values and Relevance**

<div align="center">
    <img src="https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Data%20Quality%20Assessment%20Table.png" width="500">
</div>

  - Deleted negativereason_gold and airline_sentiment_gold due to a high percentage of missing values (approx. 99.78% and 99.72%, respectively).
  - Removed tweet_coord, tweet_location, and user_timezone considering many users use VPNs, making these locations unreliable.
  - Dropped tweet_created and name as they were not directly relevant to sentiment analysis.

<div align="center">
    <img src="https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Cleaned%20data.png" width="700">
</div>

- **Step 2: Merging Negative Reason with Text**
  - For tweets with a specified negative reason, concatenated this reason with the original tweet text to provide more context to the model. This step enriches the feature set for negative sentiments.

<div align="center">
    <img src="https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Merging%20Screenshot.png" width="500">
</div>
 
- **Step 3: Text Cleaning**
  - Converted text to lowercase to maintain uniformity.
  - Removed URLs, user mentions (@), hashtags (#), punctuations, and numbers to focus on the textual content.
  - Applied these cleaning steps to create a basic_cleaned_text column.
- **Step 4: Sentiment Conversion**
  - Converted the airline_sentiment from text to a numerical format, making it suitable for the model to process. Assigned 'negative' as 0, 'neutral' as 1, and 'positive' as 2. This step is crucial for the model to perform numerical computations on the target variable.

These steps ensured a well-structured and clean dataset, ready for model training.

### 3. Model Refinement

#### Initial Model
The initial model, based on Multinomial Naive Bayes, demonstrated a decent performance with an accuracy of 78.76%. However, it showed limitations in effectively classifying neutral and positive sentiments, as indicated by lower recall and precision scores for these classes.

<div align="center">
    <img src="https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/MultinomialNB.png" height="250">
</div>

#### Enhancement Stages
- **Logistic Regression**
  - Changes: Employed Logistic Regression with optimized C and max_iter parameters.
  - Results: A notable jump in accuracy to 91.73%. The model showed excellent precision, especially for negative sentiments.
<div align="center">
    <img src="https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/LogisticRegression.png" height="250">
</div>

- **Support Vector Classifier (SVC)**
  - Changes: Utilized SVC with a linear kernel and adjusted C parameter.
  - Results: Achieved the highest accuracy of 92.79%. The model displayed strong precision and recall, indicating effective classification across different sentiments.
<div align="center">
    <img src="https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/SVC.png" height="250">
</div>

- **RandomForest Classifier**
  - Changes: Implemented RandomForest with tuned n_estimators and max_depth.
  - Results: Delivered an accuracy of 92.25%, with robust precision and recall, particularly for negative sentiments.
<div align="center">
    <img src="https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/RandomForestClassifier.png" height="250">
</div>

### 4. Model Evaluation

#### Evaluation Metric
For assessing the model's performance, I selected metrics such as accuracy, precision, recall, and F1-score. Accuracy serves as a measure of the model's overall effectiveness. Precision and recall shed light on its ability to perform accurately across different classes. The F1-score, which is the harmonic mean of precision and recall, provides a comprehensive view of both the model's precision and its thoroughness, which is particularly important in multi-class scenarios like sentiment analysis.

Additionally, I've incorporated 'Training and Evaluation Time' as a metric. This will aid in determining the most efficient classifier when time is a critical factor in the decision-making process.

#### Model Performance
The SVC model showcased exceptional precision and accuracy, notably in pinpointing negative sentiments. It also maintained balanced recall and F1-scores, indicating its overall efficiency. Despite this, the SVC takes about twice as long to process as the RandomForest model, even with comparable accuracy levels. In scenarios where time isn't a critical factor, the SVC would be the ideal choice. However, with an anticipated surge in data volume, which could extend processing times further, the faster RandomForest model might be a more suitable option due to its similar accuracy but quicker execution.

#### Improvements
<div align="center">
    <img src="https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/hyperparameter.png" width="700">
</div>

Each model iteration brought noticeable improvements. The evolution from simpler Naive Bayes to more intricate models, coupled with hyperparameter tuning, resulted in consistently better performance.

### Conclusion
Developing this sentiment analysis model was an iterative process, starting from a basic Naive Bayes to the highly effective SVC model. This journey highlighted the importance of algorithm selection and fine-tuning in creating a model that accurately discerns sentiments in tweets, with the final SVC model striking a perfect balance in performance.

## Part 3 - Path to Production

### Ongoing Model Training

To ensure our sentiment analysis model remains current and accurate, I propose a distinctive strategy for integrating fresh data:
- **Creation of a Data Integration API:** I'll design a specialized API that includes a unique feature, termed 'Data Integration'. This functionality will be activated via a user-friendly interface, specifically a button labeled 'Add new data'.
- **Automated Data Assimilation Process:** Upon activation, this feature will trigger a backend mechanism tailored to seamlessly blend the incoming data with our pre-existing dataset. The focus will be on critical data aspects like sentiment values, tweet text (and potentially the negative sentiment reasons when applicable).
- **Intelligent Model Update Protocol:** Concurrent with the data update, the system will autonomously initiate the retraining process for our chosen analytical model, in this instance, the Support Vector Classifier (SVC). This process is designed to ensure the model evolves and adapts in line with the new data characteristics.
- **Continuous Learning and Adaptation:** This approach embodies a dynamic learning environment for our model, ensuring it stays relevant and maintains high precision in sentiment analysis as new data patterns emerge.

### Ensuring Data Quality
- **Pre-Integration Validation:** Prior to incorporating new data through the API, implement a validation system. This involves checking the incoming data for accuracy, consistency, and format. It's essential to ensure that the data aligns with the existing dataset's structure and quality standards.
- **Anomaly Detection:** Integrate an anomaly detection mechanism within the data assimilation process. This helps in identifying and addressing outliers or abnormal data points that could skew the model's learning.

### Monitoring Model Performance
- **Performance Metrics:** Regularly evaluate the model using key metrics like accuracy, precision, recall, and F1-score.
- **Alert Systems:** Implement alert systems for performance deviations, enabling quick response to potential issues.

### Model Integration into Web Applications
- **API Development:** Develop an API, facilitating seamless integration of the model into web applications.
- **Load Management:** Ensure the model's scalability and robustness under varying loads, utilizing cloud services and load balancing techniques.
- **Security Measures:** Implement security protocols, particularly for data encryption and access control, to protect sensitive information processed by the model.

### Additional Resources
- [Jupyter Notebook](Nabil_ZAITA_Test/code.ipynb)
