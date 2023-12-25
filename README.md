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
![Sentiment Distribution](https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Distribution%20of%20Sentiments%20Across%20Different%20Airlines.png)

This illustration show us the prevalence of positive, neutral, and negative sentiments in tweets about different airlines. A visual plot highlights the sentiment distribution, revealing a predominant presence of negative sentiments. Variations among airlines and specific trends are evident, indicating different customer satisfaction levels.

### Analyzing Negative Tweet Causes
![Negative Tweet Reasons](https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Most%20Common%20Reasons%20for%20Negative%20Sentiments%20by%20Airline.png)

Focusing on negative tweets, this analysis identifies primary reasons for dissatisfaction, such as service issues or flight delays. A heatmap illustrates these factors across airlines, revealing unique patterns and frequent complaints.

### Sentiment Confidence vs. Retweet Analysis
![Sentiment Confidence vs Retweet Count](https://github.com/zaitanabil/Airline-Tweets-Sentiment/blob/main/Relationship%20Between%20Sentiment%20Confidence%20and%20Retweet%20Count.png)

The relationship between sentiment confidence in tweets and their retweet frequency is explored. A scatter plot is used to analyze this correlation, showing that while sentiment confidence varies, it doesn't necessarily predict retweet likelihood.

### Conclusion
The analysis underscores the predominance of negative sentiments in airline-related tweets, stressing the importance of customer service. It highlights the need for targeted improvements based on specific complaints. Additionally, the lack of a clear link between sentiment strength and social media engagement suggests the influence of other factors in online interactions. This research emphasizes the role of social media in understanding customer experiences and shaping airline strategies.

## Part 2 - Sentiment Analysis Model

### Overview
This section details the creation and refinement of a machine learning model designed to predict the sentiment of tweets related to airlines. The process includes data preprocessing, model implementation, iterative refinement, and evaluation.

### 1. Model Implementation

#### Objective
Develop a machine learning model to predict the sentiment (`airline_sentiment`) of a tweet based on its contents.

#### Methodology
To address this objective, we explored various machine learning classifiers, each with their unique strengths and suited for different types of data distributions. Our approach was to compare and select the model that best fits our data and prediction goals. The following classifiers were considered, along with their respective hyperparameter grids for tuning:

- **Multinomial Naive Bayes**: Ideal for text classification with discrete features. Hyperparameters: `classifier__alpha` values of [0.1, 1, 10].

- **Bernoulli Naive Bayes**: Suitable for making predictions from binary feature vectors. Hyperparameters: `classifier__alpha` values of [0.1, 1, 10].

- **Logistic Regression**: A robust classifier that works well for binary classification problems. Hyperparameters: `classifier__C` values of [0.1, 1, 10] and `classifier__max_iter` values of [100, 200].

- **Support Vector Classifier (SVC)**: Effective in high dimensional spaces, especially useful for text classification tasks. Hyperparameters: `classifier__C` values of [0.1, 1, 10] and `classifier__kernel` options of ['linear', 'rbf', 'poly'].

- **Random Forest Classifier**: A strong classifier that uses ensemble learning and is less prone to overfitting. Hyperparameters: `classifier__n_estimators` values of [100, 200] and `classifier__max_depth` values of [10, 20, None].

The selection of these classifiers was driven by their widespread use and proven effectiveness in sentiment analysis tasks. Our methodology involves using a pipeline to preprocess text data and grid search to find the best hyperparameters for each model. Libraries and frameworks used in this process include scikit-learn, NLTK for text processing, and Pandas for data manipulation.

### 2. Data Preprocessing

#### Overview
Data preprocessing is a critical step in any machine learning project. It involves preparing and cleaning the raw data to make it suitable for a machine learning model. This process can include handling missing values, normalizing data, tokenizing text, and other steps to enhance the quality of the data. For our sentiment analysis model, the preprocessing steps focus primarily on text data, ensuring it is clean, standardized, and structured in a way that the model can effectively learn from it.

#### Preprocessing Steps
- **Step 1: Column Deletion Based on Missing Values and Relevance**
  - Deleted negativereason_gold and airline_sentiment_gold due to a high percentage of missing values (approx. 99.78% and 99.72%, respectively).
  - Removed tweet_coord, tweet_location, and user_timezone considering many users use VPNs, making these locations unreliable.
  - Dropped tweet_created and name as they were not directly relevant to sentiment analysis.
- **Step 2: Merging Negative Reason with Text**
  - For tweets with a specified negative reason, concatenated this reason with the original tweet text to provide more context to the model. This step enriches the feature set for negative sentiments.
- **Step 3: Text Cleaning**
  - Converted text to lowercase to maintain uniformity.
  - Removed URLs, user mentions (@), hashtags (#), punctuations, and numbers to focus on the textual content.
  - Applied these cleaning steps to create a basic_cleaned_text column.
- **Step 4: Sentiment Conversion**
  - Converted the airline_sentiment from text to a numerical format, making it suitable for the model to process. Assigned 'negative' as 0, 'neutral' as 1, and 'positive' as 2. This step is crucial for the model to perform numerical computations on the target variable.

The preprocessing steps were applied meticulously to ensure that the dataset is optimized for training a sentiment analysis model. By cleaning and restructuring the text data, and converting categorical labels into numerical ones, the data is made ready for the next stages of model training.

### 3. Model Refinement

#### Initial Model
The initial model, based on Multinomial Naive Bayes, demonstrated a decent performance with an accuracy of 78.76%. However, it showed limitations in effectively classifying neutral and positive sentiments, as indicated by lower recall and precision scores for these classes.

#### Refinement Process
The refinement process involved experimenting with different classifiers and hyperparameter tuning. This approach was aimed at improving the model's ability to accurately classify sentiments across all categories, particularly focusing on enhancing the performance for neutral and positive classes.

#### Refinement Results
- **Iteration 1: Bernoulli Naive Bayes**
  - Changes: Employed BernoulliNB with a tuned alpha value.
  - Results: Accuracy improved to 86.89%, with significant gains in precision and recall across all classes.
- **Iteration 2: Logistic Regression**
  - Changes: Switched to Logistic Regression with optimized C and max_iter parameters.
  - Results: A notable jump in accuracy to 91.73%. The model showed excellent precision, especially for negative sentiments.
- **Iteration 3: Support Vector Classifier (SVC)**
  - Changes: Utilized SVC with a linear kernel and adjusted C parameter.
  - Results: Achieved the highest accuracy of 92.79%. The model displayed strong precision and recall, indicating effective classification across different sentiments.
- **Iteration 4: RandomForest Classifier**
  - Changes: Implemented RandomForest with tuned n_estimators and max_depth.
  - Results: Delivered an accuracy of 92.38%, with robust precision and recall, particularly for negative sentiments.

### 4. Model Evaluation

#### Evaluation Metric
The chosen evaluation metrics were accuracy, precision, recall, and F1-score. Accuracy provided an overall effectiveness of the model, while precision and recall offered insights into its performance across individual classes. The F1-score, being the harmonic mean of precision and recall, gave a balanced measure of the model's accuracy and completeness, essential in a multi-class classification scenario like sentiment analysis.

#### Model Performance
The final model iteration using SVC demonstrated outstanding performance with an accuracy of 92.79%, showcasing its ability to accurately classify sentiments in tweets. The precision was particularly high for negative sentiments, indicating the model's effectiveness in correctly identifying negative tweets. The recall and F1-scores across all classes were also commendable, signifying a well-balanced model.

#### Improvements
Each model iteration brought incremental improvements. The shift from Naive Bayes to more complex models like Logistic Regression and SVC significantly enhanced the model's ability to understand and classify sentiments more accurately. Hyperparameter tuning played a crucial role in optimizing each model's performance, leading to a consistent increase in accuracy and other metrics.

### Conclusion
The development of the sentiment analysis model was a methodical process that involved experimenting with various algorithms and fine-tuning hyperparameters. Starting from a baseline Naive Bayes model, the project evolved through several iterations, each enhancing the model's ability to classify sentiments with higher accuracy. The final model, an SVC, emerged as the most effective, striking a balance between high accuracy and robust performance across all sentiment classes.

## Part 3 - Path to Production

### Training the Model on an Ongoing Basis
In a production environment, it is essential to keep the machine learning model updated with the latest trends and linguistic nuances. This ongoing training can be achieved in several ways:

- Data Collection Pipeline: Establish a pipeline for continuously collecting and preprocessing new data. This could involve integrating APIs that fetch new tweet data periodically.
- Model Retraining Strategy: Implement a strategy for periodic retraining of the model. This could be time-based (e.g., monthly) or triggered by certain criteria (e.g., significant changes in data distribution).
- Version Control of Models: Use model versioning to track and manage different versions of the trained model. This allows for easy rollback to previous versions if a new model performs poorly.

### Ensuring Data Quality
High-quality data is crucial for the model's performance. To ensure data quality:

- Validation Checks: Implement checks for data completeness, consistency, and accuracy. This might include verifying the format, checking for missing values, and detecting outliers.
- Data Cleaning Procedures: Develop robust data cleaning procedures tailored to the specific needs of the data. For tweets, this might include removing spam, filtering irrelevant content, and handling slang or abbreviations.
- Feedback Loop: Establish a feedback system where incorrect predictions can be reported and analyzed. This feedback can be used to identify and rectify data quality issues.

### Monitoring Model's Performance
Continuous monitoring of the model's performance in production is essential to detect and address any issues promptly:

- Performance Metrics: Regularly evaluate the model using relevant metrics (accuracy, precision, recall, F1-score). Set up alerts for significant deviations.
- A/B Testing: Use A/B testing to compare the performance of different model versions or updates before fully deploying them.
- User Feedback: Monitor user feedback for insights into the model's real-world performance. User interactions can provide valuable information on the model's strengths and weaknesses.

### Making the Model Available as Part of a Web App
To integrate the model into a web application:

- API Development: Develop a REST API for the model, allowing other services to send data to the model and receive predictions in response. Frameworks like Flask or FastAPI can be used for this purpose.
- Scalability and Load Balancing: Ensure the infrastructure can handle varying loads. Utilize cloud services with auto-scaling capabilities and load balancers.
- Security Considerations: Implement authentication and encryption to ensure data security, especially if sensitive data is being processed.
- User Interface: For the Streamlit app, focus on user experience by designing an intuitive and responsive interface. Provide clear instructions and feedback to the users.


By addressing these aspects, the machine learning model can be effectively deployed, maintained, and utilized in a production environment, ensuring it remains reliable, accurate, and valuable to users.

### Additional Resources
- [Jupyter Notebook](Nabil_ZAITA_Test/code.ipynb)
