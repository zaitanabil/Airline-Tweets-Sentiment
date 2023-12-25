
# Sentiment Analysis of Airline Tweets: From Data Exploration to Production Implementation

This project focuses on analyzing Twitter data related to major airline companies, gathered from February 2015. The main objectives and processes of the project are:

+ **Data Visualisation:** Analyzing the sentiment distribution across airlines, identifying common reasons for negative sentiments for each airline, and exploring the relationship between sentiment confidence and retweet count.
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

Key insights from this analysis:

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

Observations:

1) Spread of Data: The majority of tweets, regardless of sentiment or confidence level, have low retweet counts. This is a typical pattern seen on social media platforms.
2) Sentiment Confidence: The data points are spread across various levels of sentiment confidence, but there doesn't appear to be a clear trend where higher sentiment confidence correlates with a higher number of retweets.
3) Sentiment Type: The plot includes different sentiment types, but there's no distinct pattern to suggest that tweets of a particular sentiment, or those with higher confidence in that sentiment, are more likely to be retweeted.

### Conclusion
The data visualization analysis of airline-related tweets yielded significant insights. Firstly, there's a clear prevalence of negative sentiments across airlines, signaling a trend where negative experiences are more commonly shared on social media. This highlights the need for airlines to focus on customer service and reputation management.

Secondly, the reasons for negative sentiments vary by airline, with common issues like customer service and flight delays. This information is crucial for airlines to address specific operational challenges.

Lastly, the analysis showed no direct link between the strength of sentiment in a tweet and its retweet count, suggesting other factors influence social media engagement.

In summary, these findings underscore the importance of social media as a tool for understanding customer experiences and shaping airline strategies.

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

### Additional Resources
- [Jupyter Notebook](path/to/jupyter_notebook.ipynb)
- [Data Source](path/to/data_source.csv)
