# Hotel Booking Cancellation Prediction

* Prepared for UMBC Data Science Master Degree Capstone by Dr Chaoji (Jay) Wang
* Author: Sai Tara Kunduru
* GitHub : https://github.com/Tarakunduru
* Linkedin : https://www.linkedin.com/in/tarakunduru
* YouTube : https://www.youtube.com/watch?v=3UKjCkjhFP0

## 2. Background

### What is it about?

The Hotel Booking Cancellation Prediction project leverages historical booking data to predict whether a hotel booking will be canceled. the dataset contains various features related to bookings. The primary objective is to develop a machine learning model for accurate predictions to assist hotel managers in inventory management and pricing strategies. The project will involve data preprocessing, EDA, feature engineering, model training, and deployment. The project aims to improve hotel revenue, reduce operational costs, and enhance customer satisfaction despite the challenges of imbalanced data, missing values, and model integration.

### Why does it matter?

The Hotel Booking Cancellation Prediction project matters because it helps hotels optimize revenue, manage room inventory, improve customer service, and make informed strategic decisions. By accurately predicting cancellations, hotels can adjust pricing, accommodate guests during peak periods, enhance customer satisfaction, and gain a competitive advantage. This project is crucial for reducing financial impact and improving overall business outcomes in the highly competitive hospitality industry.

### What are your reasearch questions?

What features in the booking data are the most significant predictors of booking cancellations?

How does the performance of the cancellation prediction model vary across different hotels or hotel chains?

Can the model's predictions be used to optimize pricing strategies and improve revenue management?


## 3. Data

* Data Source: This dataset is obtained from Kaggle and is related Hotel Booking Cancellation Prediction
* Data Size: 3.41 MB
* Data shape: Number of rows = 119210
              Number of columns = 30

* Data Dictionary
1. hotel: Type of hotel (e.g., City hotel, Resort hotel).
2. is_canceled: Binary value indicating if the booking was canceled (1) or not (0).
3. lead_time: Number of days between booking and arrival date.
4. arrival_date_year: Year of arrival date.
5. arrival_date_month: Month of arrival date.
6. arrival_date_week_number: Week number of arrival date.
7. arrival_date_day_of_month: Day of arrival date.
8. stays_in_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed.
9. stays_in_week_nights: Number of weeknights (Monday to Friday) the guest stayed.
10. adults: Number of adults in the booking.
11. children: Number of children in the booking.
12. babies: Number of babies in the booking.
13. meal: Type of meal package (e.g., Bed & Breakfast, Half board).
14. country: Country of origin of the booking.
15. market_segment: Market segment (e.g., Online TA, Offline TA, Direct).
16. distribution_channel: Booking distribution channel (e.g., TA/TO, Direct).
17. is_repeated_guest: Binary value indicating if the guest is a repeat customer (1) or not (0).
18. previous_cancellations: Number of previous bookings that were canceled by the guest.
19. previous_bookings_not_canceled: Number of previous bookings that were not canceled by the guest.
20. reserved_room_type: Type of room reserved.
21. assigned_room_type: Type of room assigned to the guest.
22. booking_changes: Number of times the guest made changes to the booking.
23. deposit_type: Type of deposit made (e.g., No Deposit, Non-refundable).
24. agent: ID of the travel agency that made the booking.
25. company: ID of the company that made the booking.
26. days_in_waiting_list: Number of days the booking was on the waiting list before confirmation.
27. customer_type: Type of customer (e.g., Transient, Contract).
28. adr: Average daily rate.
29. required_car_parking_spaces: Number of required car parking spaces.
30. total_of_special_requests: Number of special requests made by the guest.
31. reservation_status: Current status of the reservation (e.g., Check-Out, Canceled).
32. reservation_status_date: Date when the last status change occurred.

* Which variable/column will be your target/label in your ML model?

In the Hotel Booking Cancellation Prediction project, the target variable (also known as the label) will be the is_canceled column. This binary variable indicates whether a booking was canceled (1) or not (0). The goal of the machine learning model will be to predict this variable based on the other features in the dataset.

# 4. Exploratory Data Analysis

### 4.1 Data Cleaning

### 4.1.1 Checking and removing duplicates from the Data Set

* Checking the duplicates from the data set 

### 4.1.2 Analysing the data and visualization

* This plot is designed to show the relationship between two variables: whether a hotel booking was canceled (is_canceled) and whether the guest who made the booking is a repeated guest (is_repeated_guest).
* The is_canceled column likely contains binary values (0 or 1) indicating whether each booking was canceled or not, and the is_repeated_guest column also likely contains binary values indicating whether each guest is a repeated guest or not.
* The bars in the plot represent the frequency of each combination of is_canceled and is_repeated_guest values in the data.
 ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/de9731e8-7e3c-435e-89b6-e596390c60d1)

* This boxplot is designed to show the relationship between hotel booking cancellations (is_canceled), lead time (lead_time), and hotel type (hotel). 
* The x-axis represents booking cancellations, the y-axis represents lead time.
* This plot aims to compare lead time distributions for canceled and non-canceled bookings for both resort and city hotels. The title "resort and city hotel outliers" suggests a focus on identifying outliers in these distributions.
   ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/974d31fc-47d5-4720-8456-321fea578864)


* This boxplot examines the relationship between hotel booking cancellations (is_canceled), the number of weekend nights stayed (stays_in_weekend_nights), and the type of hotel (hotel). 
* The x-axis represents booking cancellations, the y-axis represents the number of weekend nights stayed.
* The plot's purpose is to compare the distribution of weekend nights stayed for canceled and non-canceled bookings across different hotel types. The title "resort and city hotel outliers" suggests a focus on identifying outliers in this distribution.
   ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/0d4d7e1c-7c90-4037-ba2e-09910c7ee53e)



* this plot visualize the distribution of total special requests (total_of_special_requests) made by guests in the hotel_data DataFrame. 
* The x-axis of the histogram represents the different counts of special requests, and the y-axis represents the frequency of each count. 
* The seaborn function histplot is used to create the histogram, with x set to 'total_of_special_requests' 
* The title "distribution of total special requests" is set using plt.title(). This plot is intended to provide a visual representation of how frequently different numbers of special requests are made by guests.
  ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/ad025964-3903-4efa-a62b-cb37099cbfbe)

* This plot explore the relationship between reserved room types (reserved_room_type), lead time for booking (lead_time), and hotel type (hotel) in the hotel_data DataFrame.
* The x-axis represents the different room types, while the y-axis represents the lead time for bookings. The catplot function is used with data set to hotel_data, x set to 'reserved_room_type', y set to 'lead_time', and col set to 'hotel'.
* This creates separate plots for each hotel type. The color of the plot points is set to green using the color parameter.
   ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/baae8d8d-8a52-4e1b-b777-b1f22ba265d8)


* This plot generates to visualize the relationship between booking cancellations (is_canceled), the number of days in the waiting list (days_in_waiting_list), and the type of deposit (deposit_type). 
* In the plot, the x-axis represents booking cancellations, the y-axis represents the number of days in the waiting list, and the col parameter is used to create separate plots for each deposit type. 
* y set to 'days_in_waiting_list', x set to 'is_canceled', and col set to 'deposit_type'. All the data points are colored red, as specified by the color parameter.
  ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/1fd9bdbf-b3c3-414a-b3c7-6751e5665771)



* This plot displays the distribution of hotel bookings across different market segments (market_segment) and their cancellation status (is_canceled).
* In the plot, the x-axis represents the market segments, and the y-axis represents the count of bookings. The hue parameter is used to differentiate between canceled and non-canceled bookings with different colors. 
* The plot is further enhanced by rotating the x-axis labels by 65 degrees and aligning them to the right for better readability, as specified by the set_xticklabels method.
 ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/9510e339-512a-4682-a290-88c2b1240126)

# 5. Feature selection methods

## 5.1 Univariate feature selection

* Features in matrix X are scaled using Min-Max scaling with MinMaxScaler().
* Utilizing SelectKBest with chi-squared (chi2) scoring, the code conducts independent feature selection based on significance to the target variable y.
* Feature scores and names are stored in dfscores and dfcolumns DataFrames, concatenated into featureScores for streamlined analysis.
* The nlargest() method extracts and prints the top 10 features by chi-squared scores, providing a concise summary.
* The printed DataFrame with 'columns' and 'Score' aids in identifying influential variables for subsequent modeling. Univariate selection simplifies models and boosts predictive performance by focusing on key features.

   ![image](https://github.com/DATA-606-2023-FALL-THURSDAY/Kunduru_Saitara/assets/143665432/58ca9f33-5a86-4fb8-899c-959014577dfc)

## 5.2 Correlation matrix with heatmap

* The image computes a correlation matrix (data_corr) for the hotel_data, revealing relationships between variables.
* A 12x9-inch heatmap is created using seaborn (sns.heatmap()) with the "RdYlGn" color map, visually representing the correlations.
* The resulting heatmap is displayed using plt.show(), offering an intuitive overview of the dataset's correlation structure.

  ![image](https://github.com/DATA-606-2023-FALL-THURSDAY/Kunduru_Saitara/assets/143665432/5580049f-6fba-48f9-a83d-63ffac4d3fb4)

## 5.3 Top Features

* The code defines top_n as the number of features to display and extracts the top N important features using nlargest() on feat_importances.
* A horizontal bar graph is created using top_features.plot(kind='barh') for visualizing the importance scores of the selected features.
* The resulting bar graph is shown with plt.show(), offering a concise visual summary of the top features.

  ![image](https://github.com/DATA-606-2023-FALL-THURSDAY/Kunduru_Saitara/assets/143665432/f49be094-fc86-48ab-be8e-4eb23f64d9e0)


 # 6. Machine Learning Models

       ![image](https://github.com/DATA-606-2023-FALL-THURSDAY/Kunduru_Saitara/assets/143665432/5169fa55-3047-4e5d-a0d4-ec0cf181a35b)



 ## 6.1 Logistic Regression Model:

* Logistic Regression model initialized and trained on X_train_imp and y_train_imp.
Predictions:
* Predictions made on training and testing data (train_pred, test_pred).
* Confusion matrix and key metrics computed: accuracy, recall, precision, F1-score.
* Training accuracy: accuracy_score(y_train_imp, train_pred).
  Testing accuracy: accuracy_score(y_test_imp, test_pred).
* Numerical values for recall, precision, and F1-score presented.


## 6.2 Decision Tree Model:

* Decision Tree Classifier initialized with entropy criterion and max depth of 10.
   Trained on imputed data (X_train_imp and y_train_imp).
* Predictions made for training and testing data (d_tree_train_pred, d_tree_test_pred).
* Evaluation Metrics:Confusion matrix, recall, and F1-score computed.
* Accuracy Scores: Training accuracy: accuracy_score(y_train_imp, d_tree_train_pred).
  Testing accuracy: accuracy_score(y_test_imp, d_tree_test_pred).

## 6.3 Random Forest Model:

* Random Forest Classifier initialized with 80 decision trees (n_estimators=80).
Trained on imputed data (X_train_imp and y_train_imp).
* Predictions made for training and testing data (rf_train_pred, rf_test_pred).
* Evaluation Metrics: Confusion matrix, recall, and F1-score computed.
* Accuracy Scores:Training accuracy: accuracy_score(y_train_imp, rf_train_pred).
Testing accuracy: accuracy_score(y_test_imp, rf_test_pred).





## Conclusions:

Cancellation count is less for repeated guests

Lead time is less for confirm bookings lead time is less for resort hotels

Stay in weekend nights is equal for both confirm and cancelled cases in both the hotels but the outliers are high in confirmed classes

Distribution of total special requests is left skewed

Lead time is high for room types C,A and D for both the hotels

Deposit type is refundable waiting list days are same but the other categories it is high.

Logistic Regression demonstrated the highest overall performance with the highest accuracy, precision, recall, and F1 score among the three classifiers.

Decision Tree exhibited slightly lower performance compared to Logistic Regression but still provided reasonable accuracy and balanced precision and recall.

Random Forest showed competitive results, balancing precision and recall, providing a good trade-off between the two.

## References:

M. R. H. Subho, M. R. Chowdhury, D. Chaki, S. Islam and M. M. Rahman, "A Univariate Feature Selection Approach for Finding Key Factors of Restaurant Business," 2019 IEEE Region 10 Symposium (TENSYMP), Kolkata, India, 2019, pp. 605-610, doi: 10.1109/TENSYMP46218.2019.8971127.
V. Aggarwal, V. Gupta, P. Singh, K. Sharma and N. Sharma, "Detection of Spatial Outlier by Using Improved Z-Score Test," 2019 3rd International Conference on Trends in Electronics and Informatics (ICOEI), Tirunelveli, India, 2019, pp. 788-790, doi: 10.1109/ICOEI.2019.8862582.
G. König, C. Molnar, B. Bischl and M. Grosse-Wentrup, "Relative Feature Importance," 2020 25th International Conference on Pattern Recognition (ICPR), Milan, Italy, 2021, pp. 9318-9325, doi: 10.1109/ICPR48806.2021.9413090.


