# Hotel Booking Cancellation Prediction

* Prepared for UMBC Data Science Master Degree Capstone by Dr Chaoji (Jay) Wang
* Author: Sai Tara Kunduru
* GitHub : https://github.com/Tarakunduru
* Linkedin : https://www.linkedin.com/in/tarakunduru

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

* What does each row represent:
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


