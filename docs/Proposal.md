{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "76a26f57",
   "metadata": {},
   "source": [
    "# Hotel Booking Cancellation Prediction"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "caf606fc",
   "metadata": {},
   "source": [
    "* Prepared for UMBC Data Science Master Degree Capstone by Dr Chaoji (Jay) Wang\n",
    "* Author: Sai Tara Kunduru\n",
    "* GitHub : https://github.com/Tarakunduru\n",
    "* Linkedin : https://www.linkedin.com/in/tarakunduru"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "22b31b1f",
   "metadata": {},
   "source": [
    "## 2. Background"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3d8aba56",
   "metadata": {},
   "source": [
    "### What is it about?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bae1f1cd",
   "metadata": {},
   "source": [
    "The Hotel Booking Cancellation Prediction project leverages historical booking data to predict whether a hotel booking will be canceled. the dataset contains various features related to bookings. The primary objective is to develop a machine learning model for accurate predictions to assist hotel managers in inventory management and pricing strategies. The project will involve data preprocessing, EDA, feature engineering, model training, and deployment. The project aims to improve hotel revenue, reduce operational costs, and enhance customer satisfaction despite the challenges of imbalanced data, missing values, and model integration."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7eb99896",
   "metadata": {},
   "source": [
    "### Why does it matter?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4758aac2",
   "metadata": {},
   "source": [
    "The Hotel Booking Cancellation Prediction project matters because it helps hotels optimize revenue, manage room inventory, improve customer service, and make informed strategic decisions. By accurately predicting cancellations, hotels can adjust pricing, accommodate guests during peak periods, enhance customer satisfaction, and gain a competitive advantage. This project is crucial for reducing financial impact and improving overall business outcomes in the highly competitive hospitality industry."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "569ad3f1",
   "metadata": {},
   "source": [
    "### What are your reasearch questions?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ebc4a325",
   "metadata": {},
   "source": [
    "What features in the booking data are the most significant predictors of booking cancellations?\n",
    "\n",
    "How does the performance of the cancellation prediction model vary across different hotels or hotel chains?\n",
    "\n",
    "Can the model's predictions be used to optimize pricing strategies and improve revenue management?\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6ef732e",
   "metadata": {},
   "source": [
    "## 3. Data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "21b37f9a",
   "metadata": {},
   "source": [
    "* Data Source: https://www.kaggle.com/datasets/mesutyurukcu/hotel-booking-cancellation-prediction\n",
    "* Data Size: 3.41 MB\n",
    "* Data shape: Number of rows = 119210\n",
    "              Number of columns = 30\n",
    "\n",
    "* What does each row represent:\n",
    "* Data Dictionary\n",
    "1. hotel: Type of hotel (e.g., City hotel, Resort hotel).\n",
    "2. is_canceled: Binary value indicating if the booking was canceled (1) or not (0).\n",
    "3. lead_time: Number of days between booking and arrival date.\n",
    "4. arrival_date_year: Year of arrival date.\n",
    "5. arrival_date_month: Month of arrival date.\n",
    "6. arrival_date_week_number: Week number of arrival date.\n",
    "7. arrival_date_day_of_month: Day of arrival date.\n",
    "8. stays_in_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed.\n",
    "9. stays_in_week_nights: Number of weeknights (Monday to Friday) the guest stayed.\n",
    "10. adults: Number of adults in the booking.\n",
    "11. children: Number of children in the booking.\n",
    "12. babies: Number of babies in the booking.\n",
    "13. meal: Type of meal package (e.g., Bed & Breakfast, Half board).\n",
    "14. country: Country of origin of the booking.\n",
    "15. market_segment: Market segment (e.g., Online TA, Offline TA, Direct).\n",
    "16. distribution_channel: Booking distribution channel (e.g., TA/TO, Direct).\n",
    "17. is_repeated_guest: Binary value indicating if the guest is a repeat customer (1) or not (0).\n",
    "18. previous_cancellations: Number of previous bookings that were canceled by the guest.\n",
    "19. previous_bookings_not_canceled: Number of previous bookings that were not canceled by the guest.\n",
    "20. reserved_room_type: Type of room reserved.\n",
    "21. assigned_room_type: Type of room assigned to the guest.\n",
    "22. booking_changes: Number of times the guest made changes to the booking.\n",
    "23. deposit_type: Type of deposit made (e.g., No Deposit, Non-refundable).\n",
    "24. agent: ID of the travel agency that made the booking.\n",
    "25. company: ID of the company that made the booking.\n",
    "26. days_in_waiting_list: Number of days the booking was on the waiting list before confirmation.\n",
    "27. customer_type: Type of customer (e.g., Transient, Contract).\n",
    "28. adr: Average daily rate.\n",
    "29. required_car_parking_spaces: Number of required car parking spaces.\n",
    "30. total_of_special_requests: Number of special requests made by the guest.\n",
    "31. reservation_status: Current status of the reservation (e.g., Check-Out, Canceled).\n",
    "32. reservation_status_date: Date when the last status change occurred.\n",
    "\n",
    "* Which variable/column will be your target/label in your ML model?\n",
    "\n",
    "In the Hotel Booking Cancellation Prediction project, the target variable (also known as the label) will be the is_canceled column. This binary variable indicates whether a booking was canceled (1) or not (0). The goal of the machine learning model will be to predict this variable based on the other features in the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9108bb9a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
