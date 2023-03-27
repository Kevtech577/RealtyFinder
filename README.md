# Real Estate Listings Web App

This is a Python web app that allows users to search for real estate properties in the US and get actual price predictions using the Random Forest Regressor algorithm.

## Table of Contents

- [Introduction](#introduction)
- [Technologies](#technologies)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

## Introduction

The purpose of this project is to provide a user-friendly web app that allows users to search for real estate properties in the US and get actual price predictions using the Random Forest Regressor algorithm. The app has two main functionalities:

1. Search Listings: Users can select a state, city, and number of listings from the sidebar, and the app will display a table with the selected listings. This functionality is implemented using the Realty Mole Property API.

2. Actual Price Prediction Using Random Forest: Users can enter an address in a text input, and the app will use the Random Forest Regressor algorithm to predict the actual price of the property. This functionality is implemented using the Scikit-learn library.

## Technologies
![unnamed](https://user-images.githubusercontent.com/112779801/224235597-07f57acc-6862-4f3c-aec4-5d479b16e72a.png)

This project was built using the following technologies:

- Python
- Streamlit
- Pandas
- Folium
- Scikit-learn
- Realty Mole Property API

## Usage

To run this web app, you need to follow these steps:

1. Clone the repository to your local machine.

    ```bash
    git clone https://github.com/your-username/real-estate-listings-web-app.git
    ```

2. Install the required libraries using pip.

    ```bash
    pip install -r requirements.txt
    ```

3. Set the `API_KEY` and `API_HOST` variables in the `app.py` file with your Realty Mole Property API credentials.

    ```python
    API_KEY = 'YOUR_API_KEY'
    API_HOST = 'realty-mole-property-api.p.rapidapi.com'
    ```

4. Run the app.

    ```bash
    streamlit run app.py
    ```

5. Open your browser and go to http://localhost:8501.

## Calculations for Investors

In addition to providing a search functionality and price predictions, the app also provides calculations for investors to help them evaluate the potential profitability of a property. These calculations include:

- Monthly mortgage payment: calculated using the total loan amount, interest rate, and loan term.
- Monthly rent: calculated using the rental income and vacancy rate.
- Monthly expenses: calculated as a percentage of the purchase price, including property taxes, insurance, and maintenance costs.
- Cash flow: calculated as the difference between the monthly rent and monthly expenses.
- Cap rate: calculated as the net operating income divided by the purchase price.
- Cash-on-cash return: calculated as the annual cash flow divided by the total cash invested (including the down payment and closing costs).

These calculations are displayed in a table for each property, along with the predicted price and other property details.

## Future Improvements

There are several improvements that could be made to this project:

- Add more search criteria, such as price range and property type.
- Use a different algorithm to improve the price prediction accuracy.
- Implement a database to store the search results and the predicted prices. 

[Real Estate UI](https://bensmook-realtychain-real-app-trx0et.streamlit.app/)
