import streamlit as st
import json
import numpy as np
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, max_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error



state_city_dict = {
'AL': ['Birmingham', 'Montgomery', 'Mobile'],
'AK': ['Anchorage', 'Fairbanks', 'Juneau'],
'AZ': ['Phoenix', 'Tucson', 'Mesa'],
'AR': ['Little Rock', 'Fort Smith', 'Fayetteville'],
'CA': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento', 'San Jose'],
'CO': ['Denver', 'Colorado Springs', 'Boulder', 'Aurora'],
'CT': ['Hartford', 'New Haven', 'Stamford', 'Bridgeport'],
'DE': ['Wilmington', 'Dover', 'Newark'],
'FL': ['Miami', 'Orlando', 'Tampa', 'Jacksonville', 'Tallahassee'],
'GA': ['Atlanta', 'Savannah', 'Augusta', 'Athens'],
'HI': ['Honolulu', 'Hilo', 'Kailua'],
'ID': ['Boise', 'Idaho Falls', 'Pocatello', 'Nampa'],
'IL': ['Chicago', 'Springfield', 'Peoria', 'Rockford'],
'IN': ['Indianapolis', 'Fort Wayne', 'Evansville', 'Bloomington'],
'IA': ['Des Moines', 'Cedar Rapids', 'Davenport', 'Sioux City'],
'KS': ['Wichita', 'Kansas City', 'Topeka', 'Manhattan'],
'KY': ['Louisville', 'Lexington', 'Frankfort', 'Bowling Green'],
'LA': ['New Orleans', 'Baton Rouge', 'Lafayette', 'Shreveport'],
'ME': ['Portland', 'Bangor', 'Augusta'],
'MD': ['Baltimore', 'Annapolis', 'Rockville', 'Frederick'],
'MA': ['Boston', 'Worcester', 'Springfield', 'Cambridge'],
'MI': ['Detroit', 'Grand Rapids', 'Ann Arbor', 'Lansing'],
'MN': ['Minneapolis', 'St. Paul', 'Duluth', 'Rochester'],
'MS': ['Jackson', 'Biloxi', 'Hattiesburg', 'Meridian'],
'MO': ['Kansas City', 'St. Louis', 'Springfield', 'Columbia'],
'MT': ['Billings', 'Missoula', 'Great Falls', 'Bozeman'],
'NE': ['Omaha', 'Lincoln', 'Bellevue', 'Grand Island'],
'NV': ['Las Vegas', 'Reno', 'Henderson', 'North Las Vegas'],
'NH': ['Manchester', 'Nashua', 'Concord', 'Dover'],
'NJ': ['Newark', 'Jersey City', 'Trenton', 'Atlantic City'],
'NM': ['Albuquerque', 'Santa Fe', 'Las Cruces', 'Roswell'],
'NY': ['New York City', 'Buffalo', 'Rochester', 'Syracuse'],
'NC': ['Charlotte', 'Raleigh', 'Greensboro', 'Wilmington'],
'ND': ['Fargo', 'Bismarck', 'Grand Forks', 'Minot'],
'OH': ['Columbus', 'Cleveland', 'Cincinnati', 'Toledo'],
'OK': ['Oklahoma City', 'Tulsa', 'Norman', 'Broken Arrow'],
'OR': ['Portland', 'Salem', 'Eugene', 'Gresham'],
'PA': ['Philadelphia', 'Pittsburgh', 'Allentown', 'Erie'],
'RI': ['Providence', 'Warwick', 'Cranston', 'Pawtucket'],
'SC': ['Charleston', 'Columbia', 'Myrtle Beach', 'Greenville'],
'SD': ['Sioux Falls', 'Rapid City', 'Aberdeen', 'Brookings'],
'TN': ['Nashville', 'Memphis', 'Knoxville', 'Chattanooga'],
'TX': ['Houston', 'Dallas', 'Austin', 'San Antonio', 'Fort Worth'],
'UT': ['Salt Lake City', 'Provo', 'Orem', 'West Jordan'],
'VT': ['Burlington', 'Essex', 'Montpelier', 'Rutland'],
'VA': ['Richmond', 'Virginia Beach', 'Norfolk', 'Arlington'],
'WA': ['Seattle', 'Spokane', 'Tacoma', 'Vancouver'],
'WV': ['Charleston', 'Huntington', 'Parkersburg', 'Wheeling'],
'WI': ['Milwaukee', 'Madison', 'Green Bay', 'Kenosha'],
'WY': ['Cheyenne', 'Casper', 'Laramie', 'Gillette']
}

load_dotenv()

API_KEY = os.getenv('X_RapidAPI_Key')
API_HOST = os.getenv('X_RapidAPI_Host')


@st.cache_data

def get_property_listings(city, state, limit):
    url = "https://realty-mole-property-api.p.rapidapi.com/saleListings"
    querystring = {"city": city, "state": state, "limit": limit}
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": API_HOST
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    real_estate_df = pd.DataFrame(json.loads(response.text))
    real_estate_df = real_estate_df[~real_estate_df['propertyType'].isin(['Land', 'Manufactured', 'Duplex-Triplex'])]
    real_estate_df = real_estate_df.drop(['county', 'addressLine1', 'city', 'state', 'zipCode', 'lastSeen', 'listedDate', 'status', 'removedDate', 'createdDate', 'id', 'addressLine2'], axis=1)
    return real_estate_df

def create_map(real_estate_df):
    map_ = folium.Map(location=[real_estate_df["latitude"].mean(), real_estate_df["longitude"].mean()], zoom_start=12)
    for i, row in real_estate_df.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"{row['formattedAddress']}<br>Price: ${row['price']}",
        ).add_to(map_)
    return map_

def fit_random_forest(real_estate_df, address, relevant_features=["latitude", "longitude", "squareFootage", "lotSize"]):
    address_df = real_estate_df[real_estate_df['formattedAddress'].str.contains(address, case=False)]
    if len(address_df) == 0:
        return 'No matching address found'
    
    X = real_estate_df[relevant_features].fillna(real_estate_df.mean())
    y = real_estate_df["price"]
    rf = RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_split=5, min_samples_leaf=1, random_state=1)
    rf.fit(X, y)

    address_info = address_df.iloc[0][relevant_features].fillna(X.mean())
    y_pred = rf.predict([address_info])[0]
    predicted_price = '${:,.0f}'.format(y_pred)

    return predicted_price





def page1():
    st.title("Real Estate Listings")
    st.sidebar.header("Search Listings")

    state_options = list(state_city_dict.keys())
    default_state = state_options[0]
    city_options = state_city_dict[default_state]
    default_city = city_options[0]

    selected_state = st.sidebar.selectbox('Select a state', options=state_options, index=0)
    selected_city = st.sidebar.selectbox('Select a city', options=state_city_dict[selected_state], index=0)
    selected_limit = st.sidebar.selectbox('Select how many listings', options=[100, 150, 200], index=0)

    st.sidebar.header("Search")
    search_button = st.sidebar.button("Search")

    if search_button:
        property_listings = get_property_listings(selected_city, selected_state, selected_limit)
        st.write(property_listings)


    real_estate_df = get_property_listings(selected_city, selected_state, selected_limit)
    st.write(real_estate_df)
    map_real_estate = create_map(real_estate_df)

    st.write("Map of Real Estate Properties")
    st_folium(map_real_estate)

    st.header("Actual Price Prediction Using KNN Neighbors")

    address = st.text_input("Enter an address:")
    if address:
        predicted_price = fit_random_forest(real_estate_df, address)
        st.write(predicted_price)

def page2():       
    with st.container():
        st.header("Investor Calculators")
        st.header("Mortgage Calculator and ROI Calculator")
        st.write("Use the following calculators to estimate your mortgage payments and ROI:")


        # Set the API endpoint and define the parameters
        api_url = 'https://api.api-ninjas.com/v1/mortgagecalculator'
        params = {
            'loan_amount': st.number_input('Loan amount', value=200000),
            'interest_rate': st.slider('Interest rate', min_value=0.0, max_value=10.0, step=0.1, value=3.5),
            'duration_years': st.selectbox('Duration in years', options=[10, 15, 20, 25, 30, 35, 40], index=4),
        }

        # Set the API key in the request headers and handle the response
        headers = {'X-Api-Key': 'knc8VXNlwMozZwpq6LFMWQ==FKgTtbuDkJuNXtSO'}
        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code == requests.codes.ok:
            result = response.json()
            st.write("Mortgage Calculator Result:")
            st.write("----------------------------")
            st.write("Monthly Payment:")
            st.write(f"   - Total: ${result['monthly_payment']['total']}")
            st.write(f"   - Mortgage: ${result['monthly_payment']['mortgage']}")
            st.write(f"   - Property Tax: ${result['monthly_payment']['property_tax']}")
            st.write(f"   - HOA: ${result['monthly_payment']['hoa']}")
            st.write(f"   - Annual Home Insurance: ${result['monthly_payment']['annual_home_ins']}")
            st.write("Annual Payment:")
            st.write(f"   - Total: ${result['annual_payment']['total']}")
            st.write(f"   - Mortgage: ${result['annual_payment']['mortgage']}")
            st.write(f"   - Property Tax: ${result['annual_payment']['property_tax']}")
            st.write(f"   - HOA: ${result['annual_payment']['hoa']}")
            st.write(f"   - Annual Home Insurance: ${result['annual_payment']['home_insurance']}")
            st.write("Total Interest Paid: ${result['total_interest_paid']}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

        # Create sidebar inputs for property value and loan amount for each property
        st.sidebar.header("LTV Calculator")
        num_properties = st.sidebar.number_input("Number of Properties", value=1, min_value=1)
        property_values = []
        loan_amounts = []

        for i in range(num_properties):
            st.sidebar.subheader(f"Property {i+1}")
            property_value = st.sidebar.number_input(f"Property {i+1} Value", value=1000000)
            loan_amount = st.sidebar.number_input(f"Property {i+1} Loan Amount", value=800000)
            property_values.append(property_value)
            loan_amounts.append(loan_amount)

        # Calculate LTV ratio for each property and display result
        for i in range(num_properties):
            ltv_ratio = np.round(loan_amounts[i] / property_values[i], 2) * 100
            st.write(f"Property {i+1} LTV Ratio:", ltv_ratio, "%")

# Create sidebar inputs for initial investment, cash flow, and holding period
    st.sidebar.header("ROI Calculator")
    initial_investment = st.sidebar.number_input("Initial Investment", value=100000)
    cash_flow = st.sidebar.number_input("Cash Flow", value=10000)
    holding_period = st.sidebar.number_input("Holding Period (years)", value=5)

    # Calculate ROI and display result
    total_return = initial_investment + cash_flow * holding_period
    roi = np.round((total_return - initial_investment) / initial_investment, 2) * 100
    st.write("ROI:", roi, "%")

def app():
    st.sidebar.title("Navigation")
    page_options = ["Page 1", "Page 2"]
    page_selection = st.sidebar.radio("Go to", page_options)

    if page_selection == "Page 1":
        page1()
    elif page_selection == "Page 2":
        page2()

if __name__ == '__main__':
    app()
