#################################################
#   CRIS: Client Retention Intelligent System   #
#################################################

# -> Overview: This application helps businesses 
#            manage and predict customer churn.
# It features:
# - Adding new customers
# - Visualizing customer data on a map
# - Predicting churn prob. using a pretrained model
# 
# **Technologies Used**:
# - Streamlit for the web interface
# - Folium for map rendering
# - PyCaret for churn prediction model
#------------------------------------------------

# Imports
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, MousePosition, Geocoder, Draw
from streamlit_folium import folium_static
import random
from pycaret.classification import load_model, predict_model
from streamlit_folium import st_folium
random.seed(88)
import base64

# App name
st.set_page_config(page_title="CRIS", page_icon=":bar_chart:")

# Load the customer data and model
data_path = 'data/Data_with_Churn_Probability.csv'
data = pd.read_csv(data_path)

# Load the model
model_path = 'best_churn_model'
model = load_model(model_path)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    .title {
        font-size: 36px;
        font-family: 'Roboto', sans-serif;
        text-align: center;
        color: #4A4A4A;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 20px;
        font-family: 'Roboto', sans-serif;
        text-align: center;
        color: #4A4A4A;
        margin-bottom: 20px;
    }
    .custom-checkbox {
        margin: 10px 0;
    }

    </style>
    """, unsafe_allow_html=True)

st.write("""
    <div style="display:flex;align-items:center;justify-content:center;">
        <img src="data:image/png;base64,{}" width="140">      
    </div>
""".format(get_base64_of_bin_file("logo.png")), unsafe_allow_html=True)


# Define function to generate random avatar URLs
def get_random_avatar():
    width, height = 150, 150
    return f"https://picsum.photos/{width}/{height}?random={random.randint(1, 1000)}"

# Add avatars to the data
data['Avatar'] = [get_random_avatar() for _ in range(len(data))]


# Function to create a map with customer data
def create_map(data, show_prob=False, location=None):
    
    if location:
        m = folium.Map(location=location, zoom_start=25, width='100%', height='100vh')
    else:
        m = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=6, width='100%', height='100vh')
        
    marker_cluster = MarkerCluster().add_to(m)

    # Pluguins
    MousePosition().add_to(m)
    Geocoder().add_to(m)

    for idx, row in data.iterrows():
        popup_content = f"""
        <div style="text-align: center;">
            <img src="{row['Avatar']}" width="50" style="border-radius: 50%; margin-bottom: 10px;">
            <p><b>ID:</b> {row['Customer ID']}</p>
            <p><b>Age:</b> {row['Age']}</p>
            <p><b>Gender:</b> {row['Gender']}</p>
            <p><b>Status:</b> {row['Customer Status']}</p>
            <p><b>Monthly Charge:</b> {row['Monthly Charge']}€</p>
        </div>
        """
        color = 'green' if row['Customer Status'] == 'Stayed' else 'lightblue'
        if row['Customer Status'] == 'Churned':
            color = 'red'

        if show_prob:
            if row['Customer Status'] == 'Churned': 
                continue

            popup_content = f"""
            <div style="text-align: center;">
                <img src="{row['Avatar']}" width="50" style="border-radius: 50%; margin-bottom: 10px;">
                <p><b>ID:</b> {row['Customer ID']}</p>
                <p><b>Current Status:</b> {row['Customer Status']}</p>
                <p><b>Status Predicted:</b> {row['Predicted Label']}</p>
                <p><b>Churn Probability:</b> {row['Prediction Probability']}</p>
            </div>
            """
            color = 'red' if row['Predicted Label'] == 'Churned' else 'green'

        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_content, max_width=600),
            icon=folium.Icon(icon="user", color=color)
        ).add_to(marker_cluster)
    
    return m


st.markdown('<div class="subtitle">Client Retention Intelligent System</div>', unsafe_allow_html=True)

# Filter options
status_filter = st.multiselect("Filter by Customer Status", options=data['Customer Status'].unique(), default=['Stayed', 'Joined'])
filtered_data = data[data['Customer Status'].isin(status_filter)]

# Show churn probability for 'Stayed' customers
show_prob = st.checkbox("Display Predicted Churn Probability for Current Customers", key="show_prob", help="Apply the trained model to show the predicted churn probability for each current customer.")

with st.expander("🔍 Search Customer by IDs"):
    search_id = st.text_input("Insert Customer ID")
    search_button = st.button("Search")
location=None
if search_button and search_id:
    customer_data = data[data['Customer ID'] == search_id]
    if not customer_data.empty:
        customer = customer_data.iloc[0]
        location = [customer['Latitude'], customer['Longitude']]
        st.success(f"Customer ID {search_id} found.")
    else:
        st.error("Customer ID not found")

customer_map = create_map(filtered_data, show_prob, location=location)
folium_static(customer_map)




# New Customer code page
st.sidebar.title("👤 Add New Customer")
if 'new_customer_location' not in st.session_state:
    st.session_state.new_customer_location = None

with st.sidebar.form("add_customer_form"):

    with st.expander("Personal Information"):
        customer_id = st.text_input("Customer ID")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", options=['Male', 'Female'])
        married = st.selectbox("Married", options=['Yes', 'No'])
        dependents = st.slider("Number of Dependents", min_value=0, max_value=10, value=0)

    with st.expander("Referrals and Offers"):
        referrals = st.slider("Number of Referrals", min_value=0, max_value=20, value=0)
        offer = st.selectbox("Offer", options=['None', 'Offer A', 'Offer B', 'Offer C', 'Offer D'])

    with st.expander("Contract Details"):
        contract = st.selectbox("Contract", options=['Month-to-Month', 'One Year', 'Two Year'])
        monthly_charge = st.slider("Monthly Charge", min_value=0.0, max_value=100.0, value=24.0)
        tenure = st.slider("Tenure in Months", min_value=0, max_value=24, value=12)

    with st.expander("Services"):
        phone_service = st.selectbox("Phone Service", options=['Yes', 'No'])
        multiple_lines = st.selectbox("Multiple Lines", options=['Yes', 'No'])
        internet_service = st.selectbox("Internet Service", options=['Yes', 'No'])
        internet_type = st.selectbox("Internet Type", options=['DSL', 'Fiber Optic', 'None'])
        online_security = st.selectbox("Online Security", options=['Yes', 'No'])
        online_backup = st.selectbox("Online Backup", options=['Yes', 'No'])
        device_protection = st.selectbox("Device Protection", options=['Yes', 'No'])
        tech_support = st.selectbox("Tech Support", options=['Yes', 'No'])
        streaming_tv = st.selectbox("Streaming TV", options=['Yes', 'No'])
        streaming_movies = st.selectbox("Streaming Movies", options=['Yes', 'No'])
        streaming_music = st.selectbox("Streaming Music", options=['Yes', 'No'])
        paperless_billing = st.selectbox("Paperless Billing", options=['Yes', 'No'])
        unlimited_data = st.selectbox("Unlimited Data",  options=['Yes', 'No'])
        device_plan = st.selectbox("Device Plan",  options=['Yes', 'No'])

    with st.expander("Payment Method"):
        payment_method = st.selectbox("Payment Method", options=['Bank Withdrawal', 'Credit Card', 'Mailed Check'])


    #select_location_button = st.form_submit_button("Select Location")
    st.markdown(f"📍**Select a User Location**")
    m = folium.Map(location=[36.7783, -119.4179], zoom_start=5)
    draw = Draw(
        export=True,
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'rectangle': False,
            'circlemarker': False,
            'marker': True
        }
    )
    draw.add_to(m)
    st.session_state.new_customer_location = st_folium(m, height=200, width=400)
    add_location = st.form_submit_button("Add Location")
    if add_location:
        location = st.session_state.new_customer_location.get('last_active_drawing')
        if location:
            geometry = location['geometry']
            if geometry['type'] == 'Point':
                latitude = geometry['coordinates'][1]
                longitude = geometry['coordinates'][0]
                st.session_state.latitude = latitude
                st.session_state.longitude = longitude
                st.write(f"Selected Location - Latitude: {latitude}, Longitude: {longitude}")

    add_customer_button = st.form_submit_button("Add Customer and Predict Churn")
    
    latitude = st.session_state.get('latitude', None)
    longitude = st.session_state.get('longitude', None)      
    if add_customer_button:
        if st.session_state.new_customer_location and latitude:
            new_customer = pd.DataFrame({
                'Customer ID': [customer_id],
                'Gender': [gender],
                'Age': [age],
                'Married': [married],
                'Number of Dependents': [dependents],
                'Latitude': [latitude],
                'Longitude': [longitude],
                'Number of Referrals': [referrals],
                'Tenure in Months': [tenure],
                'Offer': [offer],
                'Phone Service': [phone_service],
                'Multiple Lines': [multiple_lines],
                'Internet Service': [internet_service],
                'Internet Type': [internet_type],
                'Online Security': [online_security],
                'Online Backup': [online_backup],
                'Device Protection Plan': [device_protection],
                'Tech Support': [tech_support],
                'Unlimited Data': [unlimited_data],
                'Streaming TV': [streaming_tv],
                'Streaming Movies': [streaming_movies],
                'Streaming Music': [streaming_music],
                'Contract': [contract],
                'Paperless Billing': [paperless_billing],
                'Payment Method': [payment_method],
                'Monthly Charge': [monthly_charge],
                'Total Charges': [monthly_charge*12],
                'Total Refunds': 0,
                'Total Extra Data Charges': 0,
                'Avg Monthly Long Distance Charges': 0,
                'Avg Monthly GB Download': 0,
                'Premium Tech Support': [tech_support]
            })


            predictions = predict_model(model, data=new_customer.drop(columns=['Customer ID']))
            new_customer['Customer Status']  = 'Joined'
            new_customer['Predicted Label']  = predictions['prediction_label'][0]
            new_customer['Prediction Probability'] = predictions['prediction_score'][0]
            prediction_label = predictions['prediction_label'][0]
            prediction_score = int(predictions['prediction_score'][0] * 100)

            if prediction_label == 'Stayed':
                st.success(f"**The predicted status for the new customer is {prediction_label}** with a probability of **{prediction_score}%**.")
            elif prediction_label == 'Churned':
                st.error(f"**The predicted status for the new customer is {prediction_label}** with a probability of **{prediction_score}%**.")

            # Save new customer data
            data = pd.concat([data, new_customer], ignore_index=True)
            data.to_csv(data_path, index=False)
            st.write("New customer data has been saved successfully!")
            st.rerun()
        else:
            st.error("Please select a location on the map.")
