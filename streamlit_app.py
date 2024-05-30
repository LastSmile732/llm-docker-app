import streamlit as st
import requests
import pandas as pd

# Run this with: streamlit run streamlit_app.py
# Streamlit interface
st.title("Gemini Central Console Bot")
#action = st.selectbox("Choose an action:", ["draw_line_chart", "generate_answer"])
action = st.selectbox("Choose an action:", ["generate_answer"])
host = "https://localhost:9999"
# csv_path = "/opt/Enno"
# df_hourly_m = pd.read_csv(csv_path+'/UsagePerHourMerged.csv', delimiter=',')

# if action == "draw_line_chart":
#     url = host + "/analyse"
#     # UI for line chart request
#     start_date = st.date_input("Start Date")
#     end_date = st.date_input("End Date")

#     if st.button("Draw"):
#         # Filter DataFrame based on date range
#         df_hourly_m['Date'] = pd.to_datetime(df_hourly_m['Date'])
#         mask = (df_hourly_m['Date'] >= pd.Timestamp(start_date)) & (df_hourly_m['Date'] <= pd.Timestamp(end_date))
#         filtered_df = df_hourly_m.loc[mask]
        
#         total_usage_series = filtered_df['TotalUsage'].tolist()
#         series_text = ', '.join([f"{value:.2f}" for value in total_usage_series])
#         payload = {"data": series_text}
#         st.write("Drawing the line chart....")
#         # Plotting
#         # Ensure that 'Date' is the index if you want it on the x-axis
#         st.line_chart(filtered_df.set_index('Date')['TotalUsage'])

#         #Send the request to FastAPI endpoint
#         response = requests.post(url, json=payload)
#         if response.status_code == 200:
#             result = response.json()
#             content = result["choices"][0]["message"]["content"]
#             st.write(content)
            
#         else:
#             st.write(response.text)
        
if action == "generate_answer":
    user_input = st.text_area("Enter your text here")
    url = host + "/text_generation"  # URL of your FastAPI predict endpoint

    if st.button("Submit"):
        # Prepare the payload
        payload = {"prompt": user_input}
        

        # Send the request to FastAPI endpoint
        response = requests.post(url, json=payload, verify=False)
        
        # Display the response
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            st.write(content)
            
        else:
            st.write(response.text)