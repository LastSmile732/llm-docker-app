#!/bin/bash

# Start FastAPI app
echo "Starting FastAPI app..."
#uvicorn main:app --reload &
python3 main.py &
# Store FastAPI process ID
FASTAPI_PID=$!

# Start Streamlit app
echo "Starting Streamlit app..."
streamlit run streamlit_app.py &

# Store Streamlit process ID
STREAMLIT_PID=$!

# Wait for any process to exit
wait -n

# Kill the other process when one exits
kill -TERM $FASTAPI_PID
kill -TERM $STREAMLIT_PID
