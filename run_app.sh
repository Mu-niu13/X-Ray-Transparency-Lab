#!/bin/bash
# Startup script for X-Ray Transparency Lab
# Handles OpenMP conflict on Mac

# Set environment variable to allow duplicate OpenMP libraries
export KMP_DUPLICATE_LIB_OK=TRUE

# Activate virtual environment
source venv/bin/activate

# Run the Streamlit app
python -m streamlit run app.py