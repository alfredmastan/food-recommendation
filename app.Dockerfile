ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim 

# Install system dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the source code into the container.
COPY app.py app.py 
COPY params.yaml .
COPY data/final_cookbook.pkl Data/

# Expose port
EXPOSE 8501

# Run the application.
CMD ["streamlit", "run", "app.py"]