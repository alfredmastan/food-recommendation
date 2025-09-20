ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim 

# Install system dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the source code into the container.
COPY app/ app/
COPY params.yaml .
COPY data/processed_cookbook.pkl data/

# Expose port
EXPOSE 8501

# Run the application.
WORKDIR /app
CMD ["streamlit", "run", "streamlit_web.py"]