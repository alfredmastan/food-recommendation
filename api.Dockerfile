ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim 

WORKDIR /dependencies

# Install system dependencies
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Copy the model artifacts and api wrapper into the container.
COPY dependencies/ .

EXPOSE 8000

# Run FastAPI on port 8000
CMD ["python", "model_service.py"]