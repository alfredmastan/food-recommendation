ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim 

WORKDIR /dependencies

# Copy the entire dependencies folder
COPY dependencies/ .

# Install system dependencies
RUN pip install -r requirements.txt

EXPOSE 8000

# Run FastAPI on port 8000
WORKDIR /
CMD ["python", "dependencies/model_service.py"]
