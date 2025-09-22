ARG PYTHON_VERSION=3.12
FROM public.ecr.aws/lambda/python:${PYTHON_VERSION} 

# Install system dependencies
COPY service/requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install -r requirements.txt

# Copy the service files
COPY service/model_service.py ${LAMBDA_TASK_ROOT}/service/model_service.py
COPY service/model/ ${LAMBDA_TASK_ROOT}/service/model/

# Set the CMD to your handler 
CMD ["service.model_service.handler"]
