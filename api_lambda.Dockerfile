ARG PYTHON_VERSION=3.12
FROM public.ecr.aws/lambda/python:${PYTHON_VERSION} 

# Install system dependencies
COPY dependencies/requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install -r requirements.txt

# Copy the entire dependencies folder
COPY dependencies/ ${LAMBDA_TASK_ROOT}/dependencies/

# Set the CMD to your handler 
CMD ["dependencies.model_service.handler"]
