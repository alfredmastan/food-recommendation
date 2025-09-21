ARG PYTHON_VERSION=3.12
FROM public.ecr.aws/lambda/python:${PYTHON_VERSION} 

WORKDIR ${LAMBDA_TASK_ROOT}/dependencies/
# Copy the entire dependencies folder
COPY dependencies/ .

# Install system dependencies
RUN pip install -r requirements.txt

# Set the CMD to your handler 
WORKDIR ${LAMBDA_TASK_ROOT}/
CMD ["dependencies.model_service.handler"]
