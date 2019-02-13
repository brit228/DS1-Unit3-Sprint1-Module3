FROM python:3.7
WORKDIR /app
RUN mkdir output
COPY docker_run /app
RUN pip install --trusted-host pypi.python.org --extra-index-url https://test.pypi.org/simple/ -r requirements.txt
EXPOSE 80
CMD python run.py ${ENV_CSV_FILE} ${ENV_YAML_FILE}
