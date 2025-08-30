# Use an official Python runtime as a parent image
FROM python:3.10-slim

#set working directory in container
WORKDIR /app

#copy requirements file into container
COPY requirements.txt ./

#install the required python packages,including gunicorn
RUN pip install --no-cache-dir -r requirements.txt

#copy entire application folder into container
COPY . .

##expose port that flask app runs on 5001
EXPOSE 5001

##use gunicorn to serve flask application in production

# Use Gunicorn to serve the Flask application in production
# The format is: gunicorn --bind <host>:<port> <module_name>:<app_instance_name>
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5001", "flask_chatbot:app"]


