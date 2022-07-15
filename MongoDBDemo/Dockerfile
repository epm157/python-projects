FROM continuumio/anaconda3:latest
COPY ./app /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD python3 MongoDBDemo.py