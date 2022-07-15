import pymongo
from datetime import datetime
from flask import Flask
from flask import request

print('Entering app')
app = Flask(__name__)

print('Making DB connection')
#conn = pymongo.MongoClient('mongodb://admin:password@localhost:27017/')
conn = pymongo.MongoClient('mongodb://admin:password@mongodb:27017/')
print('Creating DB')
db = conn['MongoDBDemo']

@app.route('/')
def home():
    return 'Welcome!'

@app.route('/insert')
def insert():
    if 'title' not in request.args:
        return 'Please provide title'
    title = request.args.get('title')
    insertIntoDB(title)
    return f'{title} is inserted into database'

def insertIntoDB(title):
    entries = db['Entries']

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    record = {'Title': title, 'Datetime': dt_string}
    entries.insert_one(record)


if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True)





'''

#CMD ["flask", "run", "--host", "0.0.0.0" ]
    #ENV export FLASK_APP=MongoDBDemo.py
    #MongoDBDemo
    #docker run -d -p 5000:5000 --network specific-network-name mongo-demo:1.0

    #docker login --username=epm157 cloud.canister.io:5000
    #@4b!Wr9B7PvB4tR'''


'''
docker run -d \                                                             
-p 8081:8081 \
-e ME_CONFIG_MONGODB_ADMINUSERNAME=admin \
-e ME_CONFIG_MONGODB_ADMINPASSWORD=password \
--network mongo-network \
--name mongo-express \
-e ME_CONFIG_MONGODB_SERVER=mongodb \
mongo-express



docker run -d -p 27017:27017 \                                              
-e MONGO_INITDB_ROOT_USERNAME=admin -e MONGO_INITDB_ROOT_PASSWORD=password \
--name mongodb --network mongo-network mongo


docker-compose -f mongo.yaml up 


docker tag 27702810322a cloud.canister.io:5000/epm157/mongodemo:latest

docker push cloud.canister.io:5000/epm157/mongodemo

'''


