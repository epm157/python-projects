from app.config import settings

from datetime import datetime
import os
# Import the `pprint` function to print nested data:
from pprint import pprint

import bson
from pymongo import MongoClient

def print_title(title):
    """
    Utility function to print a title with an underline.
    """
    print() # Print a blank line
    print(title)
    print('=' * len(title))

MONGODB_URI = f'mongodb+srv://{settings.database_username}:{settings.database_password}@{settings.database_hostname}/{settings.database_name}?retryWrites=true&w=majority'

# Connect to your MongoDB cluster:
client = MongoClient(MONGODB_URI)

print_title("Database names")

# List all the databases in the cluster:
for db_info in client.list_database_names():
    print(db_info)

# Get a reference to the 'sample_mflix' database:
db = client['sample_mflix']

print_title("Collections in 'sample_mflix'")
# List all the collections in 'sample_mflix':
collections = db.list_collection_names()
for collection in collections:
    print(collection)

# Get a reference to the 'movies' collection:
movies = db['movies']

# Get the document with the title 'Blacksmith Scene':
print_title("'Blacksmith Scene' document")
pprint(movies.find_one({'title': 'Blacksmith Scene'}))

print_title("Insert a document for 'Parasite'")

# Insert a document for the movie 'Parasite':
insert_result = movies.insert_one({
    "title": "Parasite",
    "year": 2020,
    "plot": "A poor family, the Kims, con their way into becoming the servants of a rich family, the Parks. "
            "But their easy life gets complicated when their deception is threatened with exposure.",
    "released": datetime(2020, 2, 7, 0, 0, 0)
})

# Save the inserted_id of the document you just created:
parasite_id = insert_result.inserted_id
print("_id of inserted document: {parasite_id}".format(parasite_id=parasite_id))

print_title("Look up the inserted document")

# Look up the document you just created in the collection:
print(movies.find_one({'_id': bson.ObjectId(parasite_id)}))

print_title("Look up all documents with the title 'Parasite'")

for doc in movies.find({'title': 'Parasite'}):
    pprint(doc)

print_title("Update 'Parasite' year to 2019")

# Update the document with the correct year:
update_result = movies.update_many({'title': 'Parasite'}, {
    '$set': {
        'year': 2019
    }
})

print(f'Documents updated: {update_result.modified_count}')

print_title("Delete all documents with the title 'Parasite'")

# delete_result = movies.delete_many({'title': 'Parasite'})
#
# print(f"Documents updated: {delete_result.deleted_count}")

movie1 = movies.find_one({'_id': bson.ObjectId(parasite_id)})
movie2 = movies.find_one({'_id': parasite_id})
print(movie1 == movie2)
print(bson.ObjectId(parasite_id) == parasite_id)

print(type(parasite_id))
print(parasite_id)