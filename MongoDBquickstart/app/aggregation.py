import pymongo

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

movie_collection = db['movies']

def a_sample_movie_document():
    print_title('A Sample Movie')
    pipeline = [
        {
            '$match': {
                'title': 'A Star Is Born'
            }
        },
        {'$limit': 1}
    ]
    results = movie_collection.aggregate(pipeline)
    for movie in results:
        pprint(movie)


def a_sample_comment_document():
    print_title("A Sample Comment")
    pipeline = [
        {'$limit': 1}
    ]
    results = db['comments'].aggregate(pipeline)
    for comment in results:
        pprint(comment)

def a_star_is_born_all():
    print_title("A Star Is Born - All Documents")
    pipeline = [
        {
            '$match': {
                'title': 'A Star Is Born'
            }
        },
        {
            '$sort': {
                'year': pymongo.ASCENDING
            }
        }
    ]
    results = movie_collection.aggregate(pipeline)
    for movie in results:
        print(" * {title}, {first_castmember}, {year}".format(
            title=movie["title"],
            first_castmember=movie["cast"][0],
            year=movie["year"],
        ))


def a_star_is_born_recent():
    print_title("A Star Is Born - Most Recent")
    stage_match_title = {
        '$match': {
            'title': 'A Star Is Born'
        }
    }

    stage_sort_year_descending = {
        '$sort': {
            'year': pymongo.DESCENDING
        }
    }

    stage_limit_1 = {
        '$limit': 1
    }

    pipeline = [
        stage_match_title,
        stage_sort_year_descending,
        stage_limit_1
    ]

    results = movie_collection.aggregate(pipeline)
    for movie in results:
        print(" * {title}, {first_castmember}, {year}".format(
            title=movie["title"],
            first_castmember=movie["cast"][0],
            year=movie["year"],
        ))


def movies_with_comments():
    print_title("Movies With Comments")
    stage_lookup_comments = {
        '$lookup': {
            'from': 'comments',
            'localField': '_id',
            'foreignField': 'movie_id',
            'as': 'related_comments'
        }
    }

    stage_add_comment_count = {
        '$addFields': {
            'comment_count': {
                '$size': '$related_comments'
            }
        }
    }

    stage_match_with_comments = {
        '$match': {
            'comment_count': {
                '$gt': 2
            }
        }
    }

    limit_5 = {
        '$limit': 5
    }

    limit_1000 = {
        '$limit': 1000
    }

    pipeline = [
        #limit_1000,
        stage_lookup_comments,
        stage_add_comment_count,
        stage_match_with_comments,
        limit_5
    ]

    results = movie_collection.aggregate(pipeline)
    for movie in results:
        print_title(movie["title"])
        print("Comment count:", movie["comment_count"])

        # Loop through the first 5 comments and print the name and text:
        for comment in movie["related_comments"][:5]:
            print(" * {name}: {text}".format(
                name=comment["name"],
                text=comment["text"]))


def movies_each_year():
    print_title("Movies Grouped By Year")

    stage_group_year = {
        '$group': {
            '_id': '$year',
            'movie_count': {'$sum': 1},
            'movie_titles': {'$push': '$title'}
        }
    }

    stage_match_years = {
        '$match': {
            'year': {
                '$type': 'number',
                '$lt': 1920
            }
        }
    }

    stage_sort_year_ascending = {
        '$sort': {'_id': pymongo.ASCENDING}
    }

    pipeline = [
        stage_match_years,
        stage_group_year,
        stage_sort_year_ascending,
    ]
    results = movie_collection.aggregate(pipeline)

    # Loop through the 'year-summary' documents:
    for year_summary in results:
        # Print an underlined heading for each year:
        title = "{year}: {count} movies".format(
            year=year_summary["_id"],
            count=year_summary["movie_count"])
        print_title(title)
        # Loop through the document titles for each year and print them
        # as bullets:
        for title in year_summary["movie_titles"]:
            print(" *", title)


#a_sample_movie_document()
#a_sample_comment_document()
#a_star_is_born_all()
#a_star_is_born_recent()
movies_with_comments()
#movies_each_year()



