D = {"name": "zohreh pourmohammadi", "firstname": "zohreh", "lastname": "pourmohammadi", "age": 19, "score": 100}

def dic(D):
    if not isinstance(D, dict):
        print("input is not valid")
        return ([], [])
    else:
        return list(dict.keys(D)), list(dict.values(D))

print(dic("test"))

