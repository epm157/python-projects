import os
import pandas as pd
import hashlib

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

PATH = '/Users/ehsan/Downloads/Takeout/Google Photos'
PATH_NEW = '/Users/ehsan/Downloads/Takeout/all'

files_name_size_hash_list = []

file_names_set = set()
file_hashes_set = set()

for path, subdirs, files in os.walk(PATH):
    for name in files:
        file = os.path.join(path, name)
        fileSize = os.path.getsize(str(file))
        fileHash = md5(file)
        files_name_size_hash_list.append((file, fileSize, fileHash))



COLUMN_NAME_PATH = 'Path'
COLUMN_NAME_SIZE = 'Size'
COLUMN_NAME_HASH = 'Hash'

df = pd.DataFrame(files_name_size_hash_list, columns=[COLUMN_NAME_PATH, COLUMN_NAME_SIZE, COLUMN_NAME_HASH])
df = df.sort_values(COLUMN_NAME_SIZE, ascending=False)

#print(df.head(10))


for index, row in df.iloc[:].iterrows():
    file_path = row[COLUMN_NAME_PATH]
    file_size = row[COLUMN_NAME_SIZE]
    file_hash = row[COLUMN_NAME_HASH]

    filepath_wo_extension, file_extension = os.path.splitext(file_path)

    if file_extension in ['.csv', '.tsv', '.html', '.txt', '.zip', '.json']:
        print(f'{file_path} {row[COLUMN_NAME_SIZE]}')
        os.remove(file_path)
    elif filepath_wo_extension.endswith('.DS_Store'):
        print('Removing DS file')
        os.remove(file_path)

    file_name = os.path.basename(filepath_wo_extension) + file_extension
    if file_hash in file_hashes_set:
        #pass
        print(f'{file_name} already exists in the set')
        #os.remove(file_path)

    file_hashes_set.add(file_hash)
    file_names_set.add(file_name)

print('Finished...')



