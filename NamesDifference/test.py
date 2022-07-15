import pandas as pd

file_name = "lc.xlsx"
xl_file = pd.ExcelFile(file_name)

dfs = {sheet_name: xl_file.parse(sheet_name, header=None,)
       for sheet_name in xl_file.sheet_names}

lynes_df = dfs['Laynes']
lynes_df.columns = ['name']

cua_df = dfs['Cua']
cua_df.columns = ['name']

lynes_list = lynes_df.values.tolist()
cua_list = cua_df.values.tolist()

# names = []
# i = 1

for lynes in lynes_list:
  # names.append(f'{i}: {lynes[0]}')
  # i = i + 1
  found = False
  for cua in cua_list:
    if cua[0] == lynes[0]:
      found = True

      #print(lynes[0])
      break
  if not found:
     pass
    #names.append(lynes[0])
    #print(lynes)


with open('lynes.txt', 'w') as f:
  for name in names:
    f.write(name)
    f.write('\n')


