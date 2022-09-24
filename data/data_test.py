import pandas as pd 
user_sheet = pd.read_excel('init_location.xlsx', sheet_name='user')
user_data = user_sheet

print(user_data)