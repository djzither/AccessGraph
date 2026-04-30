import pandas as pd

def get_row_as_list(file_path, supervisor_name, position_name, sheet):
    # Pass the sheet_name parameter here
    df = pd.read_excel(file_path, sheet_name=sheet)

    match = df[
    (df['Supervisors'] == supervisor_name) & 
    (df['Job Title'].str.contains(position_name, na=False, case=False, regex=False))
    ]
    
    if not match.empty:
        return match.iloc[0].tolist()
    else:
        return "No match found."

print("starting")
supervisor = input("Supervisor: ").strip()
position = input("Position: ").strip()

# Added 'r' for raw string to handle Windows backslashes correctly
file_path = r"C:\Users\kcard4\Box\CE IT\CE Help Desk\Documents\Access List - Student Employees.xlsx"

# You can use the name of the sheet...
target_sheet = "Data Base" 
# ...or the index (0 for the first sheet, 1 for the second, etc.)
# target_sheet = 0 

data_list = get_row_as_list(file_path, supervisor, position, target_sheet)
print(data_list)