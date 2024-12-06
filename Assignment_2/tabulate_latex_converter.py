import re

def extract_data_from_tabulate(tabulate_str):
  # Split the table into lines
  lines = tabulate_str.split('\n')
  
  # Filter out separator lines (lines with only "+" or "+" and "-" characters)
  valid_lines = [
    line for line in lines 
    if not re.match(r'^\+[-+]+$', line)
  ]
  
  # Extract data (excluding the header) by splitting each valid line on "|"
  data = [line.split('|')[1:-1] for line in valid_lines[1:] if line.strip()]
  
  return data

# Function to read the file
def read_file(file_path):
  with open(file_path, 'r') as file:
    return file.read()

# Format extracted data as LaTeX rows
def format_data_as_latex(data):
    formatted_rows = []
    for row in data:
      processed_row = [
        (f"{float(cell.strip()):.3f}" if idx in {6, 8} else cell.strip())
        for idx, cell in enumerate(row)
        if idx != 7
      ]
      # Join the processed row with " & ", then append "\\" at the end
      formatted_rows.append(" & ".join(processed_row) + " \\\\")
    return formatted_rows

# Path to your file (replace with your actual file path)
file_path = 'tabulate_to_be_converted.txt'

# Read the content of the file
tabulate_str = read_file(file_path)

# Extract data from the tabulate string
data = extract_data_from_tabulate(tabulate_str)

# Format data as LaTeX rows
latex_rows = format_data_as_latex(data)

# Print the LaTeX rows
for row in latex_rows:
  print(row)
