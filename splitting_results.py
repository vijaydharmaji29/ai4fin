import pandas as pd

# Load the CSV file
file_path = './formatted_results/rnmse.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Split the 'Commodity' column by the character '_' and create two new columns
data[['Commodity Name', 'State']] = data['Commodity'].str.split('_', expand=True)

# Reorder the columns to make 'Commodity Name' and 'Location' the first two columns
data = data[['Commodity Name', 'State'] + [col for col in data.columns if col not in ['Commodity', 'Commodity Name', 'Location']]]

# Display the updated dataframe
print(data.head())

# Save the updated dataframe to a new CSV file
data.to_csv('./formatted_results/rnmse.csv', index=False)  # Replace with the desired file path
