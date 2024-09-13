import pandas as pd

# Load the CSV file
df = pd.read_csv('/Users/henkalui/Desktop/JDE/Speed_dating_project/Speed Dating Data.csv', encoding='latin1')

# Remove rows with null values in specific columns
df.dropna(subset=['gender', 'pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 
                  'pf_o_amb', 'pf_o_sha', 'dec_o', 'attr_o', 'sinc_o', 
                  'intel_o', 'fun_o', 'amb_o', 'shar_o'], inplace=True)

# Check the result
print(df.head())
print(df.info())

# Save the cleaned DataFrame to a new CSV file
df.to_csv('/Users/henkalui/Desktop/JDE/Speed_dating_project/Cleaned_Speed_Dating_Data.csv', index=False)

print("Cleaned data saved to 'Cleaned_Speed_Dating_Data.csv'")



