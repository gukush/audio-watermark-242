#Two way anova should be used for analyzing significance with two factors cloning technique and watermark embedded
#In case of more manova
import pandas as pd
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols

input_folder = "test"
factors = ['pred_col', 'model']

# Initialize an empty DataFrame 
combined_df = pd.DataFrame()

# Loop through all CSV files in the input folder and concatenate them
for file in os.listdir(input_folder):
    if file.endswith(".csv"): 
        file_path = os.path.join(input_folder, file)
        temp_df = pd.read_csv(file_path)  # Read each CSV
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)  # Append to the combined DataFrame

# Dynamically construct the formula
formula = f'height ~ C({factors[0]}) + C({factors[1]}) + C({factors[0]}):C({factors[1]})'

# Perform two-way ANOVA
model = ols(formula, data=combined_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Display the combined DataFrame
print(anova_table)