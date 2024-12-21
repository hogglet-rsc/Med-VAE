import pandas as pd
from fancyimpute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# File paths
file_path = "/home/shogg/mwdh_ml_projects/data/test_data_with_eye_foot_and_drugs_v2_forMI.csv"
output_path = "/home/shogg/mwdh_ml_projects/data/test_data_with_eye_foot_and_drugs_v2_forMI_OE_MICE.csv"

# Load dataset
print("Loading dataset...")
data = pd.read_csv(file_path)

# Feature categories
dm_type_cols = ['dm_type_Maturity Onset Diabetes of Youth', 'dm_type_Secondary - Drug Induced',
                'dm_type_Secondary - Pancreatic Pathology', 'dm_type_Type 1 Diabetes Mellitus', 
                'dm_type_Type 2 Diabetes Mellitus']

smoking_cols = ['smoking_Current smoker', 'smoking_Ex-smoker', 'smoking_Never smoked', 
                'smoking_Patient declined']

ethCode_cols = ['ethCode_1B', 'ethCode_1C', 'ethCode_1L', 'ethCode_1Z', 'ethCode_2A', 
                'ethCode_3F', 'ethCode_3G', 'ethCode_3H', 'ethCode_3J', 'ethCode_3Z',
                'ethCode_4D', 'ethCode_4Y', 'ethCode_5C', 'ethCode_5D', 'ethCode_6A', 
                'ethCode_6Z', 'ethCode_98', 'ethCode_99']

embedding_features = ['foot_risk', 'eye_risk']
binary_features = ['sex_Male']

# Convert OHC to single categorical columns (will produce NaN for all-zero rows)
print("Converting one-hot encoded columns to categorical...")
categorical_columns = pd.DataFrame({
    'dm_type': data[dm_type_cols].apply(lambda x: dm_type_cols[x.argmax()] if x.max() == 1 else np.nan, axis=1),
    'smoking': data[smoking_cols].apply(lambda x: smoking_cols[x.argmax()] if x.max() == 1 else np.nan, axis=1),
    'ethCode': data[ethCode_cols].apply(lambda x: ethCode_cols[x.argmax()] if x.max() == 1 else np.nan, axis=1)
})

# Drop original one-hot columns and add new categorical columns
data = data.drop(columns=dm_type_cols + smoking_cols + ethCode_cols)
data = pd.concat([data, categorical_columns], axis=1)

# Apply OrdinalEncoder to categorical columns
print("Applying ordinal encoding...")
oe = OrdinalEncoder()
data[['dm_type', 'smoking', 'ethCode']] = oe.fit_transform(data[['dm_type', 'smoking', 'ethCode']])

# Imputation
print("Starting MICE imputation. This may take some time...")
imputer = IterativeImputer(max_iter=10, random_state=42)
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Post-imputation adjustments
# Ordinal-encoded features (clip to valid range)
valid_ranges = {
    'foot_risk': (0, 4), 
    'eye_risk': (0, 4),
    'dm_type': (0, len(dm_type_cols)-1),
    'smoking': (0, len(smoking_cols)-1),
    'ethCode': (0, len(ethCode_cols)-1)
}

for col, (min_val, max_val) in valid_ranges.items():
    data_imputed[col] = data_imputed[col].round().clip(min_val, max_val).astype(int)

# Binary feature (ensure binary values)
for col in binary_features:
    data_imputed[col] = data_imputed[col].round().clip(0, 1).astype(int)

# Save the imputed dataset
print("Saving the imputed dataset...")
data_imputed.to_csv(output_path, index=False)
print(f"Imputation complete. File saved at: {output_path}")