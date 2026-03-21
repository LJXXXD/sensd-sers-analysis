import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the datasets
salmonella_data = pd.read_excel("SERS Data 1/Concentration study(Salmonella).xlsx")
ecoli_data = pd.read_excel("SERS Data 1/Concnetrations Study(E. coli).xlsx")

print(salmonella_data)


# Transpose the dataframes to treat each concentration column as a separate sample
salmonella_samples = salmonella_data.set_index("Raman Shift")
ecoli_samples = ecoli_data.set_index("Raman Shift")

# Assign labels to the samples
salmonella_samples["Label"] = "Salmonella"
ecoli_samples["Label"] = "E.coli"

# Combine the datasets
combined_samples = pd.concat([salmonella_samples, ecoli_samples])

# Separate features and labels
features_samples = combined_samples.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
labels_samples = combined_samples["Label"]

# Drop any rows with remaining NaN values after conversion
features_samples.dropna(inplace=True)

# Reset index to avoid duplicate label issues
features_samples.reset_index(drop=True, inplace=True)
labels_samples.reset_index(drop=True, inplace=True)

# Apply PCA
pca = PCA(n_components=2)
pca_result_samples = pca.fit_transform(features_samples)

# Create a dataframe for PCA results
pca_df_samples = pd.DataFrame(data=pca_result_samples, columns=["PC1", "PC2"])
pca_df_samples["Label"] = labels_samples

# Plotting the PCA results
plt.figure(figsize=(12, 6))

for label in pca_df_samples["Label"].unique():
    subset = pca_df_samples[pca_df_samples["Label"] == label]
    plt.scatter(subset["PC1"], subset["PC2"], label=label, alpha=0.6)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Salmonella and E. coli Concentration Data (Corrected)")
plt.legend()
plt.grid(True)
plt.show()
