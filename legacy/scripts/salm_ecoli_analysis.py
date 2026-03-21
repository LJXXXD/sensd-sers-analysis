import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_and_clean_data(pathogen_paths, pathogen_names, index_column="Raman Shift"):
    all_pathogen_samples_list = []
    for i, pathogen_path_i in enumerate(pathogen_paths):
        # Load the datasets
        pathogen_i_data = pd.read_excel(pathogen_path_i)

        # Clean column names
        pathogen_i_data.columns = pathogen_i_data.columns.str.strip()

        # Set 'Raman Shift' as the index before transposing to keep it as feature names
        pathogen_i_samples = pathogen_i_data.set_index(index_column).T

        # Assign label of pathogen to the samples
        pathogen_i_samples["Pathogen"] = pathogen_names[i]

        # Assign Concentration
        pathogen_i_samples["Concentration"] = pathogen_i_samples.index

        # Add pathogen_i to list
        all_pathogen_samples_list.append(pathogen_i_samples)

    combined_samples = pd.concat(all_pathogen_samples_list)

    return combined_samples


def prepare_samples(samples, _source_label, _synthetic_label):
    """Placeholder for unfinished legacy sample harmonization."""
    return samples


def plot_explained_variance(explained_variance):
    """Placeholder for unfinished legacy PCA scree plot."""
    _ = explained_variance


def plot_knn_classification(pca_df):
    """Placeholder for unfinished legacy KNN visualization."""
    _ = pca_df


def standardize_and_pca(data):
    # Drop the 'Pathogen' and 'Concentration' columns to isolate features
    features = data.drop(columns=["Pathogen", "Concentration"]).apply(
        pd.to_numeric, errors="coerce"
    )

    # Extract 'Pathogen' and 'Concentration' columns as separate labels
    label_Pathogen = data["Pathogen"]
    label_Concentration = data["Concentration"]

    # Drop any rows with NaN values in the features DataFrame
    features.dropna(inplace=True)

    # Reset indices to ensure clean, sequential indexing
    features.reset_index(drop=True, inplace=True)
    label_Pathogen.reset_index(drop=True, inplace=True)
    label_Concentration.reset_index(drop=True, inplace=True)

    # Standardize the features to have mean = 0 and standard deviation = 1
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform PCA on the standardized features
    pca = PCA()
    pca_result = pca.fit_transform(scaled_features)

    return pca, pca_result, label_Pathogen, label_Concentration


def main():
    filepaths = [
        "E. coli and Salmonella 1.xlsx",
        "E. coli and Salmonella 2.xlsx",
        "E. coli and Salmonella 3.xlsx",
        "E. coli and Salmonella 4.xlsx",
        "E. coli and Salmonella 5.xlsx",
    ]

    combined_data = load_and_clean_data(
        filepaths,
        pathogen_names=[f"batch_{i + 1}" for i in range(len(filepaths))],
    )

    ecoli_samples = combined_data[combined_data.index.str.contains("E. coli")]
    salmonella_samples = combined_data[combined_data.index.str.contains("Salmonella")]

    ecoli_samples = prepare_samples(ecoli_samples, "E. coli", "E. coli_syn")
    salmonella_samples = prepare_samples(salmonella_samples, "Salmonella", "Salmonella_syn")

    combined_samples = pd.concat([ecoli_samples, salmonella_samples])

    pca, pca_result, labels, identifiers = standardize_and_pca(combined_samples)

    explained_variance = pca.explained_variance_ratio_
    plot_explained_variance(explained_variance)

    pca_df = pd.DataFrame(
        data=pca_result, columns=[f"PC{i + 1}" for i in range(pca_result.shape[1])]
    )
    pca_df["Label"] = labels
    pca_df["Identifier"] = identifiers

    plot_knn_classification(pca_df)


if __name__ == "__main__":
    main()
