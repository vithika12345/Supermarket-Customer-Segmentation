import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA


# ============================================================
# COLUMN NAME HELPERS
# ============================================================

def normalize_column_name(column):
    """
    Convert a column name into a simplified form so that
    different spellings/capitalization can be detected.
    """

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "")
    )


def find_column(df, possible_names):
    """
    Find the actual dataframe column corresponding to one
    of the possible column names.
    """

    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for name in possible_names:

        normalized_name = normalize_column_name(name)

        if normalized_name in normalized_columns:

            return normalized_columns[normalized_name]

    return None


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataset(df):
    """
    Prepare the retail customer dataset for the application.

    This function:
    - removes completely empty columns
    - standardizes important column names
    - converts numerical columns
    - creates purchase-frequency feature
    - creates binary features for Yes/No fields
    - keeps categorical columns for one-hot encoding later
    """

    df = df.copy()

    # --------------------------------------------------------
    # Remove completely empty columns
    # --------------------------------------------------------

    df = df.dropna(
        axis=1,
        how="all"
    )

    # --------------------------------------------------------
    # Detect columns from the retail dataset
    # --------------------------------------------------------

    detected_columns = {}

    column_patterns = {

        "Customer ID": [
            "Customer ID",
            "CustomerID"
        ],

        "Age": [
            "Age"
        ],

        "Gender": [
            "Gender"
        ],

        "Item Purchased": [
            "Item Purchased",
            "ItemPurchased"
        ],

        "Category": [
            "Category"
        ],

        "Purchase Amount (USD)": [
            "Purchase Amount (USD)",
            "PurchaseAmountUSD",
            "Purchase Amount"
        ],

        "Location": [
            "Location"
        ],

        "Size": [
            "Size"
        ],

        "Color": [
            "Color"
        ],

        "Season": [
            "Season"
        ],

        "Review Rating": [
            "Review Rating",
            "ReviewRating"
        ],

        "Subscription Status": [
            "Subscription Status",
            "SubscriptionStatus"
        ],

        "Shipping Type": [
            "Shipping Type",
            "ShippingType"
        ],

        "Discount Applied": [
            "Discount Applied",
            "DiscountApplied"
        ],

        "Promo Code Used": [
            "Promo Code Used",
            "PromoCodeUsed"
        ],

        "Previous Purchases": [
            "Previous Purchases",
            "PreviousPurchases"
        ],

        "Payment Method": [
            "Payment Method",
            "PaymentMethod"
        ],

        "Frequency of Purchases": [
            "Frequency of Purchases",
            "FrequencyofPurchases"
        ]
    }

    for standard_name, possible_names in column_patterns.items():

        original_column = find_column(
            df,
            possible_names
        )

        if original_column is not None:

            detected_columns[
                standard_name
            ] = original_column

    # --------------------------------------------------------
    # Rename detected columns to standard names
    # --------------------------------------------------------

    rename_map = {}

    for standard_name, original_name in detected_columns.items():

        if original_name != standard_name:

            rename_map[
                original_name
            ] = standard_name

    df = df.rename(
        columns=rename_map
    )

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numeric_columns = [

        "Customer ID",

        "Age",

        "Purchase Amount (USD)",

        "Review Rating",

        "Previous Purchases"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Clean categorical columns
    # --------------------------------------------------------

    categorical_columns = [

        "Gender",

        "Item Purchased",

        "Category",

        "Location",

        "Size",

        "Color",

        "Season",

        "Subscription Status",

        "Shipping Type",

        "Discount Applied",

        "Promo Code Used",

        "Payment Method",

        "Frequency of Purchases"
    ]

    for column in categorical_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    # --------------------------------------------------------
    # Purchase Frequency Engineering
    #
    # Convert categorical purchase frequency into an
    # approximate number of purchases per year.
    #
    # Weekly       = 52
    # Fortnightly   = 26
    # Monthly       = 12
    # Quarterly     = 4
    # Annually      = 1
    # --------------------------------------------------------

    if "Frequency of Purchases" in df.columns:

        frequency_mapping = {

            "weekly": 52,

            "fortnightly": 26,

            "monthly": 12,

            "quarterly": 4,

            "annually": 1,

            "annual": 1
        }

        df["Purchase Frequency (per year)"] = (
            df["Frequency of Purchases"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(frequency_mapping)
        )

        df["Purchase Frequency (per year)"] = (
            pd.to_numeric(
                df["Purchase Frequency (per year)"],
                errors="coerce"
            )
        )

    # --------------------------------------------------------
    # Binary feature engineering
    #
    # Yes = 1
    # No  = 0
    # --------------------------------------------------------

    binary_columns = [

        "Subscription Status",

        "Discount Applied",

        "Promo Code Used"
    ]

    for column in binary_columns:

        if column in df.columns:

            normalized = (
                df[column]
                .astype("string")
                .str.strip()
                .str.lower()
            )

            encoded_column = (
                "_" +
                column.replace(" ", "_") +
                "_Encoded"
            )

            df[encoded_column] = (
                normalized
                .map({
                    "yes": 1,
                    "no": 0
                })
            )

    # --------------------------------------------------------
    # Remove rows where the main numerical attributes are
    # completely unusable.
    #
    # We do NOT require every field to be present because
    # categorical fields can be handled separately.
    # --------------------------------------------------------

    important_numeric_columns = [

        column

        for column in [

            "Age",

            "Purchase Amount (USD)",

            "Review Rating",

            "Previous Purchases",

            "Purchase Frequency (per year)"
        ]

        if column in df.columns
    ]

    if important_numeric_columns:

        # Convert infinite values to NaN
        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Remove rows where every important numeric value
        # is missing.
        df = df.dropna(
            subset=important_numeric_columns,
            how="all"
        )

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# AVAILABLE FEATURES
# ============================================================

def get_feature_options(df):
    """
    Return clustering features that are useful for the retail
    customer segmentation problem.

    The returned column names are the actual dataframe columns.
    Categorical features are encoded automatically during
    clustering.
    """

    options = []

    if df is None or df.empty:

        return options

    # --------------------------------------------------------
    # Demographics
    # --------------------------------------------------------

    if "Age" in df.columns:

        options.append({

            "display": "Age",

            "column": "Age",

            "description":
                "Customer age",

            "group":
                "Demographics",

            "type":
                "numeric"
        })

    if "Gender" in df.columns:

        options.append({

            "display": "Gender",

            "column": "Gender",

            "description":
                "Customer gender; encoded using one-hot encoding",

            "group":
                "Demographics",

            "type":
                "categorical"
        })

    # --------------------------------------------------------
    # Purchase Behavior
    # --------------------------------------------------------

    if "Purchase Amount (USD)" in df.columns:

        options.append({

            "display":
                "Purchase Amount (USD)",

            "column":
                "Purchase Amount (USD)",

            "description":
                "Amount spent on the recorded purchase",

            "group":
                "Purchase Behavior",

            "type":
                "numeric"
        })

    if "Previous Purchases" in df.columns:

        options.append({

            "display":
                "Previous Purchases",

            "column":
                "Previous Purchases",

            "description":
                "Number of purchases made previously",

            "group":
                "Purchase Behavior",

            "type":
                "numeric"
        })

    if "Frequency of Purchases" in df.columns:

        options.append({

            "display":
                "Purchase Frequency",

            "column":
                "Frequency of Purchases",

            "description":
                "Purchase frequency category",

            "group":
                "Purchase Behavior",

            "type":
                "categorical"
        })

    if "Purchase Frequency (per year)" in df.columns:

        options.append({

            "display":
                "Purchase Frequency (per year)",

            "column":
                "Purchase Frequency (per year)",

            "description":
                "Estimated purchase frequency converted to purchases per year",

            "group":
                "Purchase Behavior",

            "type":
                "numeric"
        })

    # --------------------------------------------------------
    # Product Preferences
    # --------------------------------------------------------

    if "Category" in df.columns:

        options.append({

            "display":
                "Product Category",

            "column":
                "Category",

            "description":
                "Main product category purchased",

            "group":
                "Product Preferences",

            "type":
                "categorical"
        })

    # --------------------------------------------------------
    # Customer Behavior
    # --------------------------------------------------------

    if "Review Rating" in df.columns:

        options.append({

            "display":
                "Review Rating",

            "column":
                "Review Rating",

            "description":
                "Customer review rating",

            "group":
                "Customer Behavior",

            "type":
                "numeric"
        })

    if "Subscription Status" in df.columns:

        options.append({

            "display":
                "Subscription Status",

            "column":
                "Subscription Status",

            "description":
                "Whether the customer has a subscription",

            "group":
                "Customer Behavior",

            "type":
                "categorical"
        })

    if "Discount Applied" in df.columns:

        options.append({

            "display":
                "Discount Applied",

            "column":
                "Discount Applied",

            "description":
                "Whether a discount was applied",

            "group":
                "Customer Behavior",

            "type":
                "categorical"
        })

    if "Promo Code Used" in df.columns:

        options.append({

            "display":
                "Promo Code Used",

            "column":
                "Promo Code Used",

            "description":
                "Whether a promotional code was used",

            "group":
                "Customer Behavior",

            "type":
                "categorical"
        })

    # --------------------------------------------------------
    # Optional Transaction Behavior
    # --------------------------------------------------------

    if "Shipping Type" in df.columns:

        options.append({

            "display":
                "Shipping Type",

            "column":
                "Shipping Type",

            "description":
                "Shipping method selected by the customer",

            "group":
                "Transaction Behavior",

            "type":
                "categorical"
        })

    if "Payment Method" in df.columns:

        options.append({

            "display":
                "Payment Method",

            "column":
                "Payment Method",

            "description":
                "Payment method used for the purchase",

            "group":
                "Transaction Behavior",

            "type":
                "categorical"
        })

    return options


# ============================================================
# FEATURE MATRIX CREATION
# ============================================================

def create_feature_matrix(df, selected_features):
    """
    Convert the selected raw features into a numerical matrix
    suitable for clustering.

    Numerical columns:
        Used directly.

    Categorical columns:
        Converted using one-hot encoding.

    This avoids incorrect mappings such as:
        Male = 1
        Female = 2
    """

    feature_parts = []

    feature_names = []

    for feature in selected_features:

        if feature not in df.columns:

            continue

        series = df[feature]

        # ----------------------------------------------------
        # Numeric feature
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(series):

            numeric_series = pd.to_numeric(
                series,
                errors="coerce"
            )

            numeric_series = numeric_series.replace(
                [np.inf, -np.inf],
                np.nan
            )

            median_value = (
                numeric_series.median()
            )

            if pd.isna(median_value):

                median_value = 0

            numeric_series = (
                numeric_series.fillna(
                    median_value
                )
            )

            feature_parts.append(
                numeric_series.to_numpy()
                .reshape(-1, 1)
            )

            feature_names.append(
                feature
            )

        # ----------------------------------------------------
        # Categorical feature
        # ----------------------------------------------------

        else:

            categorical_series = (
                series
                .astype("string")
                .fillna("Unknown")
                .str.strip()
            )

            dummy_df = pd.get_dummies(
                categorical_series,
                prefix=feature,
                dtype=float
            )

            # If a categorical column somehow has no usable
            # values, skip it.
            if dummy_df.shape[1] == 0:

                continue

            feature_parts.append(
                dummy_df.to_numpy(
                    dtype=float
                )
            )

            feature_names.extend(
                dummy_df.columns.tolist()
            )

    if not feature_parts:

        raise ValueError(
            "No usable features were found for clustering."
        )

    X = np.hstack(
        feature_parts
    )

    return X, feature_names


# ============================================================
# RUN CLUSTERING
# ============================================================

def run_clustering(
    df,
    selected_features,
    algorithm,
    k
):

    working_df = df.copy()

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if working_df.empty:

        raise ValueError(
            "The dataset is empty."
        )

    # --------------------------------------------------------
    # Validate selected features
    # --------------------------------------------------------

    valid_features = [

        feature

        for feature in selected_features

        if feature in working_df.columns
    ]

    if len(valid_features) < 2:

        raise ValueError(
            "Please select at least two valid attributes."
        )

    # --------------------------------------------------------
    # Validate number of clusters
    # --------------------------------------------------------

    if k < 2:

        raise ValueError(
            "Number of clusters must be at least 2."
        )

    if k >= len(working_df):

        raise ValueError(
            "Number of clusters must be smaller than "
            "the number of records."
        )

    # --------------------------------------------------------
    # Create numerical feature matrix
    # --------------------------------------------------------

    X, encoded_feature_names = create_feature_matrix(
        working_df,
        valid_features
    )

    # --------------------------------------------------------
    # Check for invalid values
    # --------------------------------------------------------

    X = np.asarray(
        X,
        dtype=float
    )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # --------------------------------------------------------
    # Clustering algorithm
    # --------------------------------------------------------

    algorithm = str(
        algorithm
    ).lower()

    if algorithm == "hierarchical":

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward"
        )

        labels = model.fit_predict(
            X_scaled
        )

        inertia = None

    else:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(
            X_scaled
        )

        inertia = float(
            model.inertia_
        )

    # --------------------------------------------------------
    # Add cluster labels
    # --------------------------------------------------------

    working_df["Cluster"] = labels

    # --------------------------------------------------------
    # PCA visualization
    # --------------------------------------------------------

    if X_scaled.shape[1] >= 2:

        pca = PCA(
            n_components=2,
            random_state=42
        )

        pca_values = pca.fit_transform(
            X_scaled
        )

    else:

        pca_values = np.column_stack([

            X_scaled[:, 0],

            np.zeros(
                len(X_scaled)
            )
        ])

    # --------------------------------------------------------
    # PCA points
    # --------------------------------------------------------

    pca_points = []

    for index in range(
        len(working_df)
    ):

        pca_points.append({

            "x":
                round(
                    float(
                        pca_values[
                            index,
                            0
                        ]
                    ),
                    4
                ),

            "y":
                round(
                    float(
                        pca_values[
                            index,
                            1
                        ]
                    ),
                    4
                ),

            "cluster":
                int(
                    labels[index]
                )
        })

    # --------------------------------------------------------
    # Cluster counts
    # --------------------------------------------------------

    counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    cluster_counts = {

        str(int(cluster)):
            int(count)

        for cluster, count
        in counts.items()
    }

    # --------------------------------------------------------
    # Cluster summary
    #
    # We summarize useful original attributes rather than
    # the generated one-hot columns.
    # --------------------------------------------------------

    summary_columns = [

        column

        for column in [

            "Age",

            "Purchase Amount (USD)",

            "Previous Purchases",

            "Purchase Frequency (per year)",

            "Review Rating"
        ]

        if column in working_df.columns
    ]

    summary = []

    grouped = working_df.groupby(
        "Cluster"
    )

    for cluster, group in grouped:

        row = {

            "cluster":
                int(cluster),

            "customers":
                int(len(group))
        }

        # ----------------------------------------------------
        # Numerical averages
        # ----------------------------------------------------

        for column in summary_columns:

            numeric_values = pd.to_numeric(
                group[column],
                errors="coerce"
            )

            if numeric_values.notna().any():

                row[column] = round(
                    float(
                        numeric_values.mean()
                    ),
                    2
                )

        # ----------------------------------------------------
        # Categorical profiles
        # ----------------------------------------------------

        if "Gender" in group.columns:

            row["gender"] = get_mode_value(
                group["Gender"]
            )

        if "Category" in group.columns:

            row["category"] = get_mode_value(
                group["Category"]
            )

        if "Subscription Status" in group.columns:

            row["subscription"] = get_mode_value(
                group["Subscription Status"]
            )

        if "Discount Applied" in group.columns:

            row["discount"] = get_mode_value(
                group["Discount Applied"]
            )

        if "Promo Code Used" in group.columns:

            row["promo_code"] = get_mode_value(
                group["Promo Code Used"]
            )

        # ----------------------------------------------------
        # Automatically generated segment description
        # ----------------------------------------------------

        row["segment"] = generate_segment_name(
            group
        )

        summary.append(
            row
        )

    # --------------------------------------------------------
    # Elbow data
    # --------------------------------------------------------

    elbow_x, elbow_y = calculate_elbow(
        X_scaled
    )

    # --------------------------------------------------------
    # Selected feature display names
    # --------------------------------------------------------

    selected_feature_names = [

        get_display_name(feature)

        for feature in valid_features
    ]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    return {

        "algorithm":
            (
                "Hierarchical Clustering"
                if algorithm == "hierarchical"
                else "K-Means"
            ),

        "k":
            int(k),

        "selected_features":
            selected_feature_names,

        "encoded_features":
            encoded_feature_names,

        "elbow_x":
            elbow_x,

        "elbow_y":
            elbow_y,

        "cluster_counts":
            cluster_counts,

        "pca_points":
            pca_points,

        "summary":
            summary,

        "inertia":
            inertia,

        "total_customers":
            len(working_df)
    }


# ============================================================
# CLUSTER MODE DESCRIPTION
# ============================================================

def get_mode_value(series):
    """
    Return the most common non-empty categorical value.
    """

    cleaned = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    cleaned = cleaned[
        cleaned != ""
    ]

    if cleaned.empty:

        return "Unknown"

    mode_values = cleaned.mode()

    if mode_values.empty:

        return "Unknown"

    return str(
        mode_values.iloc[0]
    )


# ============================================================
# AUTOMATIC CLUSTER SEGMENT NAME
# ============================================================

def generate_segment_name(group):
    """
    Generate a simple descriptive name for a cluster based
    on its purchase behavior.

    The labels are derived from the actual cluster profile;
    they are not used by the clustering algorithm itself.
    """

    names = []

    # --------------------------------------------------------
    # Spending behavior
    # --------------------------------------------------------

    if "Purchase Amount (USD)" in group.columns:

        values = pd.to_numeric(
            group["Purchase Amount (USD)"],
            errors="coerce"
        )

        if values.notna().any():

            mean_purchase = values.mean()

            if mean_purchase >= 70:

                names.append(
                    "High-Spending"
                )

            elif mean_purchase <= 35:

                names.append(
                    "Low-Spending"
                )

            else:

                names.append(
                    "Moderate-Spending"
                )

    # --------------------------------------------------------
    # Purchase frequency
    # --------------------------------------------------------

    if "Purchase Frequency (per year)" in group.columns:

        values = pd.to_numeric(
            group["Purchase Frequency (per year)"],
            errors="coerce"
        )

        if values.notna().any():

            mean_frequency = values.mean()

            if mean_frequency >= 26:

                names.append(
                    "Frequent"
                )

            elif mean_frequency <= 4:

                names.append(
                    "Occasional"
                )

    # --------------------------------------------------------
    # Subscription
    # --------------------------------------------------------

    if "Subscription Status" in group.columns:

        subscription_values = (
            group["Subscription Status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        yes_ratio = (
            subscription_values
            .eq("yes")
            .mean()
        )

        if yes_ratio >= 0.6:

            names.append(
                "Subscribers"
            )

    # --------------------------------------------------------
    # Discount behavior
    # --------------------------------------------------------

    if "Discount Applied" in group.columns:

        discount_values = (
            group["Discount Applied"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        discount_ratio = (
            discount_values
            .eq("yes")
            .mean()
        )

        if discount_ratio >= 0.6:

            names.append(
                "Discount-Oriented"
            )

    # --------------------------------------------------------
    # Previous purchase behavior
    # --------------------------------------------------------

    if "Previous Purchases" in group.columns:

        values = pd.to_numeric(
            group["Previous Purchases"],
            errors="coerce"
        )

        if values.notna().any():

            mean_previous = values.mean()

            if mean_previous >= 25:

                names.append(
                    "Repeat Buyers"
                )

    # --------------------------------------------------------
    # Default label
    # --------------------------------------------------------

    if not names:

        return "General Customer Segment"

    # Keep the label concise
    return " / ".join(
        names[:2]
    )


# ============================================================
# ELBOW METHOD
# ============================================================

def calculate_elbow(X_scaled):

    X_scaled = np.asarray(
        X_scaled,
        dtype=float
    )

    number_of_records = len(
        X_scaled
    )

    # At least two records are required for meaningful
    # clustering.
    if number_of_records < 2:

        return [1], [0.0]

    max_k = min(
        10,
        number_of_records - 1
    )

    x_values = list(
        range(
            1,
            max_k + 1
        )
    )

    y_values = []

    for k in x_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(
            X_scaled
        )

        y_values.append(
            round(
                float(
                    model.inertia_
                ),
                4
            )
        )

    return (
        x_values,
        y_values
    )


# ============================================================
# DISPLAY NAME
# ============================================================

def get_display_name(feature):

    mapping = {

        "Age":
            "Age",

        "Gender":
            "Gender",

        "Purchase Amount (USD)":
            "Purchase Amount (USD)",

        "Previous Purchases":
            "Previous Purchases",

        "Frequency of Purchases":
            "Purchase Frequency",

        "Purchase Frequency (per year)":
            "Purchase Frequency (per year)",

        "Category":
            "Product Category",

        "Review Rating":
            "Review Rating",

        "Subscription Status":
            "Subscription Status",

        "Discount Applied":
            "Discount Applied",

        "Promo Code Used":
            "Promo Code Used",

        "Shipping Type":
            "Shipping Type",

        "Payment Method":
            "Payment Method"
    }

    return mapping.get(
        feature,
        feature
    )