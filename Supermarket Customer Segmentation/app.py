from flask import Flask, render_template, request
import pandas as pd
import os

from clustering import (
    prepare_dataset,
    get_feature_options,
    run_clustering,
    calculate_elbow
)

app = Flask(__name__)

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

DATA_FOLDER = os.path.join("data")

# Your new retail customer dataset
DEFAULT_DATASET = os.path.join(
    DATA_FOLDER,
    "shopping_behavior_updated.csv"
)

# ---------------------------------------------------------
# GLOBAL DATASET STATE
# ---------------------------------------------------------

current_dataset = None
current_filename = "shopping_behavior_updated.csv"


# ---------------------------------------------------------
# LOAD DEFAULT DATASET
# ---------------------------------------------------------

def load_default_dataset():
    """
    Load the default retail customer dataset.
    Feature engineering is handled by clustering.py.
    """

    global current_dataset

    if os.path.exists(DEFAULT_DATASET):

        try:
            df = pd.read_csv(DEFAULT_DATASET)

            if not df.empty:
                current_dataset = prepare_dataset(df)
            else:
                current_dataset = pd.DataFrame()

        except Exception as e:

            print(f"Error loading default dataset: {e}")
            current_dataset = pd.DataFrame()

    else:

        print(
            f"Default dataset not found: {DEFAULT_DATASET}"
        )

        current_dataset = pd.DataFrame()

    return current_dataset


# ---------------------------------------------------------
# LOAD DATASET WHEN APPLICATION STARTS
# ---------------------------------------------------------

load_default_dataset()


# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    global current_dataset
    global current_filename

    message = None
    error = None
    results = None

    # =====================================================
    # UPLOAD DATASET
    # =====================================================

    if (
        request.method == "POST"
        and request.form.get("action") == "upload"
    ):

        uploaded_file = request.files.get("dataset")

        if not uploaded_file or not uploaded_file.filename:

            error = "Please choose a CSV file first."

        elif not uploaded_file.filename.lower().endswith(".csv"):

            error = "Please upload a CSV file only."

        else:

            try:

                df = pd.read_csv(uploaded_file)

                if df.empty:

                    error = "The uploaded CSV file is empty."

                else:

                    # -------------------------------------
                    # Prepare dataset
                    # -------------------------------------

                    current_dataset = prepare_dataset(df)

                    current_filename = uploaded_file.filename

                    message = (
                        f"Dataset '{uploaded_file.filename}' "
                        f"uploaded successfully."
                    )

            except Exception as e:

                error = (
                    f"Could not read the CSV file: {str(e)}"
                )

    # =====================================================
    # RESET DATASET
    # =====================================================

    if (
        request.method == "POST"
        and request.form.get("action") == "reset"
    ):

        current_filename = "shopping_behavior_updated.csv"

        load_default_dataset()

        if current_dataset is not None and not current_dataset.empty:

            message = (
                "Default retail customer dataset restored."
            )

        else:

            error = (
                "Could not load the default dataset. "
                "Please check the CSV file inside the data folder."
            )

    # =====================================================
    # RUN CLUSTERING
    # =====================================================

    if (
        request.method == "POST"
        and request.form.get("action") == "cluster"
    ):

        if (
            current_dataset is None
            or current_dataset.empty
        ):

            error = (
                "Please upload or load a dataset first."
            )

        else:

            # ---------------------------------------------
            # Selected features from the UI
            # ---------------------------------------------

            selected_features = request.form.getlist(
                "features"
            )

            # ---------------------------------------------
            # Selected algorithm
            # ---------------------------------------------

            algorithm = request.form.get(
                "algorithm",
                "kmeans"
            )

            # ---------------------------------------------
            # Number of clusters
            # ---------------------------------------------

            try:

                k = int(
                    request.form.get(
                        "k",
                        5
                    )
                )

            except (ValueError, TypeError):

                k = 5

            # ---------------------------------------------
            # Validation
            # ---------------------------------------------

            if len(selected_features) < 2:

                error = (
                    "Please select at least two attributes "
                    "for clustering."
                )

            elif k < 2 or k > 10:

                error = (
                    "Number of clusters must be between "
                    "2 and 10."
                )

            else:

                try:

                    # -------------------------------------
                    # Run clustering
                    # -------------------------------------

                    results = run_clustering(
                        current_dataset,
                        selected_features,
                        algorithm,
                        k
                    )

                    message = (
                        f"{algorithm.title()} clustering "
                        f"completed successfully using "
                        f"{len(selected_features)} attributes."
                    )

                except Exception as e:

                    error = (
                        f"Clustering failed: {str(e)}"
                    )

    # =====================================================
    # DATASET SAFETY CHECK
    # =====================================================

    if current_dataset is None:

        current_dataset = pd.DataFrame()

    # =====================================================
    # FEATURE OPTIONS
    # =====================================================

    try:

        feature_options = get_feature_options(
            current_dataset
        )

    except Exception as e:

        print(
            f"Error generating feature options: {e}"
        )

        feature_options = []

    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    preview_columns = [
        column
        for column in current_dataset.columns
        if not str(column).startswith("_")
    ]

    preview_df = (
        current_dataset[
            preview_columns
        ].head(8)
        if preview_columns
        else pd.DataFrame()
    )

    preview_columns = (
        preview_df.columns.tolist()
    )

    preview_rows = (
        preview_df.to_dict(
            orient="records"
        )
        if not preview_df.empty
        else []
    )

    # =====================================================
    # FEATURE COUNT
    # =====================================================

    clustering_feature_count = len(
        feature_options
    )

    # =====================================================
    # RENDER PAGE
    # =====================================================

    return render_template(
        "index.html",

        dataset=current_dataset,

        dataset_name=current_filename,

        record_count=len(
            current_dataset
        ),

        feature_count=clustering_feature_count,

        feature_options=feature_options,

        preview_columns=preview_columns,

        preview_rows=preview_rows,

        results=results,

        message=message,

        error=error
    )


# ---------------------------------------------------------
# RUN FLASK APPLICATION
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )