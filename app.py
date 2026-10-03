from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("xgboost_cancer_model.pkl")

selected_genes = pd.read_csv(
    "selected_gene_names.csv"
)["Gene"].tolist()


# ============================================================
# LOAD GENE SYMBOL MAPPING
# ============================================================

gene_mapping = pd.read_csv(
    "gene_symbol_mapping.csv"
)

gene_symbol_dict = dict(
    zip(
        gene_mapping["Ensembl_ID"],
        gene_mapping["Gene_Symbol"]
    )
)


# ============================================================
# BIOLOGICAL INFORMATION
#
# These descriptions provide general biological context.
# They are NOT used by the ML model.
# ============================================================

gene_information = {

    "MYH7": {
        "function":
            "Encodes myosin heavy chain 7, a major component "
            "of the contractile machinery in cardiac and skeletal muscle.",
        "research":
            "MYH7 has been extensively studied in muscle and "
            "cardiac biology. Its importance in this model should "
            "not by itself be interpreted as proof of a cancer-specific role."
    },

    "LENG8": {
        "function":
            "LENG8 is a protein-coding gene with comparatively "
            "limited functional characterization in commonly used annotations.",
        "research":
            "Its appearance among model-important features represents "
            "a computational finding that requires independent biological validation."
    },

    "MYL2": {
        "function":
            "Encodes myosin light chain 2, a component of the "
            "sarcomere involved in muscle contraction.",
        "research":
            "MYL2 is strongly associated with muscle and cardiac biology. "
            "Its SHAP importance here should be interpreted as a model "
            "feature rather than direct evidence of cancer causation."
    },

    "ABCA10": {
        "function":
            "ABCA10 belongs to the ATP-binding cassette transporter "
            "family of proteins.",
        "research":
            "The specific biological substrate and function of ABCA10 "
            "are not completely established. Its model contribution "
            "requires further investigation."
    },

    "ACADVL": {
        "function":
            "Encodes very-long-chain acyl-CoA dehydrogenase, "
            "an enzyme involved in mitochondrial fatty-acid oxidation.",
        "research":
            "ACADVL is involved in cellular energy metabolism. "
            "Its importance in this model does not establish a "
            "specific cancer biomarker relationship."
    },

    "SEC31B": {
        "function":
            "SEC31B is involved in COPII-dependent vesicle formation "
            "and intracellular protein transport.",
        "research":
            "Its biological role is associated with the cellular "
            "secretory and transport machinery. Further validation "
            "is required to determine cancer-specific relevance."
    },

    "EIF2AK1": {
        "function":
            "EIF2AK1 encodes a kinase involved in regulation of "
            "translation initiation during cellular stress.",
        "research":
            "EIF2AK1 is involved in cellular stress-response pathways. "
            "Its contribution in this model should be considered a "
            "computational association rather than clinical evidence."
    },

    "GOLGA6L17P": {
        "function":
            "GOLGA6L17P is a pseudogene-related genomic feature "
            "with limited functional characterization.",
        "research":
            "Its high model contribution is a computational observation. "
            "Additional biological evidence is required before assigning "
            "a cancer-specific interpretation."
    },

    "BOLA2B": {
        "function":
            "BOLA2B is associated with cellular metal and protein "
            "homeostasis-related processes.",
        "research":
            "Its appearance in the model reflects predictive importance "
            "within the current dataset and requires independent validation."
    },

    "RPN1": {
        "function":
            "RPN1 is a component of the oligosaccharyltransferase "
            "complex involved in protein glycosylation in the endoplasmic reticulum.",
        "research":
            "RPN1 participates in protein-processing pathways. "
            "Its SHAP contribution does not by itself establish "
            "a cancer biomarker relationship."
    },

    "PDZD11": {
        "function":
            "PDZD11 encodes a protein involved in protein-interaction "
            "and membrane-associated cellular processes.",
        "research":
            "Its biological role is still being characterized. "
            "The current model identifies it as a computationally "
            "important feature requiring further validation."
    },

    "CCNB1": {
        "function":
            "CCNB1 encodes Cyclin B1, a key regulator of the "
            "cell cycle and progression through mitosis.",
        "research":
            "Cell-cycle regulation is highly relevant to cancer biology, "
            "and CCNB1 has been investigated extensively in cancer research. "
            "However, model importance alone does not establish clinical "
            "biomarker validity."
    }
}


# ============================================================
# STARTUP
# ============================================================

print("========================================")
print("CANCER BIOMARKER WEBSITE")
print("========================================")
print("XGBoost model loaded")
print("Selected genes:", len(selected_genes))
print("Gene mapping loaded")
print("SHAP contribution mode: XGBoost pred_contribs")
print("Gene explanation module loaded")
print("Flask server ready")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# DOWNLOAD SAMPLE CSV
# ============================================================

@app.route("/download-sample")
def download_sample():

    return app.send_static_file(
        "test_sample.csv"
    )


# ============================================================
# PREDICTION + FEATURE CONTRIBUTIONS
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # CHECK UPLOADED FILE
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify({
                "error": "No CSV file uploaded."
            })

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "error": "No file selected."
            })


        # ----------------------------------------------------
        # READ CSV
        # ----------------------------------------------------

        sample = pd.read_csv(file)

        print("\nReceived file:", file.filename)
        print("Input shape:", sample.shape)


        # ----------------------------------------------------
        # CHECK REQUIRED GENES
        # ----------------------------------------------------

        missing_genes = [
            gene
            for gene in selected_genes
            if gene not in sample.columns
        ]

        if missing_genes:

            return jsonify({
                "error":
                    f"Missing {len(missing_genes)} required genes."
            })


        # ----------------------------------------------------
        # MAINTAIN EXACT FEATURE ORDER
        # ----------------------------------------------------

        X = sample[selected_genes]


        # ----------------------------------------------------
        # CONVERT TO NUMERIC
        # ----------------------------------------------------

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )


        # ----------------------------------------------------
        # MISSING VALUE CHECK
        # ----------------------------------------------------

        if X.isnull().any().any():

            return jsonify({
                "error":
                    "Input contains missing or non-numeric gene values."
            })


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(X)[0]

        probability = model.predict_proba(X)[0]

        normal_probability = float(
            probability[0]
        )

        cancer_probability = float(
            probability[1]
        )

        result = (
            "CANCER"
            if prediction == 1
            else "NORMAL"
        )


        # ====================================================
        # XGBOOST FEATURE CONTRIBUTIONS
        # ====================================================

        booster = model.get_booster()

        dmatrix = xgb.DMatrix(
            X,
            feature_names=selected_genes
        )

        contributions = booster.predict(
            dmatrix,
            pred_contribs=True
        )

        # Last column is the base/bias contribution
        sample_contributions = contributions[0][:-1]


        # ====================================================
        # BUILD CONTRIBUTION TABLE
        # ====================================================

        contribution_df = pd.DataFrame({

            "ensembl_id":
                selected_genes,

            "shap_value":
                sample_contributions,

            "abs_shap":
                np.abs(sample_contributions),

            "expression":
                X.iloc[0].values

        })


        # ----------------------------------------------------
        # SORT BY ABSOLUTE CONTRIBUTION
        # ----------------------------------------------------

        top_genes = (
            contribution_df
            .sort_values(
                "abs_shap",
                ascending=False
            )
            .head(10)
        )


        # ====================================================
        # PREPARE DETAILED EXPLANATIONS
        # ====================================================

        shap_results = []


        for rank, (_, row) in enumerate(
            top_genes.iterrows(),
            start=1
        ):

            ensembl_id = row[
                "ensembl_id"
            ]

            shap_value = float(
                row["shap_value"]
            )

            expression_value = float(
                row["expression"]
            )


            # ------------------------------------------------
            # GENE SYMBOL
            # ------------------------------------------------

            symbol = gene_symbol_dict.get(
                ensembl_id
            )

            if (
                pd.isna(symbol)
                or not str(symbol).strip()
            ):

                display_name = "Not mapped"

            else:

                display_name = str(
                    symbol
                )


            # ------------------------------------------------
            # CONTRIBUTION DIRECTION
            # ------------------------------------------------

            if shap_value > 0:

                direction = "Toward Cancer"

                direction_icon = "↑"

                technical_explanation = (
                    f"{display_name} contributed positively "
                    "to the XGBoost prediction for this sample. "
                    "Its feature contribution moved the model "
                    "output toward the cancer class."
                )

            elif shap_value < 0:

                direction = "Toward Normal"

                direction_icon = "↓"

                technical_explanation = (
                    f"{display_name} contributed negatively "
                    "to the XGBoost prediction for this sample. "
                    "Its feature contribution moved the model "
                    "output toward the normal class."
                )

            else:

                direction = "Neutral"

                direction_icon = "→"

                technical_explanation = (
                    f"{display_name} had approximately zero "
                    "individual contribution to the model "
                    "prediction for this sample."
                )


            # ------------------------------------------------
            # CONTRIBUTION STRENGTH
            # ------------------------------------------------

            absolute_contribution = abs(
                shap_value
            )

            if absolute_contribution >= 1.0:

                strength = "Very Strong"

            elif absolute_contribution >= 0.5:

                strength = "Strong"

            elif absolute_contribution >= 0.1:

                strength = "Moderate"

            else:

                strength = "Low"


            # ------------------------------------------------
            # BIOLOGICAL INFORMATION
            # ------------------------------------------------

            if display_name in gene_information:

                biological_function = (
                    gene_information[
                        display_name
                    ]["function"]
                )

                research_context = (
                    gene_information[
                        display_name
                    ]["research"]
                )

            else:

                biological_function = (
                    "A reliable biological description "
                    "is not currently available in the "
                    "project's local gene-information table."
                )

                research_context = (
                    "No cancer-specific interpretation "
                    "is assigned automatically. Additional "
                    "gene annotation and literature validation "
                    "are required."
                )


            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            interpretation = (
                "This gene is a model-derived candidate "
                "feature. Its SHAP contribution explains "
                "the model's decision for this sample, "
                "but it does not prove that the gene causes "
                "cancer or that it is a clinically validated "
                "biomarker."
            )


            # ------------------------------------------------
            # ADD RESULT
            # ------------------------------------------------

            shap_results.append({

                "rank":
                    rank,

                "gene":
                    display_name,

                "ensembl_id":
                    ensembl_id,

                "shap_value":
                    round(
                        shap_value,
                        6
                    ),

                "absolute_shap":
                    round(
                        absolute_contribution,
                        6
                    ),

                "expression":
                    round(
                        expression_value,
                        6
                    ),

                "direction":
                    direction,

                "direction_icon":
                    direction_icon,

                "strength":
                    strength,

                "technical_explanation":
                    technical_explanation,

                "biological_function":
                    biological_function,

                "research_context":
                    research_context,

                "interpretation":
                    interpretation

            })


        # ====================================================
        # RETURN JSON
        # ====================================================

        return jsonify({

            "prediction":
                result,

            "cancer_probability":
                round(
                    cancer_probability * 100,
                    2
                ),

            "normal_probability":
                round(
                    normal_probability * 100,
                    2
                ),

            "genes_used":
                len(selected_genes),

            "shap_genes":
                shap_results

        })


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return jsonify({
            "error":
                str(e)
        })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )