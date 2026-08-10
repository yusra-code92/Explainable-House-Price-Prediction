from flask import Flask, render_template, request
import pandas as pd
import shap
import pickle
import traceback
import matplotlib

# Flask/server does not have a GUI display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import time


app = Flask(__name__)


# ============================================================
# LOAD MODEL AND SHAP RESOURCES
# ============================================================

model = pickle.load(
    open("catboost_model.pkl", "rb")
)

feature_names = pickle.load(
    open("feature_names.pkl", "rb")
)


# Optional background data
if os.path.exists("background_data.pkl"):
    background_data = pd.read_pickle(
        "background_data.pkl"
    )
else:
    background_data = None


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ====================================================
        # 1. GET USER INPUT
        # ====================================================

        area = request.form.get("area")
        bedrooms = request.form.get("bedrooms")
        bathrooms = request.form.get("bathrooms")
        floors = request.form.get("floors")
        balcony = request.form.get("balcony")
        facing = request.form.get("facing")
        area_type = request.form.get("area_type")


        # ====================================================
        # 2. CHECK EMPTY FIELDS
        # ====================================================

        if not all([
            area,
            bedrooms,
            bathrooms,
            floors,
            balcony,
            facing,
            area_type
        ]):

            return render_template(
                "index.html",
                error="Please fill all the fields."
            )


        # ====================================================
        # 3. CONVERT NUMERIC INPUTS
        # ====================================================

        area = float(area)
        bedrooms = int(bedrooms)
        bathrooms = int(bathrooms)
        floors = int(floors)
        balcony = int(balcony)


        # ====================================================
        # 4. VALIDATE INPUT
        # ====================================================

        if (
            area <= 0
            or bedrooms < 0
            or bathrooms < 0
            or floors < 0
            or balcony < 0
        ):

            return render_template(
                "index.html",
                error="Please enter valid values."
            )


        # ====================================================
        # 5. CREATE INPUT DATAFRAME
        # ====================================================

        user_df = pd.DataFrame(
            [[
                area,
                bedrooms,
                bathrooms,
                floors,
                balcony,
                facing,
                area_type
            ]],
            columns=feature_names
        )


        print("\n======================================")
        print("USER INPUT")
        print(user_df)
        print("======================================")


        # ====================================================
        # 6. MODEL PREDICTION
        # ====================================================

        prediction = float(
            model.predict(user_df)[0]
        )


        # ====================================================
        # 7. GENERATE SHAP VALUES
        # ====================================================

        shap_values = explainer(
            user_df
        )

        values = shap_values.values[0]

        base_value = float(
            shap_values.base_values[0]
        )


        # ====================================================
        # 8. GENERATE LOCAL SHAP WATERFALL
        # ====================================================

        os.makedirs(
            "static",
            exist_ok=True
        )


        waterfall_filename = "waterfall_plot.png"

        waterfall_path = os.path.join(
            "static",
            waterfall_filename
        )


        # ----------------------------------------------------
        # Close all previous figures
        # ----------------------------------------------------

        plt.close("all")


        # ----------------------------------------------------
        # Generate SHAP waterfall
        # ----------------------------------------------------

        shap.plots.waterfall(
            shap_values[0],
            max_display=len(feature_names),
            show=False
        )


        # ----------------------------------------------------
        # Get the actual SHAP figure
        # ----------------------------------------------------

        fig = plt.gcf()


        # ----------------------------------------------------
        # Improve figure size
        # ----------------------------------------------------

        fig.set_size_inches(
            14,
            8
        )


        # ----------------------------------------------------
        # Save plot
        # ----------------------------------------------------

        fig.savefig(
            waterfall_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.5
        )


        # ----------------------------------------------------
        # Close figure
        # ----------------------------------------------------

        plt.close(fig)


        # ----------------------------------------------------
        # Verify file was actually created
        # ----------------------------------------------------

        if not os.path.exists(waterfall_path):

            raise FileNotFoundError(
                "Waterfall plot was not created."
            )


        print(
            "Waterfall plot saved successfully:"
        )

        print(
            os.path.abspath(
                waterfall_path
            )
        )


        # ====================================================
        # CACHE BUSTING
        # ====================================================

        image_timestamp = int(
            time.time()
        )


        # We send filename and timestamp separately.
        # This avoids problems with url_for().
        waterfall_image = waterfall_filename

        waterfall_version = image_timestamp


        # ====================================================
        # 9. CREATE FEATURE CONTRIBUTIONS
        # ====================================================

        feature_importance = []


        for feature, value in zip(
            feature_names,
            values
        ):

            feature_importance.append({

                "feature": feature,

                "impact": float(value)

            })


        # ----------------------------------------------------
        # Sort by absolute SHAP impact
        # ----------------------------------------------------

        feature_importance = sorted(
            feature_importance,
            key=lambda x: abs(x["impact"]),
            reverse=True
        )


        # ----------------------------------------------------
        # Top 5 features for website
        # ----------------------------------------------------

        top_features = feature_importance[:5]


        # ====================================================
        # 10. HUMAN-READABLE AI EXPLANATION
        # ====================================================

        top_three = top_features[:3]


        explanation = (
            f"The predicted house price is "
            f"{prediction:.2f} Crore. "
        )


        if prediction < base_value:

            explanation += (
                "This prediction is lower than "
                "the model's baseline prediction. "
            )

        else:

            explanation += (
                "This prediction is higher than "
                "the model's baseline prediction. "
            )


        explanation += (
            "The three most influential features "
            "for this prediction are "
        )


        feature_text = []


        for item in top_three:

            readable_name = (
                item["feature"]
                .replace("_", " ")
                .title()
            )

            feature_text.append(
                readable_name
            )


        if len(feature_text) == 1:

            explanation += (
                feature_text[0]
                + ". "
            )

        elif len(feature_text) == 2:

            explanation += (
                feature_text[0]
                + " and "
                + feature_text[1]
                + ". "
            )

        else:

            explanation += (
                ", ".join(
                    feature_text[:-1]
                )
                + " and "
                + feature_text[-1]
                + ". "
            )


        explanation += (
            "Positive SHAP values indicate that "
            "a feature increased the predicted price, "
            "while negative SHAP values indicate that "
            "the feature decreased the predicted price."
        )


        # ====================================================
        # 11. SHAP VALIDATION
        # ====================================================

        shap_sum = float(
            values.sum()
        )


        reconstructed_prediction = (
            base_value + shap_sum
        )


        difference = abs(
            prediction -
            reconstructed_prediction
        )


        # ====================================================
        # 12. DEBUG INFORMATION
        # ====================================================

        print("\n======================================")

        print(
            "Prediction :",
            prediction
        )

        print(
            "Base Value :",
            base_value
        )

        print("\nFeature Contributions:")


        for item in feature_importance:

            print(
                f"{item['feature']:15} : "
                f"{item['impact']:.4f}"
            )


        print(
            "\nSum SHAP :",
            shap_sum
        )


        print(
            "Base + SHAP :",
            reconstructed_prediction
        )


        print(
            "Difference :",
            difference
        )


        print(
            "Waterfall file :",
            os.path.abspath(
                waterfall_path
            )
        )


        print("======================================\n")


        # ====================================================
        # 13. SEND DATA TO HTML
        # ====================================================

        return render_template(

            "index.html",


            # ------------------------------------------------
            # Model Prediction
            # ------------------------------------------------

            prediction_text=(
                f"Predicted House Price: "
                f"{prediction:.2f} Crore"
            ),


            # ------------------------------------------------
            # User Inputs
            # ------------------------------------------------

            inputs={

                "area": area,

                "bedrooms": bedrooms,

                "bathrooms": bathrooms,

                "floors": floors,

                "balcony": balcony,

                "facing": facing,

                "area_type": area_type

            },


            # ------------------------------------------------
            # Top SHAP Feature Contributions
            # ------------------------------------------------

            feature_importance=top_features,


            # ------------------------------------------------
            # Waterfall Plot
            # ------------------------------------------------

            waterfall_image=waterfall_image,

            waterfall_version=waterfall_version,


            # ------------------------------------------------
            # AI Explanation
            # ------------------------------------------------

            ai_explanation=explanation,


            # ------------------------------------------------
            # SHAP Validation
            # ------------------------------------------------

            base_value=base_value,

            shap_sum=shap_sum,

            reconstructed_prediction=(
                reconstructed_prediction
            ),

            shap_difference=difference

        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        traceback.print_exc()

        return render_template(

            "index.html",

            error=str(e)

        )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )