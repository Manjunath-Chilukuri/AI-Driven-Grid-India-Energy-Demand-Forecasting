import os
import pandas as pd

REPORT_DIR = "outputs/reports"
REPORT_FILE = os.path.join(REPORT_DIR, "Model_Performance.csv")


def save_model_metrics(model_name, mae, rmse, r2):
    """
    Add or update a model's performance metrics in the report.
    """

    os.makedirs(REPORT_DIR, exist_ok=True)

    new_row = pd.DataFrame({
        "Model": [model_name],
        "MAE": [round(float(mae), 4)],
        "RMSE": [round(float(rmse), 4)],
        "R²": [round(float(r2), 6)]
    })

    if os.path.exists(REPORT_FILE):

        report = pd.read_csv(REPORT_FILE)

        report = report[report["Model"] != model_name]

        report = pd.concat(
            [report, new_row],
            ignore_index=True
        )

    else:

        report = new_row

    report.to_csv(
        REPORT_FILE,
        index=False
    )

    print(f"{model_name} metrics saved successfully.")