import pandas as pd
import numpy as np
import shap
import joblib

DROP_COLUMNS = [
    'school', 'famsize', 'Pstatus', 'reason', 'guardian', 'traveltime',
    'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
    'Walc', 'health', 'G3', 'Mjob', 'Fjob'
]

NUMERIC_COLS = ['age', 'studytime', 'failures', 'G1', 'G2', 'Fedu', 'Medu']
BINARY_COLS  = ['sex', 'address', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher']

FEATURE_LABELS = {
    "G1": "Period 1 Grade",
    "G2": "Period 2 Grade",
    "failures": "Past Failures",
    "absences": "Absences",
    "studytime": "Study Time",
    "age": "Age",
    "Medu": "Mother's Education",
    "Fedu": "Father's Education",
    "grade_drop": "Grade Drop (G2-G1)",
    "failure_x_nosupport": "Failures × No Support",
    "G1_x_G2": "G1 × G2 Interaction",
    "support_deficit": "Support Deficit",
    "parent_edu_index": "Parent Education Index",
    "disadvantage_score": "Disadvantage Score",
}


def feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['grade_drop'] = df['G2'] - df['G1']
    df['failure_x_nosupport'] = df['failures'] * (
        (df['schoolsup'] == 'no').astype(int) +
        (df['famsup'] == 'no').astype(int)
    )
    df['G1_x_G2'] = df['G1'] * df['G2']
    df['support_deficit'] = (
        (df['schoolsup'] == 'no').astype(int) +
        (df['famsup']    == 'no').astype(int) +
        (df['paid']      == 'no').astype(int)
    )
    df['parent_edu_index'] = df['Medu'] + df['Fedu']
    df['disadvantage_score'] = (
        (df['parent_edu_index'] < 4).astype(int) + df['support_deficit']
    )
    # Drop G3 only if present (not present during inference)
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(cols_to_drop, axis=1)
    return df


def get_risk_tier(prob: float) -> str:
    if prob >= 0.65:   return "HIGH"
    elif prob >= 0.35: return "MEDIUM"
    else:              return "LOW"


def get_shap_reasons(model, input_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Returns top-N SHAP reasons as list of {feature, value, direction, importance}"""
    preprocessor = model.named_steps['preprocessing']
    clf          = model.named_steps['model']

    X_transformed = preprocessor.transform(input_df)

    # Get feature names after preprocessing
    num_names = NUMERIC_COLS.copy()
    log_names = ['absences']
    try:
        bin_names = list(
            preprocessor.named_transformers_['bin']
            .named_steps['onehot']
            .get_feature_names_out(BINARY_COLS)
        )
    except Exception:
        bin_names = []

    all_feature_names = num_names + log_names + bin_names

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)

    # For binary classification shap may return list
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    indices    = np.argsort(np.abs(sv))[::-1][:top_n]
    reasons    = []
    raw_values = X_transformed[0] if hasattr(X_transformed, '__getitem__') else X_transformed.toarray()[0]

    for i in indices:
        fname = all_feature_names[i] if i < len(all_feature_names) else f"feature_{i}"
        label = FEATURE_LABELS.get(fname.split('_')[0], fname)
        # Try to get more descriptive label
        for k, v in FEATURE_LABELS.items():
            if fname.startswith(k):
                label = v
                break
        reasons.append({
            "feature": fname,
            "label":   label,
            "shap":    round(float(sv[i]), 4),
            "direction": "increases risk" if sv[i] > 0 else "reduces risk",
        })

    return reasons
