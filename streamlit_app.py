import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Network Analysis Dashboard", layout="wide")

CSV_PATH = "complaints_coverage_by_state_malaysia.csv"
GEOJSON_PATH = "malaysia.district.geojson"

SIGNAL_COL = "signal_strength"  # numeric 0-5
DATE_COL = "reported_date"      # used to derive monthly trend

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# =========================
# HELPERS
# =========================
def derive_reported_month_series(df_in: pd.DataFrame) -> pd.Series:
    """Return a Series of month names derived from DATE_COL."""
    if DATE_COL not in df_in.columns:
        return pd.Series([], dtype=str)

    d = pd.to_datetime(df_in[DATE_COL], errors="coerce")
    return d.dt.month_name()

def bucket_signal_strength(s: pd.Series) -> pd.Series:
    """Bucket numeric signal strength (0-5) into network-friendly labels."""
    x = pd.to_numeric(s, errors="coerce").fillna(0)

    def _label(v: float) -> str:
        # 0 = No Signal; 1-2 = Weak; 3 = Moderate; 4-5 = Strong
        if v <= 0:
            return "No Signal"
        if v <= 2:
            return "Weak"
        if v <= 3:
            return "Moderate"
        return "Strong"

    return x.apply(_label)

@st.cache_data(show_spinner=False)
def build_payload():
    # Load dataset
    df = pd.read_csv(CSV_PATH)

    # --- STATUS NORMALIZATION (Malaysia dataset uses Submitted/In Progress/Resolved) ---
    # Keep dashboard keys consistent: Open/In Progress/Closed
    status_counts_raw = df["status"].value_counts(dropna=False).to_dict() if "status" in df.columns else {}

    def _get_status_count(key: str) -> int:
        return int(status_counts_raw.get(key, 0))

    # Map: Submitted -> Open, In Progress -> In Progress, Resolved -> Closed
    total_open = _get_status_count("Submitted")
    total_in_progress = _get_status_count("In Progress")
    total_closed = _get_status_count("Resolved")

    summary = {
        "total_complaints": int(len(df)),
        "total_states": int(df["state"].nunique()) if "state" in df.columns else 0,
        "total_districts": int(df["district"].nunique()) if "district" in df.columns else 0,
        "total_submitted": total_open,
        "total_in_progress": total_in_progress,
        "total_resolved": total_closed,
    }

    # Simple counts
    state = df["state"].value_counts().to_dict() if "state" in df.columns else {}
    district = df["district"].value_counts().to_dict() if "district" in df.columns else {}
    issue = df["issue_type"].value_counts().to_dict() if "issue_type" in df.columns else {}

    # State-Issue matrix
    if set(["state", "issue_type"]).issubset(df.columns):
        ct = pd.crosstab(df["state"], df["issue_type"])
        ct["__total__"] = ct.sum(axis=1)
        ct = ct.sort_values("__total__", ascending=False).drop(columns=["__total__"])
        issue_matrix = {
            "states": list(ct.index),
            "issue_types": list(ct.columns),
            "matrix": {col: ct[col].astype(int).tolist() for col in ct.columns},
        }
    else:
        issue_matrix = {"states": [], "issue_types": [], "matrix": {}}

    # State-Status matrix (force dashboard order: Open/In Progress/Closed)
    # Malaysia status strings are Submitted/In Progress/Resolved -> we normalize columns
    if set(["state", "status"]).issubset(df.columns):
        # Create a normalized status column for matrix
        def norm_status(x):
            if pd.isna(x):
                return "Other"
            x = str(x).strip()
            if x == "Submitted":
                return "Open"
            if x == "In Progress":
                return "In Progress"
            if x == "Resolved":
                return "Closed"
            return x

        df_tmp = df.copy()
        df_tmp["_status_norm_"] = df_tmp["status"].apply(norm_status)

        ct = pd.crosstab(df_tmp["state"], df_tmp["_status_norm_"])
        status_order = ["Open", "In Progress", "Closed"]
        for s in status_order:
            if s not in ct.columns:
                ct[s] = 0
        ct = ct[status_order]
        ct["__total__"] = ct.sum(axis=1)
        ct = ct.sort_values("__total__", ascending=False).drop(columns=["__total__"])
        status_matrix = {
            "states": list(ct.index),
            "statuses": list(ct.columns),
            "matrix": {col: ct[col].astype(int).tolist() for col in ct.columns},
        }
    else:
        status_matrix = {"states": [], "statuses": [], "matrix": {}}

    # State-Signal Strength matrix (bucketed)
    if set(["state", SIGNAL_COL]).issubset(df.columns):
        sig = bucket_signal_strength(df[SIGNAL_COL])
        ct = pd.crosstab(df["state"], sig)

        preferred_order = ["Strong", "Moderate", "Weak", "No Signal"]
        existing = [x for x in preferred_order if x in ct.columns]
        remaining = [x for x in ct.columns if x not in existing]
        signal_order = existing + sorted(remaining)

        ct = ct[signal_order]
        ct["__total__"] = ct.sum(axis=1)
        ct = ct.sort_values("__total__", ascending=False).drop(columns=["__total__"])

        signal_matrix = {
            "states": list(ct.index),
            "signal_strengths": list(ct.columns),
            "matrix": {col: ct[col].astype(int).tolist() for col in ct.columns},
        }
    else:
        signal_matrix = {
            "states": [],
            "signal_strengths": [],
            "matrix": {},
            "error": f"Required columns 'state' and '{SIGNAL_COL}' not found in dataset."
        }

    # Reported months
    months = derive_reported_month_series(df)
    if months.empty:
        reported = {m: 0 for m in MONTH_ORDER}
    else:
        counts = months.astype(str).str.strip().str.title().value_counts().to_dict()
        reported = {m: int(counts.get(m, 0)) for m in MONTH_ORDER}

    # GeoJSON
    geojson = None
    try:
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            geojson = json.load(f)
    except Exception:
        geojson = None

    payload = {
        "summary": summary,
        "state": state,
        "district": district,
        "issue": issue,
        "signal_matrix": signal_matrix,
        "issue_matrix": issue_matrix,
        "status_matrix": status_matrix,
        "reported": reported,
        "geojson": geojson,
    }
    return payload

def main():
    # Optional: hide Streamlit chrome for a cleaner 1:1 look
    st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      header {visibility: hidden;}
      footer {visibility: hidden;}

      /* Remove Streamlit default paddings/margins */
      .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        margin: 0rem !important;
        max-width: 100% !important;
      }
      [data-testid="stAppViewContainer"] {
        padding: 0rem !important;
      }
      [data-testid="stApp"] {
        margin: 0rem !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)


    payload = build_payload()

    # Load HTML template
    with open("index_streamlit.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Inject payload
    html = html.replace("__DASH_PAYLOAD__", json.dumps(payload))

    # Render
    components.html(html, height=1900, scrolling=False)

if __name__ == "__main__":
    main()

