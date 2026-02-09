import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings

# Ignore pandas FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# Helper Functions
# =============================================================================

def _parse_date(date_value):
    """Parse date to datetime object"""
    if isinstance(date_value, (int, float)):
        date_str = str(int(date_value)).zfill(6)
        return datetime.strptime(date_str, "%Y%m")
    elif isinstance(date_value, str):
        if date_value.isdigit():
            date_str = date_value.zfill(6)
            return datetime.strptime(date_str, "%Y%m")
        else:
            return pd.to_datetime(date_value, errors="coerce")
    elif isinstance(date_value, pd.Timestamp):
        return date_value
    return pd.NaT

# =============================================================================
# Environment Handlers & Optimization
# =============================================================================

def optimize_macro_data(macro_data):
    """
    Convert macro DataFrame to dict for O(1) lookup.
    Returns: { 'indicator_name': { timestamp: value } }
    """
    optimized = {}
    for name, df in macro_data.items():
        if not pd.api.types.is_datetime64_any_dtype(df['observation_date']):
             df = df.copy()
             df['observation_date'] = pd.to_datetime(df['observation_date'])

        val_col = None
        if name == 'cpi': val_col = 'inflation_rate_yoy'
        elif name == 'interest': val_col = 'FEDFUNDS'
        elif name == 'unemployment': val_col = 'UNRATE'

        if val_col and val_col in df.columns:
            optimized[name] = df.set_index('observation_date')[val_col].to_dict()

    return optimized

def _get_vals_optimized(macro_dict, indicator_name, target_dates):
    """Get values from optimized dict for given dates"""
    data_map = macro_dict.get(indicator_name, {})
    vals = []
    found_all = True
    for d in target_dates:
        val = data_map.get(d)
        if val is None:
            found_all = False
            break
        vals.append(round(val, 2))

    return vals if found_all else None

def build_environment_prompt(date_value, macro_source, selected_features=None):
    """
    Build macro environment prompt (JSON format).
    """
    if selected_features is None or 'all' in selected_features or (isinstance(selected_features, str) and selected_features == 'all'):
        selected_features = ['inflation', 'interest_rate', 'unemployment']

    ref_date = _parse_date(date_value)
    if pd.isna(ref_date):
        return "{}"

    is_optimized = isinstance(macro_source.get('cpi'), dict)

    shifted_ref = ref_date - relativedelta(months=1)
    past_4_months_dates = [
        (shifted_ref - relativedelta(months=i)).replace(day=1)
        for i in reversed(range(4))
    ]

    data = {}

    def get_data(indicator):
        if is_optimized:
            return _get_vals_optimized(macro_source, indicator, past_4_months_dates)
        else:
            return None

    if 'inflation' in selected_features:
        vals = get_data('cpi')
        if vals and len(vals) == 4:
            data['inflation_rate'] = vals

    if 'interest_rate' in selected_features:
        vals = get_data('interest')
        if vals and len(vals) == 4:
            data['federal_funds_rate'] = vals

    if 'unemployment' in selected_features:
        vals = get_data('unemployment')
        if vals and len(vals) == 4:
            data['unemployment_rate'] = vals

    return json.dumps(data)


# =============================================================================
# Profile Handlers & Registry
# =============================================================================

def _h_age(row):
    return int(row['Q32']) if pd.notnull(row.get('Q32')) else None

def _h_gender(row):
    gender_map = {1: "Female", 2: "Male"}
    return gender_map.get(int(row['Q33']), 'unknown') if pd.notnull(row.get('Q33')) else None

def _h_education(row):
    edu_map = {
        1: "Less than high school", 2: "High school diploma", 3: "Some college but no degree",
        4: "Associate degree", 5: "Bachelor's degree", 6: "Master's degree",
        7: "Doctoral degree", 8: "Professional degree", 9: "Other"
    }
    return edu_map.get(int(row['Q36']), 'Unknown') if pd.notnull(row.get('Q36')) else None

def _h_marital(row):
    marital_map = {1: "Married or partnered", 2: "Single"}
    return marital_map.get(int(row['Q38']), 'Unknown') if pd.notnull(row.get('Q38')) else None

def _h_state(row):
    area_map = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
        "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
        "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
        "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
        "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
        "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
        "NC": "North Carolina", "ND": "North Dakota", "NE": "Nebraska", "NV": "Nevada",
        "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
        "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
        "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
        "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
        "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
        "99": "Outside the United States"
    }
    if pd.notnull(row.get('_STATE')):
        state_key = str(row.get('_STATE'))
        state_name = area_map.get(state_key, 'an unknown state')
        return state_name
    return None

def _h_housing(row):
    homeown_map = {1: "Own", 2: "Rent", 3: "Other"}
    return homeown_map.get(int(row['Q43']), 'Unknown') if pd.notnull(row.get('Q43')) else None

def _h_own_other(row):
    if pd.notnull(row.get('Q44')):
        return True if row.get('Q44') == 1 else False
    return None

def _h_health(row):
    health_map = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
    return health_map.get(int(row['Q45b']), 'Unknown') if pd.notnull(row.get('Q45b')) else None

def _h_employment(row):
    q10_options = {
        'Q10_1': "working full-time", 'Q10_2': "working part-time",
        'Q10_3': "not working but would like to work", 'Q10_4': "temporarily laid off",
        'Q10_5': "on sick or other leave", 'Q10_6': "permanently disabled or unable to work",
        'Q10_7': "retired or early retired", 'Q10_8': "student or in training",
        'Q10_9': "homemaker", 'Q10_10': "other"
    }
    selected = []
    for col, desc in q10_options.items():
        if col in row and pd.notnull(row[col]) and row[col] == 1:
            selected.append(desc)

    return selected if selected else None

def _h_income(row):
    income_map = {
        1: "Less than $10,000", 2: "$10,000 to $19,999", 3: "$20,000 to $29,999",
        4: "$30,000 to $39,999", 5: "$40,000 to $49,999", 6: "$50,000 to $59,999",
        7: "$60,000 to $74,999", 8: "$75,000 to $99,999", 9: "$100,000 to $149,999",
        10: "$150,000 to $199,999", 11: "$200,000 or more"
    }
    return income_map.get(int(row['Q47']), 'Unknown') if pd.notnull(row.get('Q47')) else None


PROFILE_REGISTRY = {
    'age': _h_age,
    'gender': _h_gender,
    'education': _h_education,
    'marital_status': _h_marital,
    'state_residence': _h_state,
    'housing_status': _h_housing,
    'own_other_home': _h_own_other,
    'health_status': _h_health,
    'employment_status': _h_employment,
    'income': _h_income
}

def build_profile_prompt(row, selected_features=None):
    """
    Build user profile prompt (JSON format).
    """
    if selected_features is None or 'all' in selected_features or (isinstance(selected_features, str) and selected_features == 'all'):
        selected_features = PROFILE_REGISTRY.keys()

    data = {}
    for feature in selected_features:
        handler = PROFILE_REGISTRY.get(feature)
        if handler:
            try:
                val = handler(row)
                if val is not None:
                    data[feature] = val
            except Exception:
                pass

    return json.dumps(data)
