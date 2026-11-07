"""
Common settings for all scripts; resolve paths from the repository root.
"""
from pathlib import Path

# paths
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "faa_2016_2026.csv.gz"
PROCESSED_DIR = ROOT / "data" / "processed"
ANALYTIC_FILE = PROCESSED_DIR / "analytic.csv.gz"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"

RANDOM_STATE = 42          # Use a shared seed to make every random step reproducible.

TARGET = "INDICATED_DAMAGE"

# temporal split
TRAIN_YEARS = range(2016, 2024)   # 2016-2023
VALID_YEARS = [2024]
TEST_YEARS = [2025, 2026]

# leakage audit
# Exclude post-strike outcome columns to prevent leakage.
LEAKAGE_COLS = (
    ["DAMAGE_LEVEL",  # The damage label is used for filtering only.
     "AOS", "COST_REPAIRS", "COST_OTHER",
     "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ",
     "EFFECT", "EFFECT_OTHER",
     "NR_INJURIES", "NR_FATALITIES",
     "REMARKS", "COMMENTS", "OTHER_SPECIFY",  # Narrative fields completed after the strike.
     "IMAGE", "LUPDATE", "SOURCE", "PERSON",
     "REMAINS_COLLECTED", "REMAINS_SENT",
     "LOCATION"]  # Post-event location text, often 'FOUND AT <airport>'.
    + [f"DAM_{p}" for p in ["RAD", "WINDSHLD", "NOSE", "ENG1", "ENG2", "ENG3",
                            "ENG4", "PROP", "WING_ROT", "FUSE", "LG", "TAIL",
                            "LGHTS", "OTHER"]]  # Damage indicators for individual aircraft parts.
)

# ID fields, redundant columns and fields with almost no data.
DROP_COLS = [
    "INDEX_NR", "REPORTED_NAME", "REPORTED_TITLE", "TRANSFER", "BIRD_BAND_NUMBER",
    "REG", "FLT", "PRECIPITATION", "ENROUTE_STATE",
    "AIRPORT", "OPERATOR", "SPECIES_ID",  # Fields duplicating AIRPORT_ID, OPID or SPECIES.
    "AMA", "AMO", "EMA", "EMO", "ENG_1_POS", "ENG_2_POS", "ENG_3_POS", "ENG_4_POS",
    "STATE", "RUNWAY", "OUT_OF_RANGE_SPECIES",
]

# feature sets
# Set A uses information available before a strike for prevention.
NUMERIC_A = ["HEIGHT", "SPEED", "DISTANCE", "LATITUDE", "LONGITUDE",
             "MONTH_SIN", "MONTH_COS", "HOUR_SIN", "HOUR_COS"]
ORDINAL_A = ["AC_MASS", "NUM_ENGS", "SIZE_ORD", "NUM_STRUCK_ORD", "NUM_SEEN_ORD"]
CATEG_A = ["PHASE_OF_FLIGHT", "TIME_OF_DAY", "SKY", "TYPE_ENG", "AC_CLASS",
           "WARNED", "FAAREGION"]
# Use target encoding for fields with many categories.
HIGHCARD_A = ["SPECIES", "AIRPORT_ID", "AIRCRAFT", "OPID"]

# Set B = A plus aircraft strike locations known only after inspection, for triage.
STRIKE_COLS = [f"STR_{p}" for p in ["RAD", "WINDSHLD", "NOSE", "ENG1", "ENG2", "ENG3",
                                     "ENG4", "PROP", "WING_ROT", "FUSE", "LG", "TAIL",
                                     "LGHTS", "OTHER"]]
EXTRA_B = STRIKE_COLS + ["ANY_INGESTION"]


def feature_lists(feature_set="A"):
    """Return column lists in order: numeric, ordinal, categorical, high-cardinality."""
    numeric = list(NUMERIC_A)
    if feature_set == "B":
        numeric += EXTRA_B
    return numeric, list(ORDINAL_A), list(CATEG_A), list(HIGHCARD_A)
