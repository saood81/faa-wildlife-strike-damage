"""
Shared settings for every script in this project.

All paths are relative to the repository root, so the scripts run the same
way on any machine:  python src/01_prepare.py
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "faa_2016_2026.csv.gz"
PROCESSED_DIR = ROOT / "data" / "processed"
ANALYTIC_FILE = PROCESSED_DIR / "analytic.csv.gz"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"

RANDOM_STATE = 42          # one seed for every random step -> reproducible

TARGET = "INDICATED_DAMAGE"

# ---------------------------------------------------------------- temporal split
# Train on the past, tune on the next year, test on the most recent data.
TRAIN_YEARS = range(2016, 2024)   # 2016-2023
VALID_YEARS = [2024]
TEST_YEARS = [2025, 2026]

# ---------------------------------------------------------------- leakage audit
# Columns recorded AFTER the strike that describe its consequences.
# Using any of them would let the model "see the answer".
LEAKAGE_COLS = (
    ["DAMAGE_LEVEL",                                   # the label itself (used only to filter)
     "AOS", "COST_REPAIRS", "COST_OTHER",              # out-of-service time and costs
     "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ",
     "EFFECT", "EFFECT_OTHER",                         # effect on flight
     "NR_INJURIES", "NR_FATALITIES",
     "REMARKS", "COMMENTS", "OTHER_SPECIFY",           # free text written after the event
     "IMAGE", "LUPDATE", "SOURCE", "PERSON",           # report-process fields
     "REMAINS_COLLECTED", "REMAINS_SENT",              # carcass handling, post-event
     "LOCATION"]                                       # often "FOUND AT <airport>", post-event
    + [f"DAM_{p}" for p in ["RAD", "WINDSHLD", "NOSE", "ENG1", "ENG2", "ENG3",
                            "ENG4", "PROP", "WING_ROT", "FUSE", "LG", "TAIL",
                            "LGHTS", "OTHER"]]         # per-part damage flags
)

# Identifiers, duplicates of other columns, or almost-empty columns.
DROP_COLS = [
    "INDEX_NR", "REPORTED_NAME", "REPORTED_TITLE", "TRANSFER", "BIRD_BAND_NUMBER",
    "REG", "FLT", "PRECIPITATION", "ENROUTE_STATE",
    "AIRPORT", "OPERATOR", "SPECIES_ID",              # text duplicates of AIRPORT_ID / OPID / SPECIES
    "AMA", "AMO", "EMA", "EMO", "ENG_1_POS", "ENG_2_POS", "ENG_3_POS", "ENG_4_POS",
    "STATE", "RUNWAY", "OUT_OF_RANGE_SPECIES",
]

# ---------------------------------------------------------------- feature sets
# Feature set A: known (or knowable) before the strike -> prevention use.
NUMERIC_A = ["HEIGHT", "SPEED", "DISTANCE", "LATITUDE", "LONGITUDE",
             "MONTH_SIN", "MONTH_COS", "HOUR_SIN", "HOUR_COS"]
ORDINAL_A = ["AC_MASS", "NUM_ENGS", "SIZE_ORD", "NUM_STRUCK_ORD", "NUM_SEEN_ORD"]
CATEG_A = ["PHASE_OF_FLIGHT", "TIME_OF_DAY", "SKY", "TYPE_ENG", "AC_CLASS",
           "WARNED", "FAAREGION"]
HIGHCARD_A = ["SPECIES", "AIRPORT_ID", "AIRCRAFT", "OPID"]   # many categories -> target encoding

# Feature set B = A + where the animal hit the aircraft (known only after
# inspection) -> post-strike inspection triage use.
STRIKE_COLS = [f"STR_{p}" for p in ["RAD", "WINDSHLD", "NOSE", "ENG1", "ENG2", "ENG3",
                                     "ENG4", "PROP", "WING_ROT", "FUSE", "LG", "TAIL",
                                     "LGHTS", "OTHER"]]
EXTRA_B = STRIKE_COLS + ["ANY_INGESTION"]


def feature_lists(feature_set="A"):
    """Return (numeric, ordinal, categorical, high-cardinality) column lists."""
    numeric = list(NUMERIC_A)
    if feature_set == "B":
        numeric += EXTRA_B
    return numeric, list(ORDINAL_A), list(CATEG_A), list(HIGHCARD_A)
