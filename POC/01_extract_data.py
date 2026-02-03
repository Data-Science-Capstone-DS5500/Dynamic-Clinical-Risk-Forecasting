"""
Extract relevant tables from MIMIC-IV demo dataset.
Download demo from: https://physionet.org/content/mimic-iv-demo/2.2/
"""

import pandas as pd
from config import DATA_RAW, VITALS_ITEMIDS, INTERVENTION_ITEMIDS

def load_demo_tables():
    """Load core tables from downloaded demo CSVs."""
    
    # Adjust path based on where you extracted the demo
    demo_path = DATA_RAW 
    
    tables = {}
    
    # ICU module tables
    icu_path = demo_path / "icu"
    tables["icustays"] = pd.read_csv(icu_path / "icustays.csv")
    tables["chartevents"] = pd.read_csv(icu_path / "chartevents.csv")
    tables["inputevents"] = pd.read_csv(icu_path / "inputevents.csv")
    tables["d_items"] = pd.read_csv(icu_path / "d_items.csv")
    
    # Hosp module tables
    hosp_path = demo_path / "hosp"
    tables["patients"] = pd.read_csv(hosp_path / "patients.csv")
    tables["admissions"] = pd.read_csv(hosp_path / "admissions.csv")
    
    return tables


def extract_vitals(chartevents: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    """Extract vital signs for ICU stays."""
    
    # Flatten all vital itemids
    all_vital_ids = []
    for ids in VITALS_ITEMIDS.values():
        all_vital_ids.extend(ids)
    
    # Filter chartevents to vitals only
    vitals = chartevents[chartevents["itemid"].isin(all_vital_ids)].copy()
    
    # Parse timestamps
    vitals["charttime"] = pd.to_datetime(vitals["charttime"])
    
    # Map itemid to vital name
    itemid_to_name = {}
    for name, ids in VITALS_ITEMIDS.items():
        for id_ in ids:
            itemid_to_name[id_] = name
    
    vitals["vital_name"] = vitals["itemid"].map(itemid_to_name)
    
    # Keep relevant columns
    vitals = vitals[["stay_id", "charttime", "vital_name", "valuenum"]].dropna()
    
    print(f"Extracted {len(vitals):,} vital measurements")
    print(f"Unique stays: {vitals['stay_id'].nunique()}")
    print(f"Vitals distribution:\n{vitals['vital_name'].value_counts()}")
    
    return vitals


def extract_interventions(inputevents: pd.DataFrame) -> pd.DataFrame:
    """Extract vasopressor administrations."""
    
    # Flatten intervention itemids
    all_interv_ids = []
    for ids in INTERVENTION_ITEMIDS.values():
        all_interv_ids.extend(ids)
    
    # Filter to vasopressors
    vasopressors = inputevents[inputevents["itemid"].isin(all_interv_ids)].copy()
    
    # Parse timestamps
    vasopressors["starttime"] = pd.to_datetime(vasopressors["starttime"])
    vasopressors["endtime"] = pd.to_datetime(vasopressors["endtime"])
    
    # Map to drug name
    itemid_to_drug = {}
    for name, ids in INTERVENTION_ITEMIDS.items():
        for id_ in ids:
            itemid_to_drug[id_] = name
    
    vasopressors["drug"] = vasopressors["itemid"].map(itemid_to_drug)
    
    # Keep relevant columns
    vasopressors = vasopressors[[
        "stay_id", "starttime", "endtime", "drug", "rate", "rateuom"
    ]].dropna(subset=["stay_id", "starttime"])
    
    print(f"Extracted {len(vasopressors):,} vasopressor records")
    print(f"Drug distribution:\n{vasopressors['drug'].value_counts()}")
    
    return vasopressors


def main():
    print("Loading MIMIC-IV demo tables...")
    tables = load_demo_tables()
    
    print(f"\nPatients: {len(tables['patients'])}")
    print(f"ICU stays: {len(tables['icustays'])}")
    print(f"Chart events: {len(tables['chartevents']):,}")
    print(f"Input events: {len(tables['inputevents']):,}")
    
    print("\n--- Extracting Vitals ---")
    vitals = extract_vitals(tables["chartevents"], tables["icustays"])
    
    print("\n--- Extracting Interventions ---")
    interventions = extract_interventions(tables["inputevents"])
    
    # Save processed extracts
    from config import DATA_PROCESSED
    vitals.to_parquet(DATA_PROCESSED / "vitals.parquet", index=False)
    interventions.to_parquet(DATA_PROCESSED / "interventions.parquet", index=False)
    tables["icustays"].to_parquet(DATA_PROCESSED / "icustays.parquet", index=False)
    
    print(f"\nSaved to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()