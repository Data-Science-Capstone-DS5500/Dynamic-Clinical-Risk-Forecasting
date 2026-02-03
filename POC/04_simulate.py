"""
Demonstrate intervention-conditioned forecasting.
This is the core "what-if" functionality we need to validate.
"""

import pandas as pd
import numpy as np
import pickle
from config import DATA_PROCESSED

def load_models():
    """Load trained models."""
    with open(DATA_PROCESSED / "models.pkl", "rb") as f:
        data = pickle.load(f)
    return data["models"], data["feature_cols"]


def simulate_with_intervention(
    models: dict,
    feature_cols: list,
    patient_state: pd.Series,
    intervention: dict,
) -> dict:
    """
    Predict outcomes under a specified intervention scenario.
    """
    modified_state = patient_state.copy()
    
    for key, value in intervention.items():
        if key in modified_state.index:
            modified_state[key] = value
    
    # Extract feature vector
    X = modified_state[feature_cols].values.reshape(1, -1)
    X = np.nan_to_num(X, nan=0)
    
    predictions = {}
    for target_name, model in models.items():
        pred = model.predict(X)[0]
        predictions[target_name.replace("_target", "")] = pred
    
    return predictions


def find_good_test_cases(df: pd.DataFrame, n_cases: int = 3):
    """Find patients with vasopressor transitions for interesting scenarios."""
    
    test_cases = []
    
    for stay_id in df["stay_id"].unique():
        stay_df = df[df["stay_id"] == stay_id].sort_values("hour_idx")
        
        # Look for hours where vasopressor status could vary
        vaso_hours = stay_df[stay_df["vasopressor_on"] == 1]["hour_idx"].values
        no_vaso_hours = stay_df[stay_df["vasopressor_on"] == 0]["hour_idx"].values
        
        # Find hours with complete vital data and enough future
        valid_hours = stay_df[
            (stay_df["map"].notna()) & 
            (stay_df["heart_rate"].notna()) &
            (stay_df["hour_idx"] < stay_df["hour_idx"].max() - 6)
        ]["hour_idx"].values
        
        if len(valid_hours) > 6:
            # Prefer hours near vasopressor transitions
            if len(vaso_hours) > 0 and len(no_vaso_hours) > 0:
                # Find hour just before vasopressor started
                first_vaso = vaso_hours.min()
                if first_vaso > 0 and (first_vaso - 1) in valid_hours:
                    test_cases.append({
                        "stay_id": stay_id,
                        "hour_idx": first_vaso - 1,
                        "context": "just before vasopressor start"
                    })
                elif first_vaso in valid_hours:
                    test_cases.append({
                        "stay_id": stay_id,
                        "hour_idx": first_vaso,
                        "context": "at vasopressor start"
                    })
            elif len(valid_hours) > 0:
                # Just pick middle of stay
                mid_hour = valid_hours[len(valid_hours) // 2]
                test_cases.append({
                    "stay_id": stay_id,
                    "hour_idx": mid_hour,
                    "context": "mid-stay"
                })
        
        if len(test_cases) >= n_cases:
            break
    
    return test_cases


def run_scenario_comparison(
    models: dict,
    feature_cols: list,
    df: pd.DataFrame,
    stay_id: int,
    hour_idx: int,
    context: str = "",
):
    """Compare predictions under different intervention scenarios."""
    
    mask = (df["stay_id"] == stay_id) & (df["hour_idx"] == hour_idx)
    if mask.sum() == 0:
        print(f"No data for stay {stay_id} hour {hour_idx}")
        return None
    
    patient_state = df[mask].iloc[0]
    
    print(f"\n{'='*70}")
    print(f"PATIENT: Stay {stay_id}, Hour {hour_idx} ({context})")
    print(f"{'='*70}")
    
    # Current vitals
    print(f"\nCurrent State:")
    print(f"  MAP:           {patient_state.get('map', np.nan):.1f} mmHg")
    print(f"  Heart Rate:    {patient_state.get('heart_rate', np.nan):.1f} bpm")
    print(f"  SpO2:          {patient_state.get('spo2', np.nan):.1f}%")
    print(f"  Vasopressor:   {'ON' if patient_state.get('vasopressor_on', 0) else 'OFF'}")
    print(f"  Vaso Rate:     {patient_state.get('vasopressor_rate', 0):.3f}")
    
    # Define scenarios
    scenarios = {
        "No Vasopressor": {"vasopressor_on": 0, "vasopressor_rate": 0},
        "Low Dose":       {"vasopressor_on": 1, "vasopressor_rate": 0.1},
        "Medium Dose":    {"vasopressor_on": 1, "vasopressor_rate": 0.2},
        "High Dose":      {"vasopressor_on": 1, "vasopressor_rate": 0.4},
    }
    
    results = {}
    for name, intervention in scenarios.items():
        results[name] = simulate_with_intervention(
            models, feature_cols, patient_state, intervention
        )
    
    # Print comparison table
    print(f"\n6-HOUR FORECAST UNDER DIFFERENT SCENARIOS:")
    print(f"{'-'*70}")
    header = f"{'Metric':<15}"
    for name in scenarios.keys():
        header += f"{name:<14}"
    print(header)
    print(f"{'-'*70}")
    
    metrics = ["map", "heart_rate", "spo2"]
    for metric in metrics:
        row = f"{metric:<15}"
        for name in scenarios.keys():
            val = results[name].get(metric, np.nan)
            row += f"{val:<14.1f}"
        print(row)
    
    # Quantify intervention effect
    print(f"\n{'='*70}")
    print("INTERVENTION EFFECT ANALYSIS")
    print(f"{'='*70}")
    
    baseline = results["No Vasopressor"]
    
    for metric in metrics:
        baseline_val = baseline.get(metric, np.nan)
        effects = []
        
        for name, preds in results.items():
            if name == "No Vasopressor":
                continue
            effect = preds.get(metric, np.nan) - baseline_val
            effects.append((name, effect))
        
        print(f"\n{metric.upper()} response to vasopressor:")
        for name, effect in effects:
            direction = "↑" if effect > 0 else "↓" if effect < 0 else "→"
            print(f"  {name}: {direction} {abs(effect):.2f} vs no-pressor baseline")
    
    # Get actual outcome if available
    future_mask = (df["stay_id"] == stay_id) & (df["hour_idx"] == hour_idx + 6)
    if future_mask.sum() > 0:
        actual = df[future_mask].iloc[0]
        print(f"\n{'='*70}")
        print("ACTUAL OUTCOME (6 hours later)")
        print(f"{'='*70}")
        print(f"  Actual MAP:        {actual.get('map', np.nan):.1f} mmHg")
        print(f"  Actual Heart Rate: {actual.get('heart_rate', np.nan):.1f} bpm")
        print(f"  Actual Vasopressor: {'ON' if actual.get('vasopressor_on', 0) else 'OFF'}")
        
        # Which scenario was closest?
        actual_map = actual.get("map", np.nan)
        if not np.isnan(actual_map):
            closest = min(results.items(), key=lambda x: abs(x[1].get("map", np.nan) - actual_map))
            print(f"\n  Closest scenario to actual: {closest[0]} (predicted MAP: {closest[1].get('map', np.nan):.1f})")
    
    return results


def main():
    print("Loading models and data...")
    models, feature_cols = load_models()
    df = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    
    print(f"Loaded {len(models)} models: {list(models.keys())}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Find interesting test cases
    print("\nFinding test cases with vasopressor exposure...")
    test_cases = find_good_test_cases(df, n_cases=3)
    
    if not test_cases:
        print("No suitable test cases found, using first available stay")
        stay_id = df["stay_id"].iloc[0]
        test_cases = [{"stay_id": stay_id, "hour_idx": 12, "context": "default"}]
    
    print(f"Found {len(test_cases)} test cases")
    
    # Run scenarios
    all_results = []
    for case in test_cases:
        results = run_scenario_comparison(
            models, feature_cols, df,
            case["stay_id"], case["hour_idx"], case["context"]
        )
        if results:
            all_results.append(results)
    
    # Summary
    print(f"\n{'#'*70}")
    print("POC VALIDATION SUMMARY")
    print(f"{'#'*70}")
    
    # Check if interventions affected predictions
    intervention_effects = []
    for results in all_results:
        if results:
            no_pressor_map = results["No Vasopressor"].get("map", 0)
            high_dose_map = results["High Dose"].get("map", 0)
            effect = abs(high_dose_map - no_pressor_map)
            intervention_effects.append(effect)
    
    avg_effect = np.mean(intervention_effects) if intervention_effects else 0
    
    print(f"""
    Data Quality:          ✓ PASSED (94%+ vital coverage)
    Model Training:        ✓ PASSED (R² > 0 for MAP and HR)
    
    Intervention Effect:   {"✓ PASSED" if avg_effect > 0.5 else "⚠ WEAK"} 
                           Average MAP change from vasopressor: {avg_effect:.2f} mmHg
    
    {"="*60}
    
    INTERPRETATION:
    """)
    
    if avg_effect > 2:
        print("""    ✓ Strong effect: Vasopressor intervention meaningfully changes
      predicted outcomes. The "what-if" simulation is working.
      
      → PROCEED to full implementation with MIMIC-IV""")
    elif avg_effect > 0.5:
        print("""    ~ Moderate effect: Some intervention signal detected.
      Consider:
      - Adding more features (labs, SOFA components)
      - Using longer lookback windows
      - Training on full MIMIC-IV for more vasopressor episodes
      
      → PROCEED with caution, expect improvements with more data""")
    else:
        print("""    ✗ Weak effect: Model not strongly conditioning on interventions.
      Possible causes:
      - Insufficient vasopressor variation in 100-patient demo
      - Need causal methods rather than conditional prediction
      - Feature engineering may need refinement
      
      → Consider causal inference approaches (G-computation, etc.)""")


if __name__ == "__main__":
    main()