#!/usr/bin/env python3
"""
Charter-AI — Maritime Data Validation CLI Script.

Validates datasets against business rules, physical constraints, and schemas:
1. Missing required fields
2. Invalid units
3. Negative values
4. Impossible vessel dimensions
5. Invalid coordinates
6. Duplicate records
7. Future historical dates
8. Inconsistent vessel classes

Usage:
    python scripts/validate_data.py --dir data/demo
    python scripts/validate_data.py --dir data/raw
    python scripts/validate_data.py --file data/demo/ports.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure workspace root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders.csv_loader import CSVLoader
from src.data.validators.business_validator import BusinessRuleValidator
from src.data.validators.schema_validator import SchemaValidator
from src.utils.logging import get_logger

logger = get_logger("data_validator")


def validate_single_file(
    file_path: Path,
    business_validator: BusinessRuleValidator,
    schema_validator: SchemaValidator,
) -> Dict[str, Any]:
    """Validate a single CSV file against both business rules and schemas."""
    dataset_name = file_path.stem
    loader = CSVLoader()

    try:
        df = loader.load(file_path)
    except Exception as e:
        return {
            "file": file_path.name,
            "dataset": dataset_name,
            "rows": 0,
            "status": "LOAD_ERROR",
            "errors": [f"Could not load file: {e}"],
        }

    # 1. Business Rule Validation
    biz_result = business_validator.validate_dataset(df, dataset_name)

    # 2. Schema Validation (if schema exists)
    schema_errors = []
    if schema_validator.get_schema_for_dataset(dataset_name):
        schema_res = schema_validator.validate_dataframe(df, dataset_name)
        if not schema_res.is_valid:
            for err in schema_res.errors[:10]:
                schema_errors.append(f"Row {err['row']} [{err['field']}]: {err['error']}")

    all_errors = biz_result["issues"] + schema_errors
    is_valid = len(all_errors) == 0

    return {
        "file": file_path.name,
        "dataset": dataset_name,
        "rows": len(df),
        "status": "PASS" if is_valid else "FAIL",
        "errors": all_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="CharterAI Maritime Data Validator")
    parser.add_argument("--dir", type=str, default="data/demo", help="Directory containing CSV datasets to validate")
    parser.add_argument("--file", type=str, default=None, help="Specific CSV file to validate")
    args = parser.parse_args()

    business_validator = BusinessRuleValidator()
    schema_validator = SchemaValidator()

    files_to_validate = []
    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"Error: File {p} does not exist.")
            sys.exit(1)
        files_to_validate.append(p)
    else:
        target_dir = Path(args.dir)
        if not target_dir.exists() or not target_dir.is_dir():
            print(f"Error: Directory {target_dir} does not exist.")
            sys.exit(1)
        files_to_validate = sorted(target_dir.glob("*.csv"))

    if not files_to_validate:
        print(f"No CSV files found to validate in {args.dir}")
        sys.exit(0)

    print("=" * 80)
    print(f" CharterAI V2 Maritime Data Quality & Integrity Validation")
    print(f" Target: {args.file or args.dir} ({len(files_to_validate)} datasets)")
    print("=" * 80)

    overall_pass = True
    total_records = 0

    for f in files_to_validate:
        res = validate_single_file(f, business_validator, schema_validator)
        total_records += res["rows"]

        status_tag = "[PASS]" if res["status"] == "PASS" else "[FAIL]"
        print(f"\n{status_tag:<8} {res['file']:<25} ({res['rows']} records)")

        if res["errors"]:
            overall_pass = False
            for err in res["errors"]:
                print(f"         - {err}")

    print("\n" + "=" * 80)
    print(f" Summary: {'PASSED - All datasets valid' if overall_pass else 'FAILED - Issues detected'}")
    print(f" Total datasets checked: {len(files_to_validate)}")
    print(f" Total records checked:  {total_records}")
    print("=" * 80)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
