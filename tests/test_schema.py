"""DDL schema contract tests to verify pipeline/schema.sql matches spec Section 12."""

import os
import re
from typing import Dict, Tuple, Set


EXPECTED_SCHEMA: Dict[str, Dict[str, Tuple[str, bool]]] = {
    "branch_master": {
        "branch_code": ("VARCHAR(20)", False),
        "branch_name": ("VARCHAR(100)", False),
        "city": ("VARCHAR(100)", False),
        "state": ("VARCHAR(100)", False),
        "region": ("VARCHAR(50)", False),
        "branch_type": ("VARCHAR(30)", False),
        "open_date": ("DATE", False),
        "closure_date": ("DATE", True),
    },
    "customer_master": {
        "customer_id": ("BIGINT", False),
        "cif_number": ("VARCHAR(20)", False),
        "first_name": ("VARCHAR(100)", False),
        "last_name": ("VARCHAR(100)", False),
        "date_of_birth": ("DATE", False),
        "gender": ("VARCHAR(20)", False),
        "marital_status": ("VARCHAR(20)", False),
        "occupation": ("VARCHAR(100)", False),
        "employment_type": ("VARCHAR(50)", False),
        "annual_income": ("NUMERIC(18,2)", False),
        "customer_since": ("DATE", False),
        "city": ("VARCHAR(100)", False),
        "state": ("VARCHAR(100)", False),
        "country": ("VARCHAR(100)", False),
        "kyc_status": ("VARCHAR(20)", False),
        "is_active": ("BOOLEAN", False),
    },
    "account_master": {
        "account_id": ("BIGINT", False),
        "customer_id": ("BIGINT", True),
        "branch_code": ("VARCHAR(20)", True),
        "account_type": ("VARCHAR(50)", False),
        "open_date": ("DATE", False),
        "account_status": ("VARCHAR(20)", False),
        "account_currency": ("VARCHAR(3)", False),
        "salary_account_flag": ("BOOLEAN", False),
        "overdraft_limit": ("NUMERIC(18,2)", False),
        "account_close_date": ("DATE", True),
    },
    "account_monthly_snapshot": {
        "account_id": ("BIGINT", False),  # Part of composite PK -> NOT NULL
        "snapshot_month": ("DATE", False),
        "current_balance": ("NUMERIC(18,2)", False),
        "average_monthly_balance": ("NUMERIC(18,2)", False),
        "min_balance_30d": ("NUMERIC(18,2)", False),
        "max_balance_30d": ("NUMERIC(18,2)", False),
        "deposit_count": ("INT", False),
        "withdrawal_count": ("INT", False),
        "debit_txn_count": ("INT", False),
        "credit_txn_count": ("INT", False),
        "fee_charged_amount": ("NUMERIC(18,2)", False),
        "salary_credit_amount": ("NUMERIC(18,2)", False),
        "account_status": ("VARCHAR(20)", False),
    },
    "card_portfolio": {
        "card_id": ("BIGINT", False),
        "customer_id": ("BIGINT", True),
        "card_type": ("VARCHAR(20)", False),
        "network": ("VARCHAR(20)", False),
        "issue_date": ("DATE", False),
        "expiry_date": ("DATE", False),
        "card_status": ("VARCHAR(20)", False),
        "primary_card_flag": ("BOOLEAN", False),
        "credit_limit": ("NUMERIC(18,2)", False),
        "rewards_program": ("VARCHAR(50)", False),
        "reward_tier": ("VARCHAR(20)", False),
    },
    "card_monthly_snapshot": {
        "card_id": ("BIGINT", False),  # PK
        "snapshot_month": ("DATE", False),
        "monthly_spend_amount": ("NUMERIC(18,2)", False),
        "monthly_txn_count": ("INT", False),
        "cash_advance_amount": ("NUMERIC(18,2)", False),
        "utilization_rate": ("NUMERIC(6,4)", False),
        "min_due_amount": ("NUMERIC(18,2)", False),
        "payment_made_amount": ("NUMERIC(18,2)", False),
        "rewards_points_balance": ("BIGINT", False),
        "card_status": ("VARCHAR(20)", False),
        "delinquency_flag": ("BOOLEAN", False),
    },
    "loan_master": {
        "loan_id": ("BIGINT", False),
        "customer_id": ("BIGINT", True),
        "branch_code": ("VARCHAR(20)", True),
        "loan_type": ("VARCHAR(50)", False),
        "sanctioned_amount": ("NUMERIC(18,2)", False),
        "disbursement_date": ("DATE", False),
        "interest_rate": ("NUMERIC(6,3)", False),
        "tenure_months": ("INT", False),
        "emi_amount": ("NUMERIC(18,2)", False),
        "loan_purpose": ("VARCHAR(100)", False),
        "origination_channel": ("VARCHAR(50)", False),
        "loan_status": ("VARCHAR(20)", False),
        "maturity_date": ("DATE", False),
    },
    "loan_monthly_snapshot": {
        "loan_id": ("BIGINT", False),  # PK
        "snapshot_month": ("DATE", False),
        "outstanding_balance": ("NUMERIC(18,2)", False),
        "emi_amount": ("NUMERIC(18,2)", False),
        "dpd_days": ("INT", False),
        "overdue_amount": ("NUMERIC(18,2)", False),
        "principal_paid_amount": ("NUMERIC(18,2)", False),
        "interest_paid_amount": ("NUMERIC(18,2)", False),
        "installment_due_amount": ("NUMERIC(18,2)", False),
        "installment_paid_amount": ("NUMERIC(18,2)", False),
        "loan_status": ("VARCHAR(20)", False),
        "restructuring_flag": ("BOOLEAN", False),
    },
    "product_holdings_monthly": {
        "customer_id": ("BIGINT", False),  # PK
        "snapshot_month": ("DATE", False),
        "savings_account_flag": ("BOOLEAN", False),
        "current_account_flag": ("BOOLEAN", False),
        "credit_card_flag": ("BOOLEAN", False),
        "personal_loan_flag": ("BOOLEAN", False),
        "home_loan_flag": ("BOOLEAN", False),
        "fixed_deposit_flag": ("BOOLEAN", False),
        "insurance_flag": ("BOOLEAN", False),
        "mutual_fund_flag": ("BOOLEAN", False),
        "demat_account_flag": ("BOOLEAN", False),
        "wealth_management_flag": ("BOOLEAN", False),
        "products_count": ("INT", False),
    },
    "transaction_fact": {
        "transaction_id": ("BIGINT", False),
        "account_id": ("BIGINT", True),
        "customer_id": ("BIGINT", True),
        "txn_timestamp": ("TIMESTAMP", False),
        "txn_date": ("DATE", False),
        "txn_month": ("DATE", False),
        "txn_type": ("VARCHAR(50)", False),
        "direction": ("VARCHAR(10)", False),
        "channel": ("VARCHAR(30)", False),
        "amount": ("NUMERIC(18,2)", False),
        "currency": ("VARCHAR(3)", False),
        "merchant_category": ("VARCHAR(100)", False),
        "merchant_name": ("VARCHAR(150)", False),
        "counterparty_type": ("VARCHAR(50)", False),
        "city": ("VARCHAR(100)", False),
        "state": ("VARCHAR(100)", False),
        "is_salary_credit": ("BOOLEAN", False),
        "is_fee": ("BOOLEAN", False),
        "is_reversal": ("BOOLEAN", False),
        "balance_after_txn": ("NUMERIC(18,2)", False),
    },
    "customer_monthly_activity": {
        "customer_id": ("BIGINT", False),  # PK
        "snapshot_month": ("DATE", False),
        "login_count": ("INT", False),
        "mobile_app_sessions": ("INT", False),
        "internet_banking_sessions": ("INT", False),
        "atm_transactions": ("INT", False),
        "branch_visits": ("INT", False),
        "debit_txn_count": ("INT", False),
        "credit_txn_count": ("INT", False),
        "total_debit_amount": ("NUMERIC(18,2)", False),
        "total_credit_amount": ("NUMERIC(18,2)", False),
        "avg_transaction_value": ("NUMERIC(18,2)", False),
        "unique_merchants": ("INT", False),
        "cash_withdrawal_count": ("INT", False),
        "card_present_txn_count": ("INT", False),
        "card_not_present_txn_count": ("INT", False),
        "days_since_last_txn": ("INT", False),
        "days_since_last_login": ("INT", False),
    },
    "digital_engagement_monthly": {
        "customer_id": ("BIGINT", False),  # PK
        "snapshot_month": ("DATE", False),
        "mobile_app_active": ("BOOLEAN", False),
        "last_login_date": ("DATE", False),
        "push_notifications_sent": ("INT", False),
        "push_notifications_opened": ("INT", False),
        "email_campaigns_sent": ("INT", False),
        "email_clicks": ("INT", False),
        "campaigns_received": ("INT", False),
        "campaigns_responded": ("INT", False),
        "web_sessions": ("INT", False),
        "notification_opt_in": ("BOOLEAN", False),
        "app_crash_count": ("INT", False),
    },
    "customer_complaints": {
        "complaint_id": ("BIGINT", False),
        "customer_id": ("BIGINT", True),
        "complaint_date": ("DATE", False),
        "complaint_month": ("DATE", False),
        "channel": ("VARCHAR(50)", False),
        "category": ("VARCHAR(100)", False),
        "severity": ("VARCHAR(20)", False),
        "resolution_days": ("INT", True),
        "resolved_flag": ("BOOLEAN", False),
        "escalated_flag": ("BOOLEAN", False),
        "csat_score": ("INT", True),
        "root_cause": ("VARCHAR(100)", False),
        "status": ("VARCHAR(20)", False),
    },
    "customer_feedback": {
        "feedback_id": ("BIGINT", False),
        "customer_id": ("BIGINT", True),
        "feedback_date": ("DATE", False),
        "feedback_month": ("DATE", False),
        "survey_channel": ("VARCHAR(50)", False),
        "survey_topic": ("VARCHAR(100)", False),
        "nps_score": ("INT", True),
        "csat_score": ("INT", True),
    },
    "churn_simulation_state": {
        "customer_id": ("BIGINT", False),
        "persona": ("VARCHAR(50)", False),
        "low_sensitivity_segment": ("BOOLEAN", False),
        "churn_month": ("DATE", True),
        "churned_flag": ("BOOLEAN", False),
        "churn_reason": ("VARCHAR(100)", True),
        "active_months_generated": ("INT", False),
    },
    "customer_churn_label": {
        "customer_id": ("BIGINT", False),  # PK
        "as_of_month": ("DATE", False),
        "prediction_horizon_months": ("INT", False),
        "churned": ("BOOLEAN", False),
        "churn_date": ("DATE", True),
        "churn_reason": ("VARCHAR(100)", True),
    },
    "churn_feature_snapshot": {
        "customer_id": ("BIGINT", False),  # PK
        "as_of_month": ("DATE", False),
        "prediction_horizon_months": ("INT", False),
        "tenure_months": ("INT", False),
        "products_count": ("INT", False),
        "balance_change_3m": ("NUMERIC(18,4)", True),
        "txn_count_change_3m": ("NUMERIC(18,4)", True),
        "login_count_change_6m": ("NUMERIC(18,4)", True),
        "complaint_count_6m": ("INT", False),
        "unresolved_complaints": ("INT", False),
        "days_since_last_login": ("INT", False),
        "salary_credit_consistency": ("NUMERIC(18,4)", True),
        "credit_utilization": ("NUMERIC(18,4)", True),
        "emi_to_income_ratio": ("NUMERIC(18,4)", True),
        "dormant_days": ("INT", False),
        "nps_avg_12m": ("NUMERIC(18,4)", True),
        "campaign_response_rate": ("NUMERIC(18,4)", True),
        "product_acquisition_velocity_6m": ("INT", False),
        "churned": ("BOOLEAN", False),
        "churn_date": ("DATE", True),
        "churn_reason": ("VARCHAR(100)", True),
    },
}


def split_table_columns(columns_def: str) -> list[str]:
    """Split column and constraint lines, ignoring commas inside parentheses."""
    parts = []
    current = []
    depth = 0
    for char in columns_def:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def parse_ddl_schema(ddl_path: str) -> Dict[str, Dict[str, Tuple[str, bool]]]:
    """Parses schema.sql directly in Python to extract exact types and nullability."""
    with open(ddl_path, "r") as f:
        content = f.read()

    # Remove SQL comments and multiple whitespace
    content_lines = []
    for line in content.splitlines():
        line_clean = re.sub(r"--.*$", "", line).strip()
        if line_clean:
            content_lines.append(line_clean)
    
    clean_ddl = " ".join(content_lines)
    
    # Split by semicolon to get statements
    statements = clean_ddl.split(";")
    
    parsed_schema = {}
    
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt:
            continue
            
        # Match CREATE TABLE statement
        match = re.match(r"CREATE\s+TABLE\s+(\w+)\s*\((.*)\)", stmt, re.IGNORECASE)
        if not match:
            continue
            
        table_name = match.group(1).lower()
        cols_def_str = match.group(2)
        
        column_definitions = split_table_columns(cols_def_str)
        table_cols = {}
        pk_cols: Set[str] = set()
        
        # Parse table constraints first
        for col_def in column_definitions:
            col_def_upper = col_def.upper()
            if col_def_upper.startswith("PRIMARY KEY"):
                # Table constraint: PRIMARY KEY (col1, col2)
                pk_match = re.search(r"PRIMARY\s+KEY\s*\((.*?)\)", col_def, re.IGNORECASE)
                if pk_match:
                    for pk_col in pk_match.group(1).split(","):
                        pk_cols.add(pk_col.strip().lower())
        
        for col_def in column_definitions:
            col_def_upper = col_def.upper()
            
            # Skip table level constraints
            tokens = col_def.split()
            if not tokens:
                continue
                
            first_token = tokens[0].upper()
            if first_token in ("PRIMARY", "FOREIGN", "CONSTRAINT", "UNIQUE") and len(tokens) > 1 and tokens[1].upper() in ("KEY", "("):
                continue
                
            col_name = tokens[0].lower()
            
            # Match data type (handles VARCHAR(20), NUMERIC(18,2) etc.)
            type_match = re.search(r"^\w+(?:\s*\(\s*\d+\s*(?:,\s*\d+\s*)?\))?", tokens[1], re.IGNORECASE)
            type_str = tokens[1]
            if type_match:
                type_str = type_match.group(0).upper().replace(" ", "")
            
            # Nullability determination
            is_nullable = True
            if "NOT NULL" in col_def_upper:
                is_nullable = False
            elif "PRIMARY KEY" in col_def_upper:
                is_nullable = False
                pk_cols.add(col_name)
            
            table_cols[col_name] = (type_str, is_nullable)
            
        # Retrofit composite primary keys to NOT NULL
        for col_name in table_cols:
            if col_name in pk_cols:
                type_str, _ = table_cols[col_name]
                table_cols[col_name] = (type_str, False)
                
        parsed_schema[table_name] = table_cols
        
    return parsed_schema


def test_schema_ddl_contract():
    """Validates that pipeline/schema.sql matches EXPECTED_SCHEMA columns, types, and nullability."""
    # Find schema.sql path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema_path = os.path.join(base_dir, "pipeline", "schema.sql")
    
    assert os.path.exists(schema_path), f"schema.sql not found at {schema_path}!"
    
    parsed = parse_ddl_schema(schema_path)
    
    # Assert all tables exist in DDL
    for table_name in EXPECTED_SCHEMA:
        assert table_name in parsed, f"Table {table_name} is missing in pipeline/schema.sql DDL!"
        
        expected_cols = EXPECTED_SCHEMA[table_name]
        parsed_cols = parsed[table_name]
        
        # Verify columns count and names
        for col_name in expected_cols:
            assert col_name in parsed_cols, f"Column '{col_name}' is missing in table '{table_name}' in schema.sql DDL!"
            
            expected_type, expected_null = expected_cols[col_name]
            parsed_type, parsed_null = parsed_cols[col_name]
            
            # Assert exact types
            assert parsed_type == expected_type, (
                f"Data type mismatch for {table_name}.{col_name}: "
                f"expected {expected_type}, got {parsed_type} in DDL!"
            )
            
            # Assert exact nullability
            assert parsed_null == expected_null, (
                f"Nullability mismatch for {table_name}.{col_name}: "
                f"expected nullable={expected_null}, got nullable={parsed_null} in DDL!"
            )
            
        # Assert no unexpected columns in parsed schema
        for col_name in parsed_cols:
            assert col_name in expected_cols, f"Unexpected column '{col_name}' in table '{table_name}' inside schema.sql DDL!"
