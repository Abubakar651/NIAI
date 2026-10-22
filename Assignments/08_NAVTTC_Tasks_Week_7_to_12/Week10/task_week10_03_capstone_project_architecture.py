# Week 10 - Task 03: Employable Capstone Project Architecture & Specification
# NAVTTC Outline Week 10 Days 4 & 5
# Reference: Industry-Grade Employable AI Project Architecture (Fintech / Credit Intelligence)

import json
import time

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class CapstoneProjectArchitecture:
    """
    Complete Architectural Blueprint and Mock Ingestion/Scoring Engine for
    NAVTTC Capstone Employable Project:
    'Real-Time Intelligent Credit Risk Prediction & Automated Decision Engine'
    """

    SYSTEM_METADATA = {
        "project_name": "CrediSense AI - Intelligent Credit Risk Decisioning Engine",
        "author": "Abubakar Iqbal",
        "role": "Machine Learning Engineer / Solution Architect",
        "institution": "NAVTTC / NIAI (NETSOL Technologies)",
        "target_industry": "Commercial Banking, Microfinance & Retail Lending",
        "core_problem": "Mitigate bad debt write-offs by identifying high-risk borrowers prior to loan disbursement while reducing manual loan processing time."
    }

    ARCHITECTURE_COMPONENTS = [
        {
            "layer": "1. Data Ingestion & Storage",
            "technologies": ["Apache Kafka", "PostgreSQL / MySQL", "AWS S3 / Azure Blob Storage"],
            "description": "Streaming and batch ingestion of applicant financials, credit bureau history, and transaction logs."
        },
        {
            "layer": "2. Feature Store & Preprocessing",
            "technologies": ["Pandas", "Scikit-Learn", "Redis (Online Feature Cache)"],
            "description": "Real-time z-score feature scaling, median imputation, debt-to-income derivation, and categorical one-hot encoding."
        },
        {
            "layer": "3. Predictive Core & Ensembling",
            "technologies": ["Random Forest Classifier", "XGBoost", "PyTorch / MLP"],
            "description": "Ensemble model calibrated with dynamic decision thresholds (T=0.50 balanced vs T=0.45 risk-averse)."
        },
        {
            "layer": "4. Automated Decision Routing Engine",
            "technologies": ["FastAPI", "Uvicorn", "Celery Task Queue"],
            "description": "3-Tier Routing: Green Lane (Instant Approval < 25%), Yellow Lane (Manual Underwriter 25-50%), Red Lane (Decline / Collateral > 50%)."
        },
        {
            "layer": "5. Model Governance & Drift Monitoring",
            "technologies": ["Prometheus", "Grafana", "Evidently AI"],
            "description": "Continuous tracking of Population Stability Index (PSI), concept drift, and monthly retraining triggers."
        }
    ]

    @staticmethod
    def simulate_loan_decision_endpoint(applicant_payload):
        """
        Simulates the microservice decision endpoint:
        Takes applicant JSON -> returns probability, risk tier, and action.
        """
        # Feature weighting heuristic matching Random Forest importances
        chk_balance = applicant_payload.get("checking_balance", 0)
        sav_balance = applicant_payload.get("savings_balance", 0)
        loan_amt = applicant_payload.get("loan_amount", 5000)
        duration = applicant_payload.get("duration_months", 24)
        emp_years = applicant_payload.get("employment_years", 2)

        # Baseline risk
        risk_score = 0.30
        if chk_balance <= 0:
            risk_score += 0.25
        if sav_balance < 500:
            risk_score += 0.15
        if loan_amt > 10000:
            risk_score += 0.10
        if duration > 36:
            risk_score += 0.08
        if emp_years < 1:
            risk_score += 0.10

        p_default = min(max(risk_score, 0.05), 0.95)

        if p_default < 0.25:
            lane = "Green Lane"
            decision = "Instant Automated Approval"
            sla = "< 30 seconds"
        elif p_default <= 0.50:
            lane = "Yellow Lane"
            decision = "Forward to Senior Underwriter (Manual Verification)"
            sla = "24 - 48 Hours"
        else:
            lane = "Red Lane"
            decision = "High Risk: Require Co-Guarantor / 120% Collateral or Decline"
            sla = "Instant Automated Rejection"

        return {
            "applicant_id": applicant_payload.get("applicant_id"),
            "predicted_default_probability": round(p_default, 4),
            "assigned_risk_tier": lane,
            "recommended_action": decision,
            "decision_sla": sla,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }


if __name__ == "__main__":
    print("=" * 65)
    print("Week 10 Task 3: Capstone Employable AI Project Architecture")
    print("Student: Abubakar Iqbal | NAVTTC Week 10")
    print("=" * 65)

    arch = CapstoneProjectArchitecture()
    print(json.dumps(arch.SYSTEM_METADATA, indent=2))

    print("\n--- 5-Layer End-to-End Enterprise Architecture ---")
    for comp in arch.ARCHITECTURE_COMPONENTS:
        print(f"\n{comp['layer']}")
        print(f"  Tech Stack:  {', '.join(comp['technologies'])}")
        print(f"  Description: {comp['description']}")

    # Simulate 3 incoming live loan application payloads across risk tiers
    test_applicants = [
        {"applicant_id": "APP-101", "checking_balance": 2500, "savings_balance": 12000, "loan_amount": 4000, "duration_months": 12, "employment_years": 6},
        {"applicant_id": "APP-102", "checking_balance": 200, "savings_balance": 800, "loan_amount": 8000, "duration_months": 24, "employment_years": 2},
        {"applicant_id": "APP-103", "checking_balance": -150, "savings_balance": 0, "loan_amount": 18000, "duration_months": 48, "employment_years": 0.5}
    ]

    print("\n--- Real-Time REST Microservice Decision Simulation ---")
    for app in test_applicants:
        result = arch.simulate_loan_decision_endpoint(app)
        print(f"\nApplicant [{result['applicant_id']}]:")
        print(f"  Default Risk Probability: {result['predicted_default_probability'] * 100:.1f}%")
        print(f"  Operational Routing:      {result['assigned_risk_tier']}")
        print(f"  Action:                   {result['recommended_action']}")

    print("\nWeek 10 Task 3 Capstone Architecture Successfully Documented & Simulated.")
