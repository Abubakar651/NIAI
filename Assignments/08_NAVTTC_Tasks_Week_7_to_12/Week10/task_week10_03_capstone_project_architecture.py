# Abubakar Iqbal
# NAVTTC Week 10 - Task 03: Capstone Employable Project Architecture

import json
import time

class CapstoneArchitecture:
    METADATA = {
        "project": "CrediSense AI - Real-Time Credit Risk & Lending Engine",
        "author": "Abubakar Iqbal",
        "role": "AI Engineer / Trainee",
        "institution": "NAVTTC / NIAI",
        "target": "Digital Banking & Commercial Lending"
    }

    PIPELINE_LAYERS = [
        {"layer": "Data Ingestion", "stack": "Kafka, PostgreSQL", "desc": "Streaming & batch financial records"},
        {"layer": "Feature Processing", "stack": "Pandas, Scikit-Learn", "desc": "Scaling, imputation, DTI calculation"},
        {"layer": "Predictive Core", "stack": "Random Forest, MLP", "desc": "Ensemble default risk probability scoring"},
        {"layer": "Decision Routing", "stack": "FastAPI, Celery", "desc": "Green (Instant), Yellow (Manual), Red (Decline)"},
        {"layer": "Monitoring", "stack": "Prometheus, Evidently", "desc": "Data drift and accuracy metrics"}
    ]

    @staticmethod
    def evaluate_applicant(payload):
        chk = payload.get("checking", 0)
        sav = payload.get("savings", 0)
        loan = payload.get("loan_amt", 5000)
        dur = payload.get("duration", 24)
        emp = payload.get("emp_years", 2)

        risk = 0.30
        if chk <= 0:
            risk += 0.25
        if sav < 1000:
            risk += 0.15
        if loan > 10000:
            risk += 0.10
        if dur > 36:
            risk += 0.08
        if emp < 1:
            risk += 0.10

        p = min(max(risk, 0.05), 0.95)
        if p < 0.25:
            lane = "Green Lane (Instant Approval)"
        elif p <= 0.50:
            lane = "Yellow Lane (Manual Verification)"
        else:
            lane = "Red Lane (Collateral / Rejection)"

        return {
            "applicant": payload.get("name"),
            "risk_probability": round(p, 4),
            "decision": lane,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

if __name__ == "__main__":
    print("NAVTTC Week 10 Task 03 - Abubakar Iqbal")
    print(json.dumps(CapstoneArchitecture.METADATA, indent=2))

    print("\nArchitecture Layers:")
    for layer in CapstoneArchitecture.PIPELINE_LAYERS:
        print(f" - {layer['layer']}: {layer['stack']} ({layer['desc']})")

    abubakar_applicants = [
        {"name": "Abubakar Iqbal", "checking": 3500, "savings": 18000, "loan_amt": 5000, "duration": 12, "emp_years": 4},
        {"name": "Applicant B", "checking": 200, "savings": 600, "loan_amt": 8000, "duration": 24, "emp_years": 2},
        {"name": "Applicant C", "checking": -100, "savings": 0, "loan_amt": 16000, "duration": 48, "emp_years": 0.5}
    ]

    print("\nDecision Engine Simulation:")
    for app in abubakar_applicants:
        res = CapstoneArchitecture.evaluate_applicant(app)
        print(f"Applicant: {res['applicant']} -> Risk: {res['risk_probability']*100:.1f}% -> Decision: {res['decision']}")
