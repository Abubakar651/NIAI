# Week 12 - Task 04: Azure AI Container Deployment, CI/CD Pipeline & Cost Governance
# NAVTTC Outline Week 12 Days 4 & 5
# Reference: Microsoft Learn / Deploy Azure AI Services in Containers & Manage Costs

import json
import time

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class AzureDeploymentManager:
    """
    Automates the deployment, containerization, and cost governance for Azure AI solutions:
      1. Container Deployment: Docker configurations for running Cognitive Services on-premises.
      2. CI/CD Pipeline: Automated build, test, and container deployment workflow.
      3. Cost Management: Monitoring API consumption, tier thresholds, and budget alerts.
    """

    @staticmethod
    def generate_docker_compose_manifest():
        """Generates standard Docker Compose for Azure Cognitive Services prebuilt container."""
        manifest = """version: '3.8'

services:
  azure-cognitive-vision:
    image: mcr.microsoft.com/azure-cognitive-services/vision/read:3.2
    container_name: azure_vision_ocr_container
    ports:
      - "5000:5000"
    environment:
      - Eula=accept
      - Billing=https://eastus.api.cognitive.microsoft.com/
      - ApiKey=${AZURE_AI_SERVICE_KEY}
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4096M
        reservations:
          cpus: '1.0'
          memory: 2048M
    restart: unless-stopped
"""
        return manifest

    @staticmethod
    def generate_cicd_pipeline_yaml():
        """Generates GitHub Actions / Azure DevOps CI/CD workflow definition."""
        workflow = """name: Deploy Azure AI Cognitive Solution

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Dependencies
        run: pip install -r requirements.txt

      - name: Run Model Unit & Integrity Tests
        run: python -m pytest tests/

      - name: Log in to Azure Container Registry (ACR)
        uses: azure/docker-login@v1
        with:
          login-server: ${{ secrets.ACR_LOGIN_SERVER }}
          username: ${{ secrets.ACR_USERNAME }}
          password: ${{ secrets.ACR_PASSWORD }}

      - name: Deploy Container to Azure Kubernetes Service (AKS)
        run: |
          az aks get-credentials --resource-group ai-rg --name credi-cluster
          kubectl apply -f k8s/deployment.yaml
"""
        return workflow

    @staticmethod
    def monitor_cost_and_quotas(current_calls, monthly_budget_usd=250.0):
        """Simulates Azure Cost Management & Consumption alerts."""
        cost_per_1000_calls = 1.50 # Standard S0 Tier pricing
        incurred_cost = (current_calls / 1000.0) * cost_per_1000_calls
        burn_rate_pct = (incurred_cost / monthly_budget_usd) * 100

        status = "NORMAL"
        action = "No action required."

        if burn_rate_pct >= 90:
            status = "CRITICAL ALERT"
            action = "Throttle automated endpoints, restrict non-critical batch jobs."
        elif burn_rate_pct >= 75:
            status = "WARNING"
            action = "Send email alert to Cloud FinOps administrator."

        return {
            "monitored_service": "Azure AI Services (S0 Multi-Service)",
            "api_calls_processed": current_calls,
            "incurred_cost_usd": round(incurred_cost, 2),
            "monthly_budget_usd": monthly_budget_usd,
            "budget_consumed_pct": f"{burn_rate_pct:.1f}%",
            "finops_status": status,
            "recommended_action": action
        }


if __name__ == "__main__":
    print("=" * 65)
    print("Week 12 Task 4: Azure Containerization, CI/CD & Cost Management")
    print("Student: Abubakar Iqbal | NAVTTC Week 12")
    print("=" * 65)

    mgr = AzureDeploymentManager()

    # 1. Container Deployment
    print("\n[1] Prebuilt Azure Cognitive Services Docker Compose Manifest:")
    docker_config = mgr.generate_docker_compose_manifest()
    print(docker_config)

    # 2. CI/CD Deployment Workflow
    print("\n[2] Enterprise CI/CD Pipeline Configuration (GitHub Actions):")
    cicd_config = mgr.generate_cicd_pipeline_yaml()
    print(cicd_config)

    # 3. Cost Governance & Monitoring
    print("\n[3] Azure Cost Management & Quota Telemetry Inspection:")
    telemetry = mgr.monitor_cost_and_quotas(current_calls=145000, monthly_budget_usd=250.0)
    for k, v in telemetry.items():
        print(f"  {k:<25}: {v}")

    print("\nWeek 12 Task 4 Container Deployment & FinOps Governance Complete.")
