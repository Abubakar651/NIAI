# Abubakar Iqbal
# NAVTTC Week 12 - Task 04: Azure AI Containers, CI/CD, and Cost Governance

class AzureDeploymentManager:
    @staticmethod
    def get_docker_compose():
        return """version: '3.8'
services:
  abubakar-vision-ocr:
    image: mcr.microsoft.com/azure-cognitive-services/vision/read:3.2
    container_name: abubakar_ocr_service
    ports:
      - "5000:5000"
    environment:
      - Eula=accept
      - Billing=https://eastus.api.cognitive.microsoft.com/
      - ApiKey=${AZURE_AI_KEY}
    restart: unless-stopped
"""

    @staticmethod
    def get_cicd_pipeline():
        return """name: Build & Deploy Azure AI Solution
on:
  push:
    branches: [ main ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Azure CLI Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      - name: Deploy to Azure Container Instances
        run: az container create --resource-group abubakar-ai-rg --file docker-compose.yml
"""

    @staticmethod
    def check_finops_budget(calls, budget_usd=250.0):
        rate_per_k = 1.50
        incurred = (calls / 1000.0) * rate_per_k
        consumed_pct = (incurred / budget_usd) * 100

        if consumed_pct >= 90:
            status = "CRITICAL: Approaching budget ceiling"
        elif consumed_pct >= 75:
            status = "WARNING: 75% budget threshold reached"
        else:
            status = "NORMAL: Spend within operational targets"

        return {
            "calls": calls,
            "cost_usd": round(incurred, 2),
            "budget_usd": budget_usd,
            "consumed": f"{consumed_pct:.1f}%",
            "status": status
        }

if __name__ == "__main__":
    print("NAVTTC Week 12 Task 04 - Abubakar Iqbal")

    abubakar_mgr = AzureDeploymentManager()

    print("\nDocker Compose Manifest:")
    print(abubakar_mgr.get_docker_compose().strip())

    print("\nCI/CD Pipeline Workflow:")
    print(abubakar_mgr.get_cicd_pipeline().strip())

    print("\nCost Governance & Monitoring Telemetry:")
    abubakar_telemetry = abubakar_mgr.check_finops_budget(calls=130000, budget_usd=250.0)
    for k, v in abubakar_telemetry.items():
        print(f" {k:<12}: {v}")
