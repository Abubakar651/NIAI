# Week 11 - Task 02: Azure AI Security, Authentication & Responsible AI
# NAVTTC Outline Week 11 Day 2
# Reference: Microsoft Learn / Secure and Monitor Microsoft Azure AI Services

import hashlib
import time

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class AzureSecurityAndGovernance:
    """
    Implements enterprise security patterns for Azure AI Services:
      1. Authentication: API Key Rotation vs Azure AD (Microsoft Entra ID) Managed Identity
      2. Network Isolation: Virtual Network (VNet) Service Endpoints & Private Links
      3. Encryption: Customer-Managed Keys (CMK) with Azure Key Vault
      4. Responsible AI: 6 Pillars validation framework
    """

    RESPONSIBLE_AI_PILLARS = {
        "Fairness": "AI systems must treat all demographic groups without bias or discrimination.",
        "Reliability & Safety": "AI systems must operate within defined parameters and withstand adversarial inputs.",
        "Privacy & Security": "Customer financial records and PII must be protected against exfiltration.",
        "Inclusiveness": "AI applications must empower and engage people of all physical capabilities.",
        "Transparency": "AI models must have explainable decision drivers (e.g. Feature Importance).",
        "Accountability": "Human operators and designated compliance officers remain responsible for AI outputs."
    }

    def __init__(self, resource_name="credi-sense-ai-eastus"):
        self.resource_name = resource_name
        self.primary_key = self._generate_key(seed="primary")
        self.secondary_key = self._generate_key(seed="secondary")
        self.managed_identity_enabled = True
        self.vnet_restricted = True

    @staticmethod
    def _generate_key(seed=""):
        raw = f"{seed}-{time.time()}-azure-ai-secret"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def rotate_keys(self, target="primary"):
        """Simulates zero-downtime key rotation via Azure Resource Manager (ARM)."""
        if target == "primary":
            self.primary_key = self._generate_key(seed="new_primary")
            print(f"[SECURITY] Rotated Primary Key for '{self.resource_name}'. Secondary key maintains uptime.")
        else:
            self.secondary_key = self._generate_key(seed="new_secondary")
            print(f"[SECURITY] Rotated Secondary Key for '{self.resource_name}'.")

    def validate_request(self, auth_header, client_ip):
        """Simulates Azure API Gateway token and firewall evaluation."""
        # Simulated subnet firewall allowlist
        allowed_subnets = ["10.0.1.", "172.16.0."]
        ip_authorized = any(client_ip.startswith(subnet) for subnet in allowed_subnets)

        if not ip_authorized:
            return {"status": 403, "message": f"Forbidden: IP {client_ip} rejected by Azure Virtual Network (VNet) firewall."}

        # Check Bearer (Managed Identity) or Subscription-Key
        if auth_header.startswith("Bearer EntraID-Token") or auth_header == self.primary_key or auth_header == self.secondary_key:
            return {"status": 200, "message": "Authorized: Request authenticated via Azure Identity / Validated Key."}
        else:
            return {"status": 401, "message": "Unauthorized: Invalid Azure Subscription-Key or expired Entra ID token."}


if __name__ == "__main__":
    print("=" * 65)
    print("Week 11 Task 2: Azure AI Security Architecture & Governance")
    print("Student: Abubakar Iqbal | NAVTTC Week 11")
    print("=" * 65)

    sec = AzureSecurityAndGovernance(resource_name="ai-credit-risk-service")

    print(f"\n[1] Azure AI Resource: {sec.resource_name}")
    print(f"  Primary Key:   {sec.primary_key[:8]}...{sec.primary_key[-4:]}")
    print(f"  Secondary Key: {sec.secondary_key[:8]}...{sec.secondary_key[-4:]}")
    print(f"  Entra ID Managed Identity: {'Enabled (Role: Cognitive Services User)' if sec.managed_identity_enabled else 'Disabled'}")
    print(f"  VNet Private Endpoint:     {'Enforced' if sec.vnet_restricted else 'Public'}")

    # 1. Test Key Rotation
    print("\n[2] Executing Automated Zero-Downtime Key Rotation:")
    sec.rotate_keys("primary")
    print(f"  New Primary Key: {sec.primary_key[:8]}...{sec.primary_key[-4:]}")

    # 2. Test Firewall & Auth Gateway
    print("\n[3] Testing API Gateway Security Invocations:")
    # Valid intranet request with Entra ID
    res1 = sec.validate_request(auth_header="Bearer EntraID-Token-Admin", client_ip="10.0.1.45")
    print(f"  Test 1 (VNet IP + Managed Identity): Code {res1['status']} -> {res1['message']}")

    # Untrusted public IP
    res2 = sec.validate_request(auth_header=sec.primary_key, client_ip="198.51.100.22")
    print(f"  Test 2 (Public Internet IP):         Code {res2['status']} -> {res2['message']}")

    # Invalid credential
    res3 = sec.validate_request(auth_header="FakeKey123", client_ip="10.0.1.80")
    print(f"  Test 3 (Wrong Key on Valid VNet):    Code {res3['status']} -> {res3['message']}")

    # 3. Responsible AI Principles
    print("\n[4] Microsoft Responsible AI 6 Core Principles:")
    for pillar, desc in sec.RESPONSIBLE_AI_PILLARS.items():
        print(f"  * {pillar:<22} : {desc}")

    print("\nWeek 11 Task 2 Security & Responsible AI Framework Completed.")
