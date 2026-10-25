# Abubakar Iqbal
# NAVTTC Week 11 - Task 02: Azure AI Security, Auth, and Responsible AI

import hashlib
import time

class AzureSecurity:
    RESPONSIBLE_AI = [
        "Fairness: Unbiased treatment across demographic groups",
        "Reliability & Safety: Consistent performance under edge cases",
        "Privacy & Security: Encryption of customer financial records",
        "Inclusiveness: Accessible to people of all abilities",
        "Transparency: Explainable predictions and audit trails",
        "Accountability: Humans retain responsibility for AI decisions"
    ]

    def __init__(self, resource="abubakar-ai-services-eastus"):
        self.resource = resource
        self.primary_key = self._generate_key("primary")
        self.secondary_key = self._generate_key("secondary")

    @staticmethod
    def _generate_key(label):
        seed = f"{label}-{time.time()}-abubakar-secret"
        return hashlib.sha256(seed.encode()).hexdigest()[:32]

    def rotate_key(self, key_type="primary"):
        if key_type == "primary":
            self.primary_key = self._generate_key("primary")
            return self.primary_key
        self.secondary_key = self._generate_key("secondary")
        return self.secondary_key

    def check_access(self, auth_token, ip_addr):
        allowed_ips = ["10.0.1.", "172.16.0."]
        if not any(ip_addr.startswith(prefix) for prefix in allowed_ips):
            return 403, f"Denied: IP {ip_addr} blocked by VNet firewall"

        valid = [self.primary_key, self.secondary_key, "Bearer EntraID-Token"]
        if auth_token in valid:
            return 200, "Authorized: Validated via Azure Managed Identity / Key"
        return 401, "Unauthorized: Invalid subscription key"

if __name__ == "__main__":
    print("NAVTTC Week 11 Task 02 - Abubakar Iqbal")

    abubakar_sec = AzureSecurity(resource="abubakar-ai-services-eastus")
    print(f"Resource: {abubakar_sec.resource}")
    print(f"Primary Key:   {abubakar_sec.primary_key}")
    print(f"Secondary Key: {abubakar_sec.secondary_key}")

    new_key = abubakar_sec.rotate_key("primary")
    print(f"Rotated Primary Key: {new_key}")

    print("\nAuthentication & Network Checks:")
    test_cases = [
        ("Bearer EntraID-Token", "10.0.1.45"),
        (abubakar_sec.primary_key, "192.168.1.10"),
        ("invalid-key-xyz", "10.0.1.50")
    ]
    for token, ip in test_cases:
        code, msg = abubakar_sec.check_access(token, ip)
        print(f"IP: {ip:<12} -> Code {code}: {msg}")

    print("\nResponsible AI 6 Pillars:")
    for pillar in AzureSecurity.RESPONSIBLE_AI:
        print(f" - {pillar}")
