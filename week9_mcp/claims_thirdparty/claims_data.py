"""Claims data for the claims-system MCP server.

The claims platform team's claims database. Adjuster notes are free-text
entries written by the adjuster handling each claim.
"""

CLAIM_NUMBER_RE_HINT = "claim numbers look like CLM-YYYY-nnnnn (e.g. CLM-2024-88120)"

CLAIMS = {
    "CLM-2024-88120": {
        "claimant": "A. Marchetti",
        "policy_number": "POL-2024-88120",
        "status": "in_review",
        "reported": "2024-11-02",
        "claimed_amount": 25000,
        "adjuster_notes": [
            "2024-11-03: Initial report - burst pipe in laundry room caused flooding.",
            "2024-11-05: On-site inspection. Water damage to flooring and drywall, mostly the ground floor.",
            "2024-11-08: Estimate received. Total repair $25,000, all water-related.",
            "2024-11-10: Flagged for policy limit check - is this covered under Endorsement W cap?",
        ],
    },
    "CLM-2024-88114": {
        "claimant": "K. Obona",
        "policy_number": "POL-2024-88114",
        "status": "approved",
        "reported": "2024-10-19",
        "claimed_amount": 3400,
        "adjuster_notes": [
            "2024-10-20: Chimney fire, small. Covered under fire and lightning.",
            "2024-10-25: Repair completed; paid in full.",
        ],
    },
    "CLM-2024-88123": {
        "claimant": "D. Rose",
        "policy_number": "POL-2024-88123",
        "status": "denied",
        "reported": "2024-11-12",
        "claimed_amount": 8000,
        "adjuster_notes": [
            "2024-11-13: Cellar flooded from rising groundwater.",
            "2024-11-15: Renter contents policy excludes flooding - denial drafted.",
            "2024-11-16: Denial notice issued.",
        ],
    },
}