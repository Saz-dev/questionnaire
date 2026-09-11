"""
Mock world for the Week 7 race, built on the REAL bike-manual RAG app
instead of an invented domain: 10 service tickets + 1 unresolvable
"stress" ticket. search_manual (tools.py) queries the actual PDFs in
questionnaire/documents/ via the app's real vectordb.py/embedder.py/
rag.py pipeline — nothing about retrieval is mocked here, only the
service-ticket records are.

Every fact a ticket's `expected_true_spec` depends on was verified live
against the real index (rerank mode, top_k=3, filtered to the ticket's
own manual) before being written down here — see WEEK7_REPORT.md §（the
verification note) for the retrieval spot-check.
"""

BIKE_MODEL_TO_MANUAL = {
    "Honda H'ness CB350": "OMHness.pdf",
    "Honda Hornet 2.0": "OMHornet20.pdf",
    "Honda CB300R": "OMCB300R.pdf",
}

CUSTOMER_EXCESS = 50  # flat deductible on a covered (warranty) claim.

# `dependency` marks tickets where the spec to check is only discoverable
# by reading the technician notes — the reported symptom alone (misfire,
# wobble, drained battery...) never names it, so step 3 (search_manual)
# genuinely depends on what step 2 (get_service_ticket's notes) found,
# not on a field already sitting on the ticket.
TICKETS = [
    {
        "ticket_id": "TCK-01",
        "bike_model": "Honda H'ness CB350",
        "mileage_km": 8000,
        "reported_symptom": "Engine misfire, rough idle, poor fuel economy.",
        "technician_notes": (
            "Customer admits to filling up with 87-octane fuel from a "
            "roadside station for the past month — no premium fuel "
            "available nearby."
        ),
        "search_query": "What fuel octane should I use?",
        "spec_found": "91 octane or higher",
        "actual_practice": "used 87 octane",
        "coverage_status": "not_covered",
        "parts_cost": 45,
        "labor_cost": 80,
        "dependency": True,
    },
    {
        "ticket_id": "TCK-02",
        "bike_model": "Honda H'ness CB350",
        "mileage_km": 3200,
        "reported_symptom": "Hard starting, weak spark on cold mornings.",
        "technician_notes": (
            "Technician confirms the installed spark plug is the "
            "specified MR6K-9, correctly gapped and torqued per the "
            "factory procedure; no aftermarket parts found."
        ),
        "search_query": "What is the recommended spark plug for this bike?",
        "spec_found": "MR6K-9",
        "actual_practice": "MR6K-9 installed, per spec",
        "coverage_status": "covered",
        "parts_cost": 15,
        "labor_cost": 60,
        "dependency": True,
    },
    {
        "ticket_id": "TCK-03",
        "bike_model": "Honda H'ness CB350",
        "mileage_km": 15000,
        "reported_symptom": "Chain skipping under load, noisy drivetrain.",
        "technician_notes": (
            "Owner fitted an aftermarket 100-link chain after a prior "
            "repair — 4 links short of what this model calls for."
        ),
        "search_query": "How many links does the standard drive chain have?",
        "spec_found": "104 links",
        "actual_practice": "100-link aftermarket chain fitted",
        "coverage_status": "not_covered",
        "parts_cost": 120,
        "labor_cost": 90,
        "dependency": True,
    },
    {
        "ticket_id": "TCK-04",
        "bike_model": "Honda H'ness CB350",
        "mileage_km": 5000,
        "reported_symptom": "Rear wheel wobble appearing after a repair visit.",
        "technician_notes": (
            "Dealer workshop re-torqued the rear axle nut using a "
            "calibrated torque wrench per standard procedure at the "
            "last service; wobble appeared afterward with no owner "
            "intervention since."
        ),
        "search_query": "What is the torque for the rear axle nut?",
        "spec_found": "88 N·m",
        "actual_practice": "torqued to 88 N·m by dealer, per spec",
        "coverage_status": "covered",
        "parts_cost": 0,
        "labor_cost": 70,
        "dependency": True,
    },
    {
        "ticket_id": "TCK-05",
        "bike_model": "Honda H'ness CB350",
        "mileage_km": 6500,
        "reported_symptom": "Brake feels spongy, reduced stopping power.",
        "technician_notes": (
            "Owner topped up the brake fluid reservoir themselves with "
            "standard DOT 3 fluid bought locally, unaware of this "
            "model's specific requirement."
        ),
        "search_query": "Which brake fluid should I use?",
        "spec_found": "DOT 4",
        "actual_practice": "topped up with DOT 3",
        "coverage_status": "not_covered",
        "parts_cost": 25,
        "labor_cost": 55,
        "dependency": True,
    },
    {
        "ticket_id": "TCK-06",
        "bike_model": "Honda H'ness CB350",
        "mileage_km": 2000,
        "reported_symptom": "Battery drains overnight.",
        "technician_notes": (
            "Correct YTZ7 battery, installed by an authorised Honda "
            "dealer at first service, confirmed against the service "
            "invoice."
        ),
        "search_query": "What is the battery specification in the vehicle specifications table?",
        "spec_found": "YTZ7",
        "actual_practice": "YTZ7 installed, per spec",
        "coverage_status": "covered",
        "parts_cost": 90,
        "labor_cost": 30,
        "dependency": True,
    },
    {
        "ticket_id": "TCK-07",
        "bike_model": "Honda H'ness CB350",
        "mileage_km": 4000,
        "reported_symptom": "Tyres wearing unevenly, feels unstable at speed.",
        "technician_notes": (
            "Owner has been running the front tyre at 150 kPa on "
            "advice from a tyre shop, well below this model's factory "
            "spec for riding solo."
        ),
        "search_query": "What is the tyre pressure for the front when riding alone?",
        "spec_found": "200 kPa",
        "actual_practice": "ran at 150 kPa",
        "coverage_status": "not_covered",
        "parts_cost": 0,
        "labor_cost": 40,
        "dependency": True,
    },
    {
        "ticket_id": "TCK-08",
        "bike_model": "Honda H'ness CB350",
        "mileage_km": 1200,
        "reported_symptom": "Engine oil warning light, unusual knocking noise.",
        "technician_notes": (
            "Service records confirm 10W-30 SJ-grade oil used at every "
            "change, exactly per the owner's manual, changed on "
            "schedule."
        ),
        "search_query": "What is the recommended engine oil for the motorcycle?",
        "spec_found": "10W-30 or 5W-30, SJ grade",
        "actual_practice": "10W-30 SJ used, per spec",
        "coverage_status": "covered",
        "parts_cost": 180,
        "labor_cost": 150,
        "dependency": True,
    },
    {
        "ticket_id": "TCK-09",
        "bike_model": "Honda H'ness CB350",
        "mileage_km": 9000,
        "reported_symptom": "Headlight stopped working, main fuse blown twice.",
        "technician_notes": (
            "Technician replaced the main fuse with the exact "
            "factory-specified rating; failure recurred within a day "
            "with no aftermarket electrical accessories installed."
        ),
        "search_query": "What is the fuse rating in the specifications table?",
        "spec_found": "30 A",
        "actual_practice": "30 A fuse fitted, per spec",
        "coverage_status": "covered",
        "parts_cost": 60,
        "labor_cost": 90,
        "dependency": True,
    },
    {
        "ticket_id": "TCK-10",
        "bike_model": "Honda H'ness CB350",
        "mileage_km": 18000,
        "reported_symptom": "Front tyre feels unsafe, visible wear.",
        "technician_notes": (
            "Tread depth measured at 1.0 mm remaining; tyre has never "
            "been replaced since purchase despite the mileage."
        ),
        "search_query": "What is the minimum tread depth for the front tyre?",
        "spec_found": "1.5 mm minimum",
        "actual_practice": "worn to 1.0 mm, never replaced",
        "coverage_status": "not_covered",
        "parts_cost": 70,
        "labor_cost": 25,
        "dependency": True,
    },
]

# Not part of the 10-ticket race. The bike model on the ticket isn't one
# of the 3 manuals in documents/, and the notes' one alternative isn't
# either, so no sequence of correct tool calls resolves this ticket —
# used once to demonstrate budget enforcement.
STRESS_TICKET = {
    "ticket_id": "TCK-99",
    "bike_model": "Honda Rebel 500",
    "mileage_km": 10000,
    "reported_symptom": "Clutch slipping under load.",
    "technician_notes": (
        "Technician could not find this model in the dealer's manual "
        "database. A second invoice references a different model, "
        "'Honda CBR250', which also isn't on file."
    ),
    "search_query": "clutch specifications",
    "coverage_status": None,
    "parts_cost": 40,
    "labor_cost": 60,
    "dependency": False,
}


def expected_result(ticket: dict) -> dict:
    """Ground truth for grading pass/fail — independent of any tool call."""
    status = ticket["coverage_status"]
    if status == "covered":
        payout = max(ticket["parts_cost"] + ticket["labor_cost"] - CUSTOMER_EXCESS, 0)
    else:
        payout = 0
    return {"status": status, "payout": payout}
