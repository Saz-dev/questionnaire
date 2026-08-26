from rag import RAG
from tracing import make_trace, write_trace
from vectordb import build_index

QUESTIONS = [
    # --- exact-token spec lookups ---
    "What is the main fuse rating?",
    "What is the recommended spark plug for this bike?",
    "What is the minimum tread depth for the front tyre?",
    "What is the torque for the rear axle nut?",
    "What is the idle speed?",
    "What battery does the CB350 use?",
    "What is the correct drive chain slack?",
    "What is the clutch lever freeplay?",
    "What fuel octane should I use?",
    "What is the tyre pressure for the front when riding alone?",
    "How many links does the standard drive chain have?",
    "What type is the headlight?",
    "How long is the warranty period in India?",
    "What is the recommended engine oil for the motorcycle?",
    "Which brake fluid should I use?",

    # --- procedural ---
    "What is the first step to change the spark plug?",
    "How do I adjust the drive chain slack?",
    "How do I check the clutch lever freeplay?",
    "How do I check the engine oil level?",
    "What is the procedure for replacing the air filter?",
    "How do I bleed the front brakes?",
    "What is the procedure to start the bike after a long storage period?",

    # --- vague / realistic messy phrasing ---
    "my bike feels loose when I brake, what do I check",
    "engine sounds weird after long ride",
    "chain seems too tight is that bad",
    "bike won't start in the morning",
    "vibration at high speed, is that normal",
    "headlight seems dim what could be wrong",
    "clutch feels heavy lately",

    # --- cross-document: same kind of question, different bike ---
    "what oil does the hornet use",
    "what oil does the cb300r use",
    "what is the tyre pressure for the hornet",
    "what is the tyre pressure for the cb300r",
    "what is the spark plug for the hornet",
    "what is the spark plug for the cb300r",
    "what is the battery for the hornet",
    "what is the battery for the cb300r",

    # --- deliberately out of scope: the app should refuse ---
    "what is the top speed of the CB350",
    "what is the capital of France",
    "how much does the bike cost",
    "can I finance this bike through a dealership",
    "what is the resale value of a used CB300R",
    "what color options are available",
]

assert len(QUESTIONS) >= 40, "need enough variety for a real random sample"

MODE = "rerank"  # the app's current default end-to-end mode


def main():
    db = build_index()
    rag = RAG(db, mode=MODE)
    for q in QUESTIONS:
        result = rag.answer(q, mode=MODE)
        trace = make_trace(q, result, mode=MODE)
        write_trace(trace)
        print(f"logged {trace['trace_id']}  grounded={trace['grounded']}  {q!r}")


if __name__ == "__main__":
    main()
