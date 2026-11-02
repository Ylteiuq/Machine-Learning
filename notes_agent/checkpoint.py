from pathlib import Path
import json

CHECKPOINT = Path(__file__).resolve().parent / "checkpoint.json"

def save_checkpoint(state: dict) -> None:
    with open(CHECKPOINT, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)        

def load_checkpoint() -> dict:
    with open(CHECKPOINT, "r", encoding="utf-8") as file:
        return json.load(file)