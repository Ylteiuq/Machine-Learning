from pathlib import Path
import json

CHECKPOINT = Path(__file__).resolve().parent / "checkpoint.json"

def save_checkpoint(state: dict) -> None:
    temporary_file = CHECKPOINT.with_suffix(".tmp")
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)
    temporary_file.replace(CHECKPOINT)

def load_checkpoint() -> dict:
    with open(CHECKPOINT, "r", encoding="utf-8") as file:
        return json.load(file)
