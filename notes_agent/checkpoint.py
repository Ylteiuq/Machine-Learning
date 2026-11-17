from pathlib import Path
import json

CHECKPOINT = Path(__file__).resolve().parent / "checkpoint.json"

def save_checkpoint(state: dict, path: Path = None) -> None:
    path = Path(path) if path is not None else CHECKPOINT
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = path.with_suffix(".tmp")
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)
    temporary_file.replace(path)

def load_checkpoint(path: Path = None) -> dict:
    path = Path(path) if path is not None else CHECKPOINT
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
