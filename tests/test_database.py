from pathlib import Path


def test_project_structure() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "backend" / "app" / "main.py").exists()
    assert (root / "backend" / "requirements.txt").exists()
    assert (root / "data" / "cadets").exists()
