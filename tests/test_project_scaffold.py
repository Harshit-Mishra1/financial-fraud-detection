from pathlib import Path


def test_core_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").exists()
    assert (root / "requirements-train.txt").exists()
    assert (root / "requirements-serve.txt").exists()
    assert (root / "docker-compose.yml").exists()
    assert (root / "Dockerfile.api").exists()
    assert (root / ".gitignore").exists()
