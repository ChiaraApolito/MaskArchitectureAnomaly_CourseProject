"""Setup centralizzato per i notebook Colab del progetto.

I notebook (Task5, Task8, ...) condividono lo stesso ambiente: montare Google
Drive, clonare/aggiornare la repo, installare le dipendenze, definire i path su
Drive. Tutta quella logica vive qui, una volta sola.

Uso tipico in cima a un notebook (vedi anche la sezione "Setup su Colab" del README):

    from google.colab import drive; drive.mount("/content/drive")
    import sys; sys.path.insert(0, "/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject/notebook")
    from colab_setup import bootstrap
    P = bootstrap(task="task8")          # "task5" nell'altro notebook

`bootstrap()` e' idempotente: monta Drive (se serve), clona la repo (se manca),
installa le dipendenze UNA volta per VM (file-sentinella) e ritorna i path.
Usando sempre lo stesso server Colab, dopo il primo setup le chiamate successive
saltano tutto e sono istantanee.

Nota chicken-and-egg: questo file vive DENTRO la repo (cartella notebook/), quindi
al primissimo avvio (repo non ancora clonata) non e' importabile. Per quel caso la
repo va clonata prima -- a mano (vedi README) o con le poche righe inline che ogni
notebook ha nella prima cella.
"""
from __future__ import annotations

import os
import sys
import subprocess
import datetime
from dataclasses import dataclass
from pathlib import Path

# --- Configurazione fissa del progetto (un solo punto di verita') ----------
GIT_USERNAME = "ChiaraApolito"
REPO_NAME = "MaskArchitectureAnomaly_CourseProject"
DEFAULT_BRANCH = "finetuning/coco-to-cityscapes"
REPO_URL = f"https://github.com/{GIT_USERNAME}/{REPO_NAME}.git"

DRIVE_ROOT = Path("/content/drive/MyDrive")
PROJECT_DIR = DRIVE_ROOT / REPO_NAME
DEPS_MARKER = Path("/content/.eomt_deps_ok")   # per-VM: evita reinstalli nella stessa sessione


def log(msg: str) -> None:
    """print() con timestamp e flush immediato (output ordinato su Colab/VS Code)."""
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


@dataclass
class Paths:
    """Path principali su Drive, restituiti da bootstrap()."""
    drive_root: Path
    project: Path
    eomt: Path
    eval: Path
    large_files: Path
    weights: Path
    cityscapes_dir: Path
    anomaly_zip: Path


def _paths() -> Paths:
    large = DRIVE_ROOT / "FAIML_project_and_presentation" / "01_Project" / "large_files"
    return Paths(
        drive_root=DRIVE_ROOT,
        project=PROJECT_DIR,
        eomt=PROJECT_DIR / "eomt",
        eval=PROJECT_DIR / "eval",
        large_files=large,
        weights=large / "weights",
        cityscapes_dir=large / "datasets" / "cityscapes",
        anomaly_zip=large / "datasets" / "anomaly" / "Anomaly_Validation_Datasets.zip",
    )


def mount_drive() -> bool:
    """Monta Drive se siamo su Colab (idempotente). Ritorna True se su Colab."""
    try:
        from google.colab import drive
    except ImportError:
        log("Non su Colab: salto il mount di Drive.")
        return False
    drive.mount("/content/drive")   # se gia' montato non fa nulla
    return True


def _run(cmd, cwd=None) -> None:
    """Esegue un comando mostrando l'output reale (stderr incluso) se fallisce."""
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        log(out)
    if r.returncode != 0:
        raise RuntimeError(f"Comando fallito (exit {r.returncode}): {' '.join(cmd)}")


def clone_or_update_repo(branch: str = DEFAULT_BRANCH, update: bool = True) -> None:
    """Clona la repo in PROJECT_DIR se assente, altrimenti la riallinea al branch remoto.

    update=True replica il comportamento dei notebook attuali (git reset --hard al
    branch: scarta eventuali modifiche locali nella copia su Drive). Metti update=False
    per non toccare la copia locale.
    """
    if not (PROJECT_DIR / ".git").exists():
        log(f"Clono la repo in {PROJECT_DIR} ...")
        PROJECT_DIR.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--branch", branch, REPO_URL, str(PROJECT_DIR)])
    elif update:
        log("Aggiorno la repo al branch remoto ...")
        _run(["git", "fetch", "origin"], cwd=str(PROJECT_DIR))
        _run(["git", "checkout", branch], cwd=str(PROJECT_DIR))
        _run(["git", "reset", "--hard", f"origin/{branch}"], cwd=str(PROJECT_DIR))
    else:
        log("Repo gia' presente: nessun aggiornamento (update=False).")


# Dipendenze extra oltre a eomt/requirements.txt, per task
_EXTRA_PIP = {
    "task7": ["ood_metrics"],
    "task8": ["ood_metrics", "xlsxwriter"],
}


def install_deps(task: str | None = None, reinstall: bool = False) -> None:
    """Installa eomt/requirements.txt (+ extra del task) UNA volta per VM.

    Il file-sentinella DEPS_MARKER evita reinstalli (e anche il check lento di pip)
    nelle ri-esecuzioni della stessa sessione. reinstall=True forza comunque.
    """
    if DEPS_MARKER.exists() and not reinstall:
        log("Dipendenze gia' installate in questa VM: skip.")
        return
    req = _paths().eomt / "requirements.txt"
    log(f"Installo le dipendenze da {req} ...")
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])
    for pkg in _EXTRA_PIP.get(task or "", []):
        log(f"Installo {pkg} ...")
        _run([sys.executable, "-m", "pip", "install", "-q", pkg])
    DEPS_MARKER.touch()
    log("Dipendenze pronte.")


def wandb_login(key_env: str = "WANDB_API_KEY") -> None:
    """Imposta la WandB API key SENZA hardcodarla: env -> Colab Secrets -> getpass."""
    if os.environ.get(key_env):
        log("WANDB_API_KEY gia' impostata.")
        return
    try:
        from google.colab import userdata   # Colab Secrets (icona chiave nella sidebar)
        k = userdata.get(key_env)
        if k:
            os.environ[key_env] = k
            log("WANDB_API_KEY presa da Colab Secrets.")
            return
    except Exception:
        pass
    from getpass import getpass
    os.environ[key_env] = getpass("WandB API key: ")
    log("WANDB_API_KEY impostata." if os.environ.get(key_env) else "WANDB_API_KEY NON impostata!")


def bootstrap(branch: str = DEFAULT_BRANCH, task: str | None = None,
              update: bool = True, install: bool = True, reinstall: bool = False) -> Paths:
    """Setup completo e idempotente per un notebook. Ritorna i Paths del progetto.

    Passi: mount Drive -> clona/aggiorna repo -> aggiunge i path a sys.path ->
    installa le dipendenze (una volta per VM). task in {"task5","task8",None}.
    """
    mount_drive()
    clone_or_update_repo(branch=branch, update=update)

    P = _paths()
    for p in (P.project, P.eomt, P.eval):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    if install:
        install_deps(task=task, reinstall=reinstall)

    log("Bootstrap completato.")
    log(f"  project = {P.project}")
    log(f"  eomt    = {P.eomt}")
    log(f"  weights = {P.weights}")
    return P
