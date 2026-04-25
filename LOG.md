# Project Log — MaskArchitectureAnomaly Course Project

Traccia cronologica delle modifiche al progetto. Aggiorna questo file ad ogni sessione.

---

## Sessione 1 — Setup iniziale (da git log)

**Commit `0669638`** — Aggiunto `CLAUDE.md`
- File di istruzioni per Claude Code con regole di progetto, architettura e comandi.

**Commit `c3958d7`** — Modificato `.gitignore`

**Commit `b1811d4`** — `compare_emot.py` creato su Colab
- Prima versione sperimentale dello script di confronto (poi eliminata).

**Commit `400538d`** — Eliminato il file `compare_emot.py`

**Commit `4496e48`** — Creato `prova.py` (test / sandbox)

**Commit `6b9cc32`** — Aggiunta cartella `eomtConfrontoOfficial/`
- Contiene una versione del compare_models pensata per essere importata come modulo
  (`python -m eomtConfrontoOfficial.compare_models`).
- Usa `sys.path` relativo al `parent.parent` per trovare `eomt/`.

---

## Sessione 2 — Punto 4: pipeline di confronto (con Claude)

**Commit `5958a64`** — Creato `compare_models.py` (prima versione Claude)
- Script standalone nella root del progetto.
- Implementa confronto qualitativo e quantitativo tra i due modelli EoMT.

**Commit `89ceb50`** — Fix `img_paths` per l'analisi quantitativa
- La parte qualitativa ora usa lo stesso numero di sample (`args.n_samples`)
  della parte quantitativa, invece di scorrere tutte le immagini.

---

## Sessione 3 — Raffinamento pipeline (sessione corrente)

### File modificati

**`compare_models.py`** — Riscritto completamente (root del progetto)

Motivazione: la versione precedente era stata eliminata dal working tree (`git status: D compare_models.py`). Riscritta con le seguenti differenze rispetto alla versione precedente:

| Aspetto | Versione precedente | Versione attuale |
|---|---|---|
| Caricamento modello | non specificato | strip del prefisso `network.*` dal checkpoint Lightning |
| Download backbone | timm scaricava DINOv2 da internet | `ckpt_path="skip_timm_download"` evita il download |
| Inference COCO | argmax → remap | aggregazione probabilità → argmax (più corretto) |
| Preprocessing COCO | non specificato | resize + zero-padding (come Lightning panoptic) |
| Preprocessing CS | non specificato | windowed inference con overlap (come Lightning semantic) |
| Mapping COCO→CS | parziale | mapping completo documentato (things + stuff) |
| Output quantitativo | print a schermo | salva anche `miou_comparison.txt` |

**Chiave di design — mapping COCO (133 cls) → Cityscapes (19 cls):**
- Le probabilità per classi COCO semanticamente equivalenti vengono **sommate** prima dell'argmax.
  Esempio: `wall-brick + wall-concrete + wall-other + wall` → `Cityscapes:wall`.
- Classi COCO senza corrispondente (airplane, pizza, cat...) → pixel marcati come `ignore=255`.
- Questo è più corretto di fare argmax prima, perché sfrutta tutta la distribuzione di probabilità
  invece di perdere confidenza distribuita su più classi equivalenti.

**`README.md`** — Sezione "Point 4" riscritta

Aggiunta documentazione completa con:
- Istruzioni download `gtFine_trainvaltest.zip` da cityscapes-dataset.com
- Struttura attesa delle cartelle `data/`
- Tabella parametri CLI
- Guida alla lettura dell'output
- Spiegazione del ragionamento della pipeline

---

## Stato attuale del working tree

```
compare_models.py          ← script principale (riscritto questa sessione, non committato)
eomtConfrontoOfficial/
└── compare_models.py      ← versione alternativa con import a modulo (non committata)
README.md                  ← aggiornato questa sessione (non committato)
LOG.md                     ← questo file (nuovo)
trained_models/
├── eomt_cityscapes.bin    ← modello EoMT semantic su Cityscapes (365 MB)
├── eomt_coco.bin          ← modello EoMT panoptic su COCO (358 MB)
├── erfnet_pretrained.pth
└── erfnet_encoder_pretrained.pth.tar
data/
├── 4_val/                 ← 500 immagini Cityscapes val (solo leftImg8bit)
└── Validation_Dataset/    ← dataset anomaly detection (FS, RoadAnomaly, RoadObstacle)
```

---

## Prossimi step (dal project_guide.pdf)

- [ ] **Punto 4 — Quantitativo**: scaricare `gtFine_trainvaltest.zip` da cityscapes-dataset.com
      e lanciare `python compare_models.py --gt_dir data/gtFine/val`
- [ ] **Punto 5**: fine-tune del modello COCO su Cityscapes (semantic segmentation)
      - Iniziare congelando tutto tranne la prediction head
      - Usare AMP per ridurre i tempi su Colab
      - Valutare con la stessa pipeline del punto 4
- [ ] **Punti 6-8**: anomaly segmentation con post-hoc methods (MSP, MaxLogit, MaxEntropy, RbA)
      su ERFNet ed EoMT, sui dataset in `data/Validation_Dataset/`
