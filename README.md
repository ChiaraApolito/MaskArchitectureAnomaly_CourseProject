# How to
Create a python environment from termina 
```
>>> py install 3.11

>>> py -3.11 -m venv .venv
```

Next, install the requirements
```
>>> .venv/Scripts/Activate
>>> pip install -r requirements.txt
```

Then, you have to [download the models](https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav) `eomt_cityscapes.bin` and `eomt_coco.bin` and save them in the folder `trained_models`

# Point 4 — Confronto EoMT-Cityscapes vs EoMT-COCO

Script: [`compare_models.py`](eomtConfrontoOfficial/compare_models.py)

---

## 1. Dataset da scaricare e dove metterli

### Immagini di validazione — `leftImg8bit`


1. Registrati su [cityscapes-dataset.com](https://www.cityscapes-dataset.com/login/)
2. Vai su **Downloads → leftImg8bit_trainvaltest.zip** (~1 GB)

Estrarre solo la cartella `val` e posizionarla dentro la cartella (da creare eventualmente) `data` presente in questa repo. Rinominare la cartella `val` con `4_val` per riconoscimento.

```
data/4_val/
├── frankfurt/
│   ├── frankfurt_000000_000294_leftImg8bit.png
│   └── ...
├── lindau/
└── munster/
```

### Annotazioni GT — `gtFine` (solo per valutazione quantitativa)

Le annotazioni semantiche devono essere scaricate dal sito ufficiale di Cityscapes:

1. Registrati su [cityscapes-dataset.com](https://www.cityscapes-dataset.com/login/)
2. Vai su **Downloads → gtFine_trainvaltest.zip** (~241 MB)
3. Estrai lo zip e copia la cartella `gtFine/` dentro `data/`, basta solo la cartella `val` :

```
data/
├── 4_val/                   
└── gtFine/                  ← estrai qui 
    └── val/
        ├── frankfurt/
        │   ├── frankfurt_000000_000294_gtFine_labelIds.png
        │   └── ...
        ├── lindau/
        └── munster/
```

> Il file che serve è `*_gtFine_labelIds.png` (ID raw delle classi, non le versioni colorate).
> La struttura `gtFine/val/<città>/` è quella standard dopo l'estrazione: non serve riorganizzare nulla.

---

## 2. Comandi
### Solo qualitativo (nessuna GT richiesta)

```bash
python -m eomtConfrontoOfficial.compare_models --n_samples 5
```

Genera 5 immagini di confronto in `comparison_outputs/`.

### Qualitativo + quantitativo

```bash
python -m eomtConfrontoOfficial.compare_models --n_samples 5 --gt_dir data/gtFine/val
```

### Tutti i parametri disponibili

| Parametro | Default | Descrizione |
|---|---|---|
| `--val_dir` | `data/4_val` | Cartella immagini Cityscapes val |
| `--cs_ckpt` | `trained_models/eomt_cityscapes.bin` | Checkpoint modello Cityscapes |
| `--coco_ckpt` | `trained_models/eomt_coco.bin` | Checkpoint modello COCO |
| `--gt_dir` | `None` | Cartella gtFine/val (abilita la metrica mIoU) |
| `--n_samples` | `5` | Numero di immagini per la parte qualitativa |
| `--out_dir` | `comparison_outputs` | Cartella di output |

---

## 3. Output e come leggerlo

### Output qualitativo — `comparison_outputs/<nome>_comparison.png`

Ogni immagine ha **3 pannelli affiancati**:

```
[ Input image ] | [ EoMT-Cityscapes (semantic) ] | [ EoMT-COCO (panoptic → CS) ]
```

- **Pannello 2** mostra la predizione semantica del modello addestrato su Cityscapes (19 classi, colori standard Cityscapes).
- **Pannello 3** mostra la predizione del modello COCO **riproiettata** nello spazio Cityscapes tramite il mapping descritto nella sezione 4. I colori sono gli stessi del pannello 2, il che permette di vedere visivamente le discrepanze.

Cosa osservare:
- Classi ben coperte dal mapping (road, sky, person, car, vegetation) → i pannelli saranno simili.
- Classi assenti nel vocabolario COCO (pole, fence, rider) → il pannello COCO avrà zone nere o colori errati.
- Il modello COCO può predire regioni *non-stradali* (es. tavolo, animali) che non hanno senso su Cityscapes.

### Output quantitativo — `comparison_outputs/miou_comparison.txt`

```
Evaluated on 500 images
Class                  CS-model (%)  COCO-model (%)
-----------------------------------------------------
road                          97.2           81.3
sidewalk                      82.1           34.7
building                      91.5           60.2
...
-----------------------------------------------------
mIoU                          75.4           38.1
```

Come leggerlo:
- **CS-model** è il riferimento: addestrato direttamente su Cityscapes, deve avere mIoU alto (~75%).
- **COCO-model** è il confronto: non addestrato su scene stradali, valori bassi sono attesi.
- Le classi con mIoU COCO > 0 sono quelle coperte dal mapping; le classi con 0% non hanno una classe COCO corrispondente.
- Il gap tra CS e COCO motiva il fine-tuning del punto 5 (il punto 5 del pdf del progetto, qui salvato come `project_guide.pdf` ).

---

## 4. Ragionamento dietro la pipeline

### Il problema: due spazi di classi incompatibili

| Modello | Task | # classi | Dataset |
|---|---|---|---|
| `eomt_cityscapes.bin` | Semantic segmentation | 19 | Cityscapes |
| `eomt_coco.bin` | Panoptic segmentation | 133 (80 things + 53 stuff) | COCO |

Confrontare i due modelli sulle stesse immagini di Cityscapes è **non-trivial** perché:
1. Il task è diverso (semantic vs panoptic).
2. Il vocabolario di classi è diverso: la maggior parte delle 133 classi COCO (pizza, giraffa, computer...) non esistono in Cityscapes.

### Scelta: valutazione semantica comune

Per rendere il confronto **fair e significativo** su un'unica metrica (mIoU a 19 classi):

1. **Modello Cityscapes** → predice direttamente 19 classi Cityscapes. Nessuna trasformazione necessaria.

2. **Modello COCO** → predice 133 classi. Le probabilità vengono **aggregate** per classe Cityscapes tramite un mapping semantico esplicito:

```
COCO wall-brick  ┐
COCO wall-stone  ├─► Cityscapes wall
COCO wall-other  │
COCO wall        ┘

COCO tree   ┐
COCO grass  ├─► Cityscapes vegetation
COCO grass2 ┘

COCO road     ──► Cityscapes road
COCO pavement ──► Cityscapes sidewalk
COCO sky-other──► Cityscapes sky
...
```

Le classi COCO senza equivalente in Cityscapes (airplane, pizza, cat...) vengono ignorate: i pixel che il modello COCO assegna a queste classi vengono marcati come `255` (ignore) nella metrica, analogamente a come Cityscapes ignora i pixel `void`.

### Perché aggregare le probabilità e non fare argmax prima?

Fare argmax prima (COCO class per pixel → remap) avrebbe ignorato i casi in cui più classi COCO con lo stesso mapping Cityscapes si *dividono* i voti. Aggregare le probabilità prima di argmax permette al modello di sommare la confidenza di classi equivalenti (es. `wall-brick + wall-concrete + wall-other`) prima di confrontarle con le altre 19 opzioni.

### Preprocessing differenziato per i due modelli

I due modelli usano img_size diversi e gestione dell'aspect ratio diversa:

- **Cityscapes** (`img_size=1024`): le immagini Cityscapes sono `1024×2048` → il modello usa **windowing** (sliding window senza overlap se possibile), taglia 2 crop da 1024×1024, raccoglie le logits e le media. Questo è il comportamento standard del codice Lightning.
- **COCO** (`img_size=640`): il modello panoptico usa **resize + padding** (fit dentro 640×640 con zero-padding). Si recupera la regione valida prima di calcolare le logits finali.

### Cosa ci dice il confronto

- Se il mIoU del modello COCO è significativamente inferiore → conferma che il training su COCO non è sufficiente per Cityscapes, motivando il fine-tuning (punto 5).
- Le classi dove COCO performa meglio (es. `person`, `car`) sono quelle più rappresentate anche in COCO.
- Le classi dove COCO performa peggio o zero (es. `rider`, `pole`, `fence`) sono assenti o rare nel vocabolario COCO.

---
---

# Mask Architecture for Road Scenes
This is the starting repository for two projects:
- Mask Architecture Anomaly Segmentation for Road Scenes  [[Project Description](https://drive.google.com/file/d/1Vz08DHsP_mojpCTAQTR6NHVq-2rEqAZM/view?usp=sharing)]
- Comprehensive Road Scene Understanding for Autonomous Driving  [[Project Description](https://drive.google.com/file/d/1tq5F_j_8O2vlGWbkU1ayPjYvCml1VEwr/view?usp=sharing)]

This repository consists of the code base for training/testing ERFNet on the Cityscapes dataset and perform anomaly segmentation. It also contains some code referring to EoMT. Some of this code may be unnecessary for your project.

## Folders
For instructions, please refer to the README in each folder:

* [eval](eval) contains tools for evaluating/visualizing an ERFNet model's output and performing anomaly segmentation.
* [trained_models](trained_models) Contains the ERFNet trained models for the baseline eval. 
* [eomt](eomt) It is almost the original folder of the EoMT project. Inside it you will find code to train and pretrained checkpoints for EoMT.

