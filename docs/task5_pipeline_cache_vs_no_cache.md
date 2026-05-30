# Task 5 — Fine-tuning EoMT da COCO a Cityscapes: la pipeline vecchia e quella nuova

Questo documento spiega **come funzionava prima** il fine-tuning del Task 5, **come
funziona adesso** e **perché** abbiamo cambiato approccio. I termini tecnici sono usati
ma spiegati man mano; in fondo c'è un glossario.

---

## 0. Cosa stiamo cercando di fare (il contesto)

Abbiamo un modello, **EoMT**, già addestrato sul dataset **COCO**. Vogliamo riadattarlo
("fine-tuning") al dataset **Cityscapes**, che sono foto di scene stradali con 19 classi
semantiche (strada, auto, pedone, edificio…). L'obiettivo finale è la **segmentazione
semantica**: assegnare a ogni pixel dell'immagine la sua classe.

Il modello è fatto, semplificando, di due parti:

- **Encoder / backbone** (una rete DINOv2 ViT): la parte grande e costosa. Legge
  l'immagine e la trasforma in una sequenza di vettori numerici chiamati **token**
  (descrittori di porzioni d'immagine). È il ~90% dei parametri.
- **Testa di predizione** ("head"): la parte piccola e veloce (query, `mask_head`,
  `class_head`, scale-block) che dai token produce le maschere e le classi.

La guida del progetto (Task 5) suggerisce:
> *"A good first experiment is to fine-tune just the prediction head. Then you can
> gradually unfreeze the last layers and compare."*

Cioè: prima allena solo la testa (encoder **congelato**), poi scongela qualche strato
finale dell'encoder. È esattamente la pipeline nuova. Quella vecchia provava una
scorciatoia che si è rivelata sbagliata.

---

## 1. La pipeline VECCHIA: il "token cache"

### L'idea (perché sembrava sensata)

Se durante lo Stage 1 l'encoder è **congelato** (i suoi pesi non cambiano), allora a
parità di immagine produce **sempre gli stessi token**. Sembra quindi uno spreco
ricalcolare ogni epoca il forward dell'encoder — che è la parte costosa — quando il
risultato non cambia mai.

L'ottimizzazione era: fare **una volta sola** il forward dell'encoder su tutte le
immagini di training, salvare i token su disco (la **cache**), e poi durante il training
leggere i token già pronti invece di ricalcolarli. In teoria: training molto più veloce,
perché si esegue solo la testa (piccola).

Nel codice questo viveva in [eomt/training/head_cache_utils.py](../eomt/training/head_cache_utils.py),
in due varianti:

- `precompute_backbone_token_cache` + `CachedBackboneTokenWithTargetsDataset`: salva i
  **token dell'encoder** (`batch_*.pt`) per allenare anche gli ultimi strati.
- `precompute_head_cache` + `CachedHeadDataset`: una cache ancora più leggera che salva
  solo l'input della `class_head` e i target di classe già matchati (`sample_*.pt`).

### Il bug che l'ha resa inutilizzabile

Il problema sta in un dettaglio della pipeline dati: la **data augmentation**.

Durante il training, ogni immagine non viene usata "così com'è": a ogni epoca subisce
trasformazioni **casuali** — ribaltamento orizzontale (*flip*), ridimensionamento
(*scale*), ritaglio (*crop*). Questo si chiama **augmentation** e serve a far vedere al
modello tante varianti della stessa scena, così generalizza meglio e non impara a memoria.

Ecco la catena dell'errore (vedi `CachedBackboneTokenWithTargetsDataset.__getitem__`,
[head_cache_utils.py:331](../eomt/training/head_cache_utils.py#L331)):

1. La cache dei **token** veniva calcolata su **una specifica vista aumentata**
   dell'immagine — diciamo: ribaltata e ritagliata in un certo modo.
2. Durante il training, però, le **maschere di verità** (i target: "questi pixel sono
   auto, questi strada"…) venivano rilette **al volo dal dataset originale**, che
   applica una **nuova augmentation casuale** — un flip/scale/crop **diverso**.
3. Risultato: i **token dicono una cosa** (la scena ribaltata in un modo) e le
   **maschere ne dicono un'altra** (la scena ribaltata in un altro modo). Sono
   **spazialmente disallineati**.

La *loss* (la funzione che misura l'errore e guida l'addestramento) confrontava quindi
predizioni e target che **non si riferiscono alla stessa immagine**. In particolare la
**mask loss** e la **dice loss** — quelle che misurano quanto bene le maschere predette
si sovrappongono a quelle vere — venivano ottimizzate verso un bersaglio sbagliato.

**Conseguenza pratica osservata**: la metrica **mIoU** (mean Intersection-over-Union, la
misura standard di qualità della segmentazione: quanto si sovrappongono in media le
regioni predette con quelle vere, 0–100) saliva nelle primissime iterazioni e poi
**collassava intorno a ~14**, un valore pessimo. Il modello stava letteralmente
imparando a inseguire un bersaglio incoerente.

### Un secondo danno: niente augmentation

Anche ignorando il disallineamento, la cache ha un difetto strutturale: i token vengono
congelati su **una sola vista** per immagine. Quindi l'effetto benefico
dell'augmentation (vedere mille varianti) **sparisce**: il modello vede sempre la stessa
versione fissa di ogni immagine. Questo porta a **overfitting** — impara a memoria il
training set e va peggio sui dati nuovi.

### In sintesi, perché l'abbiamo abbandonata

| Aspetto | Effetto della cache |
|---|---|
| Velocità | ✅ Più veloce (encoder calcolato una volta) |
| Correttezza | ❌ Token e maschere disallineati → loss su bersaglio sbagliato |
| Generalizzazione | ❌ Niente augmentation → overfitting |
| Risultato | ❌ mIoU collassa a ~14 |

Il guadagno di velocità non vale niente se il modello che ne esce è rotto.

---

## 2. La pipeline NUOVA: training "live" a tre stadi

Abbiamo buttato la cache e siamo tornati al **forward normale**: a ogni step l'immagine
passa per l'**augmentation live** (trasformazioni casuali fresche) e poi per **tutto** il
modello. Immagine e maschere derivano così **sempre dalla stessa identica vista
aumentata** → nessun disallineamento. È più lento per epoca, ma è **corretto**.

Il training segue un **gradual unfreezing a tre stadi**, nello spirito della guida ("first
fine-tune just the prediction head, then gradually unfreeze the last layers"). I config
sono in [eomt/configs/dinov2/finetuning/](../eomt/configs/dinov2/finetuning/). A ogni stadio
si sblocca un po' di più, partendo dai pesi dello stadio precedente.

### Stage 0 — solo `class_head` + `mask_head` (tutto il resto congelato)
File: `coco_to_cityscapes_stage0_heads.yaml`

- `freeze_all_except_class_mask_heads: True` → si allenano **solo** `class_head` e
  `mask_head` (~1.8M, ~2%). Encoder (DINO), `query` e `upscale` restano congelati ai
  pesi COCO.
- `load_ckpt_class_head: False` → la `class_head` **riparte da zero (random)**: COCO ha
  un numero di classi diverso da Cityscapes (19), la vecchia testa non è riutilizzabile.
- `lr: 1e-4` → la testa è nuova, può imparare in fretta.

Obiettivo: passo più conservativo. Si adatta la **classificazione** alle 19 classi e si
lascia che `mask_head` **ri-orienti le maschere** sulle feature COCO fisse, senza ancora
toccare query/decoder. Output: punto di partenza dello Stage 1.

### Stage 1 — tutta la testa, encoder congelato
File: `coco_to_cityscapes_freeze_backbone.yaml`

- Si parte **dai pesi dello Stage 0**, con `load_ckpt_class_head: True` (la testa è già
  a 19 classi, va tenuta).
- `freeze_encoder: True` → l'encoder è **congelato**, ma ora si allena **tutta** la testa:
  `query` + `mask_head` + `class_head` + scale-block (~6.7M, ~7%). Rispetto allo Stage 0
  si aggiungono `query` e `upscale`, così si adattano anche le query e il decoder conv.
- `lr: 1e-4`.

Obiettivo: adattare l'intera testa (anche il *dove* delle maschere) **senza toccare** il
backbone DINOv2 pre-addestrato.

### Stage 2 — scongela gli ultimi strati
File: `coco_to_cityscapes_unfreeze_last.yaml`

- Si parte **dai pesi salvati dallo Stage 1** (non da COCO): nel notebook si passa il
  best checkpoint dello Stage 1 via `--model.ckpt_path`, con `load_class_head=True`
  (ora la testa è già a 19 classi, va tenuta).
- `freeze_encoder_except_last_n: 2` → si **scongelano gli ultimi 2 blocchi** del
  backbone (gli altri restano congelati) **più tutta la testa** (`query` + `mask_head` +
  `class_head` + `upscale`). In questo modo la testa **si co-adatta** ai nuovi patch token
  prodotti dai blocchi appena scongelati, invece di lavorare "fuori taratura". È il
  *gradual unfreezing* canonico: in Stage 2 alleni tutto ciò che allenavi in Stage 1
  **più** gli ultimi blocchi dell'encoder (~22% dei parametri).
- `lr: 1e-5` (10× più basso) + `llrd: 0.8` → learning rate molto basso e **LLRD**
  (*Layer-wise Learning Rate Decay*): gli strati più profondi e generici dell'encoder
  ricevono un LR ancora più piccolo degli strati finali. Serve a **non distruggere** le
  feature DINOv2 pre-addestrate, che sono preziose. La testa (non-backbone) usa il base LR
  1e-5, quindi si rifinisce in modo gentile partendo dai pesi di Stage 1.
- L'optimizer e lo scheduler ripartono **puliti** (no resume del trainer): così il
  *polynomial LR schedule* calcola correttamente i `total_steps` del solo Stage 2.

> **Nota storica.** Inizialmente il ramo `freeze_encoder_except_last_n`
> ([lightning_module.py](../eomt/training/lightning_module.py)) scongelava solo
> `class_head` + `mask_head` + gli ultimi blocchi, **lasciando `query` e `upscale`
> congelati ai valori di Stage 1**. Era un'asimmetria incoerente: cambiavi le feature
> dell'encoder ma bloccavi due moduli (le query e il decoder conv) che proprio su quelle
> feature devono operare. Ora il codice scongela l'intera testa, com'è corretto.

### Perché ora è corretto

- **Allineamento**: immagine e maschere vengono dalla stessa augmentation → la
  mask/dice loss ottimizza il bersaglio giusto.
- **Augmentation viva**: ogni epoca vede viste nuove → meno overfitting.
- **Trasferimento graduale (3 stadi)**: prima solo `class_head`+`mask_head` (rischio
  minimo), poi tutta la testa, infine un ritocco fine degli ultimi blocchi dell'encoder
  (rischio controllato da LR basso + LLRD). Ogni stadio parte dai pesi del precedente. È
  la ricetta classica e robusta di fine-tuning.

### Cosa compensa la perdita di velocità

Per recuperare il tempo perso (non avendo più la cache) usiamo accorgimenti che **non**
intaccano la correttezza:

- **AMP** (`bf16-mixed`): *Automatic Mixed Precision*. Si calcola in precisione ridotta
  (bfloat16) dove possibile → meno memoria, più veloce. `bf16` è più stabile di `fp16`
  sulle loss dice/focal.
- **`torch.compile`**: compila il modello in un grafo ottimizzato → step più rapidi.
- **risoluzione 512×512** invece di 1024×1024 → circa **1/4** del calcolo, batch più
  grandi sulla GPU L4.
- **EarlyStopping**: ferma il training quando la `val_iou_all` non migliora più (default
  5 epoche di pazienza) → non si sprecano epoche inutili.

---

## 3. Cosa resta nel repo (codice morto)

Il codice della cache è ancora presente ma **non più usato**:
[eomt/training/head_cache_utils.py](../eomt/training/head_cache_utils.py) e i percorsi
`*_prova_finale.yaml` / `eomt_finetuning.py`. Sono lì solo per riferimento storico; la
pipeline ufficiale del Task 5 è quella a tre stadi descritta sopra. Il vecchio notebook è
stato rinominato `ignore_Task5_fineTuning.ipynb`; quello buono è
`task5_emot_finetuning_on_cityscapes.ipynb`.

---

## Glossario

- **Backbone / Encoder**: la rete grande (DINOv2 ViT) che trasforma l'immagine in token.
  La parte costosa da calcolare.
- **Token**: vettori numerici che descrivono porzioni dell'immagine; l'output
  dell'encoder, l'input della testa.
- **Testa / head**: la parte piccola del modello (query, `mask_head`, `class_head`) che
  dai token produce maschere e classi.
- **Congelare (freeze)**: impedire a un insieme di pesi di aggiornarsi durante il
  training (`requires_grad = False`). Si allena solo il resto.
- **Cache (dei token)**: salvare su disco un risultato calcolato una volta per non
  ricalcolarlo. Qui: i token dell'encoder congelato.
- **Data augmentation**: trasformazioni casuali applicate alle immagini a ogni epoca
  (flip, scale, crop) per migliorare la generalizzazione.
- **Disallineamento (misalignment)**: quando i dati di input (token) e i target
  (maschere) si riferiscono a viste/trasformazioni diverse della stessa immagine.
- **Loss**: la funzione che misura l'errore del modello e ne guida l'apprendimento.
  Qui include *focal* (classi), *mask* e *dice* (qualità delle maschere).
- **mIoU**: *mean Intersection-over-Union*, metrica di qualità della segmentazione
  (0–100): media della sovrapposizione tra regioni predette e vere.
- **Overfitting**: il modello impara a memoria il training set e va peggio sui dati nuovi.
- **Fine-tuning**: riadattare un modello già addestrato a un nuovo dataset/compito.
- **Learning rate (LR)**: quanto "grande" è ogni passo di aggiornamento dei pesi.
- **LLRD** (*Layer-wise Learning Rate Decay*): LR più piccolo per gli strati profondi,
  più grande per quelli finali; protegge le feature pre-addestrate.
- **AMP / bf16-mixed**: *Automatic Mixed Precision*; calcoli in precisione ridotta
  (bfloat16) per velocità e risparmio di memoria.
- **Checkpoint (.ckpt / .bin)**: file con i pesi del modello salvati a un certo punto del
  training.
- **EarlyStopping**: callback che ferma il training quando la metrica di validazione
  smette di migliorare.
