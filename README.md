# MedVision CAD

Educational AI diagnostic assistant for **chest X-rays** and **brain MRI**. It is a hospital-network style prototype: multi-label findings, Grad-CAM / segmentation overlays, structured reports, audit logging, and a React workstation UI.

**Not a medical device.** Not for diagnosis, triage, or any real patient care. Outputs are research/demo only.

## What is implemented

| Requirement | Implementation |
|---|---|
| Chest X-ray multi-label (14 NIH pathologies) | CheXNet-style DenseNet121, BCE training script |
| Brain MRI classification | ResNet50 (`glioma`, `meningioma`, `pituitary`, `notumor`) |
| Explainability | Grad-CAM heatmaps |
| Segmentation | U-Net trainer + Grad-CAM threshold fallback; Mask R-CNN trainer |
| Report generation | Structured clinical template + LSTM decoder trainer with BLEU-4 |
| API / UI | FastAPI + React (Vite) |
| DICOM | `pydicom` window/level conversion |
| Audit | SHA-256 file fingerprints, JSONL access log (no raw PHI stored by default) |
| Multi-agent CAD | Detection, localization, report, consistency critic (`MedXpertCAD`) |

Without trained checkpoints the API still runs in **`imagenet_demo`** mode (ImageNet-initialized backbones). Heatmaps and reports work; pathology accuracy is **not** clinical until you train on NIH / Kaggle data.

## Quick start (Windows)

Python 3.14 is supported with current PyTorch wheels (`torch>=2.14`).

```powershell
cd C:\Users\bygkl\medical-diagnosis-ai
copy .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
# Start from the backend folder so this package is used (repo-root app.py is a separate Gradio sketch).
cd backend
..\.venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd C:\Users\bygkl\medical-diagnosis-ai\frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Use **Load demo sample** or upload a PNG/JPEG/DICOM.

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Training (public datasets)

Place weights in `weights/` using the names in `.env.example`.

**ChestX-ray14 (NIH)**  
Download images + `Data_Entry_2017.csv`, then:

```powershell
python -m training.train_chest --csv data\raw\Data_Entry_2017.csv --images data\raw\images --epochs 8
```

**Brain Tumor MRI (Kaggle)**  
Expected folders `Training|Testing/{glioma,meningioma,notumor,pituitary}`:

```powershell
python -m training.train_mri --data data\raw\brain_mri --epochs 10
```

**Tumor segmentation (Dice target > 0.95 after enough labeled slices):**

```powershell
python -m training.train_segmentation --images data\raw\mri_seg\images --masks data\raw\mri_seg\masks
```

**Reports (BLEU-4, Open-I / IU X-Ray style JSONL):**

```powershell
python -m training.train_report --jsonl data\raw\reports.jsonl
```

**Mask R-CNN (COCO-style instances):**

```powershell
python -m training.train_maskrcnn --images data\raw\instances\images --coco data\raw\instances\annotations.json
```

Industry targets (BLEU-4 > 0.41, Dice > 95%) need the full public corpora, GPU training (SageMaker Studio Lab / EC2 `g4dn` / local CUDA), and hyperparameter search. The trainers log the metrics you need to report.

## Architecture

```
React workstation  →  FastAPI  →  MedXpertCAD
                                  ├─ ChestXrayNet / MRIClassifier
                                  ├─ Grad-CAM (+ optional U-Net)
                                  └─ Structured report + audit JSONL
```

Suggested AWS sketch (not provisioned here): S3 for DICOM, GPU EC2/SageMaker for training, Lambda for DICOM preprocess, API on GPU instance or CPU demo.

## HIPAA-oriented engineering (real deployment)

This repo is a **starting pattern**, not a compliance program:

- Do not persist uploads (`STORE_UPLOADS=false`)
- Log fingerprints, never filenames that contain MRNs
- TLS, IAM, encryption at rest, BAA with the cloud provider
- Human-in-the-loop; keep the radiologist as the attending interpreter

## Tests

```powershell
$env:PYTHONPATH = "$pwd\backend"
pytest backend\tests -q
```
