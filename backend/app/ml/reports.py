from __future__ import annotations

from datetime import datetime, timezone

from app.ml.imaging import CHEST_LABELS, MRI_LABELS


DISCLAIMER = (
    "Research / education prototype only. Not a medical device, not FDA/CE cleared, "
    "and not for clinical diagnosis. A licensed radiologist must interpret all studies."
)


def structured_report(
    modality: str,
    findings: list[dict],
    primary: str,
    regions: list[dict],
    model_mode: str,
) -> str:
    present = [f["label"] for f in findings if f["present"]]
    if modality == "chest_xray":
        technique = "Frontal chest radiograph. Comparison: none available."
        if not present:
            findings_text = (
                "The cardiomediastinal silhouette is within expected limits for this prototype. "
                "No high-confidence multi-label finding exceeded the decision threshold."
            )
            impression = "No acute cardiopulmonary finding detected by the assistant (thresholded)."
        else:
            scored = ", ".join(
                f"{f['label']} ({f['probability']:.0%})" for f in findings if f["present"]
            )
            findings_text = (
                f"Computer-aided detection suggests: {scored}. "
                "Lung volumes and technical factors were not independently verified."
            )
            impression = f"Possible {primary}. Correlate with history, labs, and prior imaging."
        localization = _loc_sentence(regions)
        return _wrap(
            "CHEST RADIOGRAPH — AI PRELIMINARY REPORT",
            technique,
            findings_text + " " + localization,
            impression,
            model_mode,
        )

    present_tumor = [f for f in findings if f["present"] and f["label"] != "notumor"]
    technique = "Brain MRI (single slice / uploaded series representative frame)."
    if not present_tumor:
        findings_text = "No high-confidence neoplastic pattern was assigned by the classifier."
        impression = "No tumor class predicted above threshold, or class notumor selected."
    else:
        top = present_tumor[0]
        findings_text = (
            f"The classifier favors {top['label']} (p={top['probability']:.2f}). "
            "Mass effect, midline shift, and extra-axial vs intra-axial origin require radiologist review."
        )
        impression = f"Possible {top['label']}. Recommend full series review and clinical correlation."
    localization = _loc_sentence(regions)
    return _wrap(
        "BRAIN MRI — AI PRELIMINARY REPORT",
        technique,
        findings_text + " " + localization,
        impression,
        model_mode,
    )


def _loc_sentence(regions: list[dict]) -> str:
    if not regions:
        return "No discrete localization mask was retained after thresholding."
    r = regions[0]
    x0, y0, x1, y1 = r["bbox"]
    return (
        f"An attention/segmentation focus covers {r['mask_coverage']:.0%} of the frame "
        f"(approx. box {x0:.2f},{y0:.2f}–{x1:.2f},{y1:.2f} in normalized coordinates)."
    )


def _wrap(title: str, technique: str, findings: str, impression: str, mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    labels = ", ".join(CHEST_LABELS) if "CHEST" in title else ", ".join(MRI_LABELS)
    return "\n".join(
        [
            title,
            f"Generated: {stamp} | Engine: MedVision CAD ({mode})",
            "",
            "TECHNIQUE:",
            technique,
            "",
            "FINDINGS:",
            findings,
            "",
            "IMPRESSION:",
            impression,
            "",
            "MODEL ONTOLOGY:",
            labels,
            "",
            "DISCLAIMER:",
            DISCLAIMER,
        ]
    )
