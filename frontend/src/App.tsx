import { useCallback, useEffect, useMemo, useState } from "react";
import { analyze, getAudit, getHealth, type AnalyzeResponse, type Health, type Modality } from "./api";

const SAMPLES: Record<Modality, string> = {
  chest_xray: "/samples/chest_xray.png",
  brain_mri: "/samples/brain_mri.png",
};

export default function App() {
  const [modality, setModality] = useState<Modality>("chest_xray");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audit, setAudit] = useState<Record<string, unknown>[]>([]);
  const [tab, setTab] = useState<"overlay" | "heatmap" | "mask">("overlay");

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
    getAudit()
      .then((d) => setAudit(d.events))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    setResult(null);
    setFile(null);
    setPreview(null);
    setError(null);
  }, [modality]);

  const onFile = useCallback((next: File | null) => {
    setFile(next);
    setResult(null);
    setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(next ? URL.createObjectURL(next) : null);
  }, [preview]);

  const run = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const data = await analyze(modality, file);
      setResult(data);
      const log = await getAudit();
      setAudit(log.events);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  const loadSample = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(SAMPLES[modality]);
      const blob = await res.blob();
      const sample = new File([blob], modality === "chest_xray" ? "sample-xray.png" : "sample-mri.jpg", {
        type: blob.type || "image/png",
      });
      onFile(sample);
      const data = await analyze(modality, sample);
      setResult(data);
      const log = await getAudit();
      setAudit(log.events);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load or analyze the sample image.");
    } finally {
      setBusy(false);
    }
  };

  const present = useMemo(
    () => result?.findings.filter((f) => f.present).sort((a, b) => b.probability - a.probability) ?? [],
    [result],
  );

  return (
    <div className="shell">
      <aside>
        <div className="brand">
          <span className="mark">MV</span>
          <div>
            <strong>MedVision CAD</strong>
            <p>Chest X-ray &amp; brain MRI assistant</p>
          </div>
        </div>
        <nav>
          <button className={modality === "chest_xray" ? "active" : ""} onClick={() => setModality("chest_xray")}>
            Chest X-ray
          </button>
          <button className={modality === "brain_mri" ? "active" : ""} onClick={() => setModality("brain_mri")}>
            Brain MRI
          </button>
        </nav>
        <div className="health">
          <span className={health ? "dot ok" : "dot"} />
          {health ? `${health.app} · ${health.device} · ${modality === "chest_xray" ? health.chest_mode : health.mri_mode}` : "API offline"}
        </div>
        <section className="audit">
          <h3>Audit log</h3>
          <ul>
            {audit.slice(0, 8).map((row, i) => (
              <li key={i}>
                <em>{String(row.event)}</em>
                <span>{String(row.primary ?? row.env ?? "")}</span>
              </li>
            ))}
          </ul>
        </section>
      </aside>

      <main>
        <header>
          <h1>{modality === "chest_xray" ? "Chest radiograph analysis" : "Brain MRI analysis"}</h1>
          <p>
            Multi-label findings, Grad-CAM localization, and a structured preliminary report. Educational prototype —
            not for clinical use.
          </p>
        </header>

        <div className="workspace">
          <section className="panel">
            <label
              className="drop"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                onFile(e.dataTransfer.files[0] ?? null);
              }}
            >
              <input
                type="file"
                accept="image/*,.dcm,.dicom"
                hidden
                onChange={(e) => onFile(e.target.files?.[0] ?? null)}
              />
              {preview ? <img src={preview} alt="Upload preview" /> : <span>Drop a PNG, JPEG, or DICOM file</span>}
            </label>
            <div className="actions">
              <button type="button" onClick={loadSample} disabled={busy}>
                Load demo sample
              </button>
              <button type="button" className="primary" onClick={run} disabled={!file || busy}>
                {busy ? "Analyzing…" : "Run CAD"}
              </button>
            </div>
            {error && <p className="error">{error}</p>}
          </section>

          <section className="panel results">
            {!result && <p className="muted">Results appear here after inference.</p>}
            {result && (
              <>
                <div className="meta">
                  <span className="chip">{result.model_mode}</span>
                  <span className="chip">{result.latency_ms} ms</span>
                  <span className="chip">{result.primary_impression}</span>
                  {result.dice_proxy != null && <span className="chip">Dice proxy {(result.dice_proxy * 100).toFixed(1)}%</span>}
                </div>
                <div className="tabs">
                  {(["overlay", "heatmap", "mask"] as const).map((id) => (
                    <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
                      {id}
                    </button>
                  ))}
                </div>
                <img
                  className="viz"
                  alt={tab}
                  src={`data:image/png;base64,${tab === "overlay" ? result.overlay_png_b64 : tab === "heatmap" ? result.heatmap_png_b64 : result.segmentation_png_b64 ?? result.overlay_png_b64}`}
                />
                <h3>Findings</h3>
                <ul className="findings">
                  {(present.length ? present : result.findings)
                    .slice()
                    .sort((a, b) => b.probability - a.probability)
                    .slice(0, 8)
                    .map((f) => (
                      <li key={f.label}>
                        <span>{f.label.replaceAll("_", " ")}</span>
                        <b>{(f.probability * 100).toFixed(1)}%</b>
                        <i style={{ width: `${Math.round(f.probability * 100)}%` }} />
                      </li>
                    ))}
                </ul>
              </>
            )}
          </section>
        </div>

        {result && (
          <section className="panel report">
            <h2>Preliminary report</h2>
            <pre>{result.report}</pre>
            <p className="disclaimer">{result.disclaimer}</p>
          </section>
        )}
      </main>
    </div>
  );
}
