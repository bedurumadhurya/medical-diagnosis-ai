export type Modality = "chest_xray" | "brain_mri";

export type Finding = {
  label: string;
  probability: number;
  present: boolean;
};

export type Region = {
  label: string;
  score: number;
  bbox: number[];
  mask_coverage: number;
};

export type AnalyzeResponse = {
  request_id: string;
  modality: Modality;
  model_mode: "trained" | "imagenet_demo";
  findings: Finding[];
  primary_impression: string;
  heatmap_png_b64: string;
  overlay_png_b64: string;
  segmentation_png_b64: string | null;
  regions: Region[];
  report: string;
  dice_proxy: number | null;
  disclaimer: string;
  latency_ms: number;
  agents: Record<string, string>;
};

export type Health = {
  status: string;
  app: string;
  device: string;
  chest_mode: string;
  mri_mode: string;
  unet_loaded: boolean;
};

const API = "/api/v1";

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${API}/health`);
  if (!res.ok) throw new Error("API is not reachable");
  return res.json();
}

export async function analyze(modality: Modality, file: File): Promise<AnalyzeResponse> {
  const path = modality === "chest_xray" ? "analyze/xray" : "analyze/mri";
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API}/${path}`, { method: "POST", body });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || "Analyze failed");
  }
  return res.json();
}

export async function getAudit(): Promise<{ events: Record<string, unknown>[] }> {
  const res = await fetch(`${API}/audit`);
  if (!res.ok) throw new Error("Audit fetch failed");
  return res.json();
}
