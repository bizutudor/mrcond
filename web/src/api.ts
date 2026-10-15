// Thin fetch wrappers for the MRCond REST API (FastAPI, see src/mrcond/api.py).
import type {
  CheckResult,
  DocumentPages,
  DocumentSummary,
  FieldDecision,
  FieldSpec,
  ImplantSummary,
  LabelDocument,
  ProtocolPlan,
  ScannerProfile,
} from "./types";
import { formatErrorDetail } from "./lib/errors";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      detail = formatErrorDetail(body.detail, res.statusText);
    } catch {
      // not JSON
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  fields: () => request<FieldSpec[]>("/api/fields"),
  documents: () => request<DocumentSummary[]>("/api/documents?include_synthetic=true"),
  pages: (docId: string) => request<DocumentPages>(`/api/documents/${encodeURIComponent(docId)}/pages`),
  annotation: (docId: string) => request<LabelDocument>(`/api/documents/${encodeURIComponent(docId)}/annotation`),
  extraction: (docId: string, extractor: string) =>
    request<LabelDocument>(
      `/api/documents/${encodeURIComponent(docId)}/extractions/${encodeURIComponent(extractor)}`,
    ),
  verify: (docId: string, body: { reviewer: string; base: string; decisions: FieldDecision[]; notes?: string }) =>
    request<LabelDocument>(`/api/documents/${encodeURIComponent(docId)}/verify`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  implants: () => request<ImplantSummary[]>("/api/implants"),
  implant: (id: string) => request<LabelDocument>(`/api/implants/${encodeURIComponent(id)}`),
  scannerProfiles: () => request<ScannerProfile[]>("/api/scanner-profiles"),
  check: (body: { implant_ids: string[]; scanner: ScannerProfile; protocol: ProtocolPlan }) =>
    request<CheckResult>("/api/check", { method: "POST", body: JSON.stringify(body) }),
};
