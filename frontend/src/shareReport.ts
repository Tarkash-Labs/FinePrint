export type ShareableAnalysis = {
  risk_score: number | null;
  compatibility_score: number | null;
  verdict: 'ACCEPT' | 'NEGOTIATE' | 'REJECT' | null;
  verdict_reason: string | null;
  summary: string | null;
  requirement_breakdown: { requirement: string; met: boolean; explanation: string }[];
  red_flags: {
    clause_title: string;
    clause_text?: string;
    plain_english_explanation: string;
    negotiation_tip: string;
    suggested_rewrite?: string;
    severity: 'high' | 'medium' | 'low';
  }[];
  safe_clauses: { clause_title: string; plain_english_explanation: string }[];
  negotiation_email: string | null;
  timing?: { e4b_ms?: number; moe_ms?: number; dense_explain_ms?: number; dense_email_ms?: number } | null;
};

const SHARE_VERSION = 1;
/** Stay under common proxy/browser URL limits after base64 expansion. */
const MAX_SHARE_URL_CHARS = 14_000;

type SharedReportPayload = {
  v: number;
  contract_type: string;
  analysis: ShareableAnalysis;
};

function base64UrlEncode(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function base64UrlDecode(encoded: string): Uint8Array {
  const padLen = (4 - (encoded.length % 4)) % 4;
  const b64 = encoded.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat(padLen);
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function gzipEncode(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof CompressionStream === 'undefined') return bytes;
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function gzipDecode(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'undefined') return bytes;
  try {
    const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  } catch {
    return bytes;
  }
}

/** Slim payload for URLs — drop clause_text (often huge) and timing. */
function slimAnalysis(analysis: ShareableAnalysis): ShareableAnalysis {
  return {
    ...analysis,
    timing: null,
    red_flags: analysis.red_flags.map(({ clause_text: _ct, ...flag }) => flag),
  };
}

export async function encodeShareReport(
  contractType: string,
  analysis: ShareableAnalysis,
): Promise<string> {
  const payload: SharedReportPayload = {
    v: SHARE_VERSION,
    contract_type: contractType,
    analysis: slimAnalysis(analysis),
  };
  const bytes = new TextEncoder().encode(JSON.stringify(payload));
  const compressed = await gzipEncode(bytes);
  return base64UrlEncode(compressed);
}

export async function decodeShareReport(
  token: string,
): Promise<{ contract_type: string; analysis: ShareableAnalysis }> {
  const raw = await gzipDecode(base64UrlDecode(token));
  const parsed = JSON.parse(new TextDecoder().decode(raw)) as SharedReportPayload;
  if (!parsed?.analysis || parsed.v !== SHARE_VERSION) {
    throw new Error('Invalid or unsupported share link.');
  }
  return {
    contract_type: parsed.contract_type || 'general',
    analysis: parsed.analysis,
  };
}

export async function buildShareUrl(
  contractType: string,
  analysis: ShareableAnalysis,
): Promise<string> {
  const token = await encodeShareReport(contractType, analysis);
  const base = `${window.location.origin}${window.location.pathname}`;
  const url = `${base}#r=${token}`;
  if (url.length > MAX_SHARE_URL_CHARS) {
    throw new Error('Report is too large to share in a link. Export PDF instead.');
  }
  return url;
}

export function readShareTokenFromLocation(): string | null {
  const hash = window.location.hash;
  if (hash.startsWith('#r=')) return hash.slice(3);
  const legacy = new URLSearchParams(window.location.search).get('report');
  return legacy;
}

export function isLegacyServerReportLink(): boolean {
  const params = new URLSearchParams(window.location.search);
  return Boolean(params.get('report')) && !window.location.hash.startsWith('#r=');
}
