// Robust API base resolution for both Render (prod) and local dev.
// Priority:
// 1) Explicit override via localStorage.API_BASE (if not localhost on Render)
// 2) Same-origin (when served via http/https)
// 3) Local dev fallback http://127.0.0.1:8000

function getSameOrigin() {
  try {
    if (typeof window !== "undefined" && /^https?:$/.test(location.protocol)) {
      return location.origin.replace(/\/+$/, "");
    }
  } catch {}
  return "";
}

function sanitizeBase(v) {
  return (v || "").trim().replace(/\/+$/, "");
}

const sameOrigin = getSameOrigin();
const isRender = typeof window !== "undefined" && /onrender\.com$/i.test(location.hostname);
const ls = (typeof window !== "undefined" && window.localStorage) ? window.localStorage.getItem("API_BASE") : "";
let fromLS = sanitizeBase(ls);

// Auto-heal: if running on Render and override points to localhost, ignore it.
if (isRender && /(^|\/\/)(localhost|127\.0\.0\.1)(:|\/|$)/i.test(fromLS)) {
  try { window.localStorage.removeItem("API_BASE"); } catch {}
  fromLS = "";
}

let base = fromLS || sameOrigin || "http://127.0.0.1:8000";
export const API_BASE = sanitizeBase(base);

export const UPLOAD_ENDPOINT = `${API_BASE}/api/v1/lawfirm/upload-document`;
export const QUERY_ENDPOINT  = `${API_BASE}/api/v1/lawfirm/query`;

export async function postJSON(url, body) {
  try {
    const res = await fetch(url, {
      method: "POST",
      mode: "cors",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let data; try { data = JSON.parse(text); } catch { data = { error: text }; }
    if (!res.ok) throw new Error(data?.error || `${res.status} ${res.statusText}`);
    return data;
  } catch (e) {
    console.error("postJSON error:", e);
    throw e;
  }
}

export async function postFormData(url, formData) {
  try {
    const res = await fetch(url, {
      method: "POST",
      mode: "cors",
      body: formData, // do NOT set Content-Type; browser sets multipart boundary
    });
    const text = await res.text();
    let data; try { data = JSON.parse(text); } catch { data = { error: text }; }
    if (!res.ok) throw new Error(data?.error || `${res.status} ${res.statusText}`);
    return data;
  } catch (e) {
    console.error("postFormData error:", e);
    throw e;
  }
}

// Optional: tiny helper available in console to quickly inspect current base.
// window.__RAG_API_BASE__ = API_BASE;