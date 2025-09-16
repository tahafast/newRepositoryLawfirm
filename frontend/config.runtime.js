// Robust API base resolution (prod + local)
function sanitize(v){ return (v||"").toString().trim().replace(/\/+$/,""); }
function isHttpUrl(v){ try{ const u=new URL(v); return /^https?:$/.test(u.protocol); }catch{return false} }

// 1) explicit override
let fromLS = "";
try { fromLS = sanitize(localStorage.getItem("API_BASE")); } catch {}

// 2) same-origin (works on Render)
const origin = (typeof location!=="undefined" && /^https?:$/.test(location.protocol)) ? sanitize(location.origin) : "";
const onRender = typeof location!=="undefined" && /onrender\.com$/i.test(location.hostname);
const isLocalHost = (h)=>/(^|\/\/)(localhost|127\.0\.0\.1|0\.0\.0\.0)(:|\/|$)/i.test(h||"");

// Auto-heal: if we're on Render but override is localhost/invalid, drop it
if (fromLS && (!isHttpUrl(fromLS) || (onRender && isLocalHost(fromLS)))) {
  try { localStorage.removeItem("API_BASE"); } catch {}
  fromLS = "";
}

export const API_BASE = sanitize(fromLS || origin || "http://127.0.0.1:8000");
export const UPLOAD_ENDPOINT = `${API_BASE}/api/v1/lawfirm/upload-document`;
export const QUERY_ENDPOINT  = `${API_BASE}/api/v1/lawfirm/query`;
export const HEALTH_ENDPOINT = `${API_BASE}/healthz`;

export async function ping(){
  try { const r = await fetch(HEALTH_ENDPOINT, { method:"GET", mode:"cors" }); return r.ok; } catch { return false; }
}

export async function postJSON(url, body){
  const res = await fetch(url, { method:"POST", mode:"cors", headers:{ "Content-Type":"application/json" }, body: JSON.stringify(body) });
  const text = await res.text(); let data; try{ data=JSON.parse(text) }catch{ data={error:text} }
  if(!res.ok) throw new Error(data?.error || data?.detail || `${res.status} ${res.statusText}`); return data;
}

export async function postFormData(url, fd){
  const res = await fetch(url, { method:"POST", mode:"cors", body: fd });
  const text = await res.text(); let data; try{ data=JSON.parse(text) }catch{ data={error:text} }
  if(!res.ok) throw new Error(data?.error || data?.detail || `${res.status} ${res.statusText}`); return data;
}

// Optional debug:
// window.__API_BASE__ = API_BASE;