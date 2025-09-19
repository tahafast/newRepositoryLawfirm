// BRAG AI – robust API base resolution
// Priority:
// 1) localStorage.API_BASE if valid (explicit override)
// 2) same-origin (Render / any deployed host)
// 3) localhost fallback for dev

function sanitize(v){ return (v||"").toString().trim().replace(/\/+$/,""); }
function isHttpUrl(v){ try{ const u=new URL(v); return /^https?:$/.test(u.protocol); }catch{return false} }

const ls = (typeof window!=="undefined" && window.localStorage) ? window.localStorage.getItem("API_BASE") : "";
let fromLS = sanitize(ls);
const origin = (typeof window!=="undefined" && /^https?:$/.test(location.protocol)) ? sanitize(location.origin) : "";

// Auto-heal bad overrides on hosted envs (localhost override on Render, etc.)
if(fromLS && (!isHttpUrl(fromLS) || /(localhost|127\.0\.0\.1|0\.0\.0\.0)/i.test(fromLS) && origin && !/(localhost|127\.0\.0\.1|0\.0\.0\.0)/i.test(location.hostname))){
  try{ localStorage.removeItem("API_BASE"); }catch{}
  fromLS = "";
}

const API_BASE = sanitize(fromLS || origin || "http://127.0.0.1:8000");
export { API_BASE };

export const UPLOAD_ENDPOINT = `${API_BASE}/api/v1/lawfirm/upload-document`;
export const QUERY_ENDPOINT  = `${API_BASE}/api/v1/lawfirm/query`;
export const HEALTH_ENDPOINT = `${API_BASE}/healthz`;

export async function ping(){
  try{
    const r = await fetch(HEALTH_ENDPOINT, { method:"GET", mode:"cors" });
    return r.ok;
  }catch{ return false; }
}

export async function postJSON(url, body){
  const res = await fetch(url, {
    method:"POST", mode:"cors",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify(body)
  });
  const text = await res.text();
  let data; try{ data = JSON.parse(text) } catch{ data = { error:text } }
  if(!res.ok) throw new Error(data?.error || data?.detail || `${res.status} ${res.statusText}`);
  return data;
}

export async function postFormData(url, fd){
  const res = await fetch(url, { method:"POST", mode:"cors", body:fd });
  const text = await res.text();
  let data; try{ data = JSON.parse(text) } catch{ data = { error:text } }
  if(!res.ok) throw new Error(data?.error || data?.detail || `${res.status} ${res.statusText}`);
  return data;
}

// Debug hook if you need it in console:
// window.__API_BASE__ = API_BASE;