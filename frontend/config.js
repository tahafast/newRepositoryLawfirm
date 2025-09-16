// Robust API base resolution for both Render (prod) and local dev.
// Priority:
// 1) Explicit override via localStorage.API_BASE (if valid)
// 2) Same-origin detection (when served via http/https)
// 3) Local dev fallback http://127.0.0.1:8000

function getSameOrigin() {
  try {
    if (typeof window !== "undefined" && window.location && /^https?:$/.test(window.location.protocol)) {
      // If we're served from /ui path (Render setup), the API is at the root
      let origin = window.location.origin;
      // Remove any /ui path if present
      if (window.location.pathname.startsWith('/ui')) {
        return origin; // API is at root, not /ui
      }
      return origin.replace(/\/+$/, "");
    }
  } catch (e) {
    console.warn("Error getting same origin:", e);
  }
  return "";
}

function sanitizeBase(v) {
  return (v || "").toString().trim().replace(/\/+$/, "");
}

function isValidUrl(str) {
  try {
    const url = new URL(str);
    return /^https?:$/.test(url.protocol);
  } catch {
    return false;
  }
}

// Detect environment
const sameOrigin = getSameOrigin();
const isRender = typeof window !== "undefined" && window.location && /onrender\.com$/i.test(window.location.hostname);
const isLocal = typeof window !== "undefined" && window.location && /(localhost|127\.0\.0\.1|0\.0\.0\.0)/i.test(window.location.hostname);

// Get localStorage override
const ls = (typeof window !== "undefined" && window.localStorage) ? window.localStorage.getItem("API_BASE") : "";
let fromLS = sanitizeBase(ls);

// Auto-heal: remove invalid localStorage values
if (fromLS && (!isValidUrl(fromLS) || (isRender && /(localhost|127\.0\.0\.1)/i.test(fromLS)))) {
  try { 
    window.localStorage.removeItem("API_BASE"); 
    console.log("Removed invalid API_BASE from localStorage:", fromLS);
  } catch {}
  fromLS = "";
}

// Determine API base with fallback chain
let base;
if (fromLS && isValidUrl(fromLS)) {
  base = fromLS;
  console.log("Using API_BASE from localStorage:", base);
} else if (sameOrigin && (isRender || isLocal)) {
  base = sameOrigin;
  console.log("Using same-origin API base:", base);
} else {
  base = "http://127.0.0.1:8000";
  console.log("Using fallback API base:", base);
}

export const API_BASE = sanitizeBase(base);

export const UPLOAD_ENDPOINT = `${API_BASE}/api/v1/lawfirm/upload-document`;
export const QUERY_ENDPOINT  = `${API_BASE}/api/v1/lawfirm/query`;

// Debug info - available in browser console
console.log("=== API Configuration ===");
console.log("Environment:", { isRender, isLocal, hostname: typeof window !== "undefined" ? window.location?.hostname : "N/A" });
console.log("Same Origin:", sameOrigin);
console.log("Final API_BASE:", API_BASE);
console.log("UPLOAD_ENDPOINT:", UPLOAD_ENDPOINT);
console.log("QUERY_ENDPOINT:", QUERY_ENDPOINT);
console.log("========================");

export async function postJSON(url, body) {
  console.log("postJSON request:", { url, body });
  try {
    const res = await fetch(url, {
      method: "POST",
      mode: "cors",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    console.log("postJSON response status:", res.status, res.statusText);
    const text = await res.text();
    let data; 
    try { 
      data = JSON.parse(text); 
    } catch { 
      data = { error: text }; 
      console.warn("Failed to parse JSON response:", text);
    }
    if (!res.ok) {
      const errorMsg = data?.error || data?.detail || `${res.status} ${res.statusText}`;
      console.error("postJSON failed:", errorMsg);
      throw new Error(errorMsg);
    }
    console.log("postJSON success:", data);
    return data;
  } catch (e) {
    console.error("postJSON error:", { url, error: e.message, stack: e.stack });
    throw e;
  }
}

export async function postFormData(url, formData) {
  console.log("postFormData request:", { url, formData });
  try {
    const res = await fetch(url, {
      method: "POST",
      mode: "cors",
      body: formData, // do NOT set Content-Type; browser sets multipart boundary
    });
    console.log("postFormData response status:", res.status, res.statusText);
    const text = await res.text();
    let data; 
    try { 
      data = JSON.parse(text); 
    } catch { 
      data = { error: text }; 
      console.warn("Failed to parse JSON response:", text);
    }
    if (!res.ok) {
      const errorMsg = data?.error || data?.detail || `${res.status} ${res.statusText}`;
      console.error("postFormData failed:", errorMsg);
      throw new Error(errorMsg);
    }
    console.log("postFormData success:", data);
    return data;
  } catch (e) {
    console.error("postFormData error:", { url, error: e.message, stack: e.stack });
    throw e;
  }
}

