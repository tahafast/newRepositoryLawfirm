const DEFAULT = "http://127.0.0.1:" + (localStorage.getItem("API_PORT") || "8000");
export const API_BASE = (localStorage.getItem("API_BASE") || new URLSearchParams(location.search).get("api") || DEFAULT).replace(/\/+$/,"");
export const UPLOAD_ENDPOINT = `${API_BASE}/api/v1/lawfirm/upload-document`;
export const QUERY_ENDPOINT  = `${API_BASE}/api/v1/lawfirm/query`;
export async function postJSON(url, body) {
  const res = await fetch(url, { method:"POST", mode:"cors", headers:{ "Content-Type":"application/json" }, body:JSON.stringify(body) });
  const text = await res.text(); let data; try{ data = JSON.parse(text) } catch{ data = { error:text } }
  if (!res.ok) throw new Error(data?.error || res.statusText); return data;
}
export async function postFormData(url, formData) {
  const res = await fetch(url, { method:"POST", mode:"cors", body:formData });
  const text = await res.text(); let data; try{ data = JSON.parse(text) } catch{ data = { error:text } }
  if (!res.ok) throw new Error(data?.error || res.statusText); return data;
}
