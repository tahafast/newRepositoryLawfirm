import { API_BASE } from "./config.js";

const INDEXED_DOCS_ENDPOINT = `${API_BASE}/api/v1/lawfirm/docs/indexed`;
const DOC_SAMPLES_ENDPOINT = `${API_BASE}/api/v1/lawfirm/docs/indexed/samples`;
const PREVIEW_SAMPLE_LIMIT = 5;

let cachedDocs = [];

function normalizeDocs(payload) {
  if (!payload) return [];
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload.documents)) return payload.documents;
  if (Array.isArray(payload.items)) return payload.items;
  if (Array.isArray(payload.results)) return payload.results;
  return [];
}

function coerceDoc(entry) {
  const document =
    entry?.document ??
    entry?.name ??
    entry?.file_name ??
    entry?.source ??
    "unknown";

  const toNumber = (value, fallback = 0) => {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
  };

  const lastSeen =
    entry?.last_seen ?? entry?.lastSeen ?? entry?.timestamp ?? "";

  return {
    document: String(document),
    chunks: toNumber(entry?.chunks ?? entry?.count ?? entry?.total_chunks, 0),
    min_page: toNumber(entry?.min_page ?? entry?.minPage, 0),
    max_page: toNumber(entry?.max_page ?? entry?.maxPage, 0),
    last_seen: lastSeen ? String(lastSeen) : "",
  };
}

function setStatus(message) {
  const statusEl = document.querySelector("#indexedDocsStatus");
  if (!statusEl) return;
  statusEl.textContent = message || "";
}

function renderDocs(docs) {
  const tbody = document.querySelector("#indexedDocsTbody");
  if (!tbody) return;

  if (!docs.length) {
    tbody.innerHTML = "";
    return;
  }

  const rows = docs
    .map(
      (doc) => /* html */ `
      <tr>
        <td>${doc.document}</td>
        <td>${doc.chunks}</td>
        <td>${doc.min_page}</td>
        <td>${doc.max_page}</td>
        <td>${doc.last_seen || "-"}</td>
        <td>
          <button class="docs-preview-btn" data-doc="${encodeURIComponent(
            doc.document
          )}" onclick="previewIndexedDoc(event)">Preview</button>
        </td>
      </tr>`
    )
    .join("");

  tbody.innerHTML = rows;
}

async function loadDocs({ showAll = false, forceReload = true } = {}) {
  const shouldFetch = forceReload || cachedDocs.length === 0;

  if (shouldFetch) {
    setStatus("Loading documents...");
    renderDocs([]);
  }

  try {
    let docs = [];
    if (shouldFetch) {
      const url = `${INDEXED_DOCS_ENDPOINT}?_=${Date.now()}`;
      const response = await fetch(url, { method: "GET" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      docs = normalizeDocs(payload).map(coerceDoc);
      cachedDocs = [...docs];
    } else {
      docs = [...cachedDocs];
    }

    const searchInput = document.querySelector("#indexedDocsSearch");
    const query = (searchInput?.value || "").trim().toLowerCase();

    let filteredDocs = docs;
    if (query && !showAll) {
      filteredDocs = docs.filter((doc) =>
        doc.document.toLowerCase().includes(query)
      );
    }

    filteredDocs.sort((a, b) => {
      const timeA = Date.parse(a.last_seen || "") || 0;
      const timeB = Date.parse(b.last_seen || "") || 0;
      if (timeA !== timeB) {
        return timeB - timeA;
      }
      return a.document.localeCompare(b.document);
    });

    renderDocs(filteredDocs);

    if (!filteredDocs.length) {
      setStatus(query ? "No documents match your search." : "No documents returned.");
    } else {
      setStatus(
        `Showing ${filteredDocs.length} document${filteredDocs.length > 1 ? "s" : ""}.`
      );
    }
  } catch (error) {
    console.error("IndexedDocs load error:", error);
    renderDocs([]);
    setStatus("Unable to load indexed documents right now.");
  }
}

function closePreviewModal() {
  const modal = document.getElementById("docsPreviewModal");
  const modalList = document.getElementById("docsPreviewList");
  const modalLoading = document.getElementById("docsPreviewLoading");
  if (!modal) return;
  modal.classList.add("docs-hidden");
  if (modalList) modalList.innerHTML = "";
  if (modalLoading) modalLoading.style.display = "none";
}

function formatPage(page) {
  const num = Number(page);
  return Number.isFinite(num) ? num : "-";
}

async function previewIndexedDoc(event) {
  event?.preventDefault?.();
  const button = event?.currentTarget || event?.target;
  const docName = decodeURIComponent(button?.dataset?.doc || "");
  if (!docName) return;

  const modal = document.getElementById("docsPreviewModal");
  const modalTitle = document.getElementById("docsPreviewTitle");
  const modalLoading = document.getElementById("docsPreviewLoading");
  const modalList = document.getElementById("docsPreviewList");
  if (!modal || !modalTitle || !modalLoading || !modalList) return;

  modalTitle.textContent = `Preview - ${docName}`;
  modalLoading.style.display = "block";
  modalList.innerHTML = "";
  modal.classList.remove("docs-hidden");

  try {
    const url = new URL(DOC_SAMPLES_ENDPOINT);
    url.searchParams.set("document", docName);
    url.searchParams.set("k", String(PREVIEW_SAMPLE_LIMIT));

    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    const items = Array.isArray(data?.items) ? data.items : [];

    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "docs-preview-empty";
      empty.textContent = "No samples found for this document.";
      modalList.appendChild(empty);
    } else {
      items.forEach((item) => {
        const card = document.createElement("div");
        card.className = "docs-preview-card";

        const meta = document.createElement("div");
        meta.className = "docs-preview-meta";
        meta.textContent = `Page ${formatPage(item.page)} - ${
          item.source || docName
        }`;
        card.appendChild(meta);

        const text = document.createElement("div");
        text.className = "docs-preview-text";
        const sampleText = (item.text || "").trim();
        if (sampleText) {
          text.textContent = sampleText;
        } else {
          const em = document.createElement("em");
          em.textContent = "No text in payload";
          text.appendChild(em);
        }
        card.appendChild(text);
        modalList.appendChild(card);
      });
    }
  } catch (error) {
    console.error("IndexedDocs preview error:", error);
    const fail = document.createElement("div");
    fail.className = "docs-preview-empty";
    fail.textContent = "Unable to fetch samples for this document.";
    modalList.appendChild(fail);
  } finally {
    modalLoading.style.display = "none";
  }
}

window.previewIndexedDoc = previewIndexedDoc;

function initModalHandlers() {
  const modal = document.getElementById("docsPreviewModal");
  if (!modal) return;

  const closeBtn = document.getElementById("docsPreviewClose");
  if (closeBtn) {
    closeBtn.addEventListener("click", () => closePreviewModal());
  }

  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      closePreviewModal();
    }
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !modal.classList.contains("docs-hidden")) {
      closePreviewModal();
    }
  });
}

window.addEventListener("DOMContentLoaded", () => {
  initModalHandlers();

  const showAllBtn = document.querySelector("#indexedDocsShowAll");
  if (showAllBtn) {
    showAllBtn.addEventListener("click", () =>
      loadDocs({ showAll: true, forceReload: true })
    );
  }

  const searchInput = document.querySelector("#indexedDocsSearch");
  if (searchInput) {
    searchInput.addEventListener("input", () =>
      loadDocs({ showAll: false, forceReload: false })
    );
  }

  loadDocs({ showAll: true, forceReload: true });
});
