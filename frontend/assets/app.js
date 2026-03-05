let accessToken = null;
let lastQuery = "";
let lastQueryResults = [];

const PROMPT_HISTORY_KEY = "legacylens_prompt_history";
const PROMPT_HISTORY_MAX = 10;

const el = (id) => document.getElementById(id);

function loadPromptHistory() {
  try {
    const raw = localStorage.getItem(PROMPT_HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function savePromptHistory(query, answer, results) {
  if (!query.trim()) return;
  const q = query.trim();
  let history = loadPromptHistory();
  history = history.filter((h) => h.query !== q);
  history.unshift({
    query: q,
    ts: Date.now(),
    answer: answer || "",
    results: results || [],
  });
  history = history.slice(0, PROMPT_HISTORY_MAX);
  try {
    localStorage.setItem(PROMPT_HISTORY_KEY, JSON.stringify(history));
  } catch {}
}

function deleteHistoryItem(idx) {
  let history = loadPromptHistory();
  history = history.filter((_, i) => i !== idx);
  try {
    localStorage.setItem(PROMPT_HISTORY_KEY, JSON.stringify(history));
  } catch {}
  renderPromptHistory();
}

function renderPromptHistory() {
  const list = el("prompt-history-list");
  if (!list) return;
  const history = loadPromptHistory();
  if (history.length === 0) {
    list.innerHTML = '<li class="history-empty">No prompts yet.</li>';
    return;
  }
  list.innerHTML = history
    .map(({ query, ts }, i) => {
      const preview = query.length > 45 ? query.slice(0, 45) + "…" : query;
      const date = new Date(ts).toLocaleString();
      return `<li class="history-item-wrap"><button type="button" class="history-item" data-idx="${i}" title="${escapeHtml(date)}">${escapeHtml(preview)}</button><button type="button" class="history-item-delete" data-idx="${i}" title="Delete">×</button></li>`;
    })
    .join("");
}

function displayCachedAnswer(answer, results) {
  const answerEl = el("chat-answer");
  if (!answerEl) return;
  lastQueryResults = results || [];
  const html = formatAnswerWithCode(answer || "(No answer)", lastQueryResults);
  answerEl.innerHTML = html;
  answerEl.style.display = "block";
  answerEl.querySelectorAll(".answer-code-block pre code, .cite-source-block pre code").forEach(safeHighlightCode);
  const LINES_PER_LOAD = 10;
  const fileChunksCache = {};
  function buildFullContentFromChunks(chunks) {
    if (!chunks || !chunks.length) return { lines: [], total: 0 };
    chunks.sort((a, b) => (a.start_line || 0) - (b.start_line || 0));
    const lineMap = {};
    for (const c of chunks) {
      const s = c.start_line || 0;
      (c.code_snippet || "").split("\n").forEach((line, i) => {
        lineMap[s + i] = line;
      });
    }
    const total = Math.max(...chunks.map((c) => c.end_line || 0), 0);
    const lines = [];
    for (let i = 1; i <= total; i++) lines.push(lineMap[i] ?? "");
    return { lines, total };
  }
  answerEl.querySelectorAll(".answer-show-more").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const path = btn.dataset.path;
      const direction = btn.dataset.direction || "down";
      let startLine = parseInt(btn.dataset.startLine, 10) || 0;
      let endLine = parseInt(btn.dataset.endLine, 10) || 0;
      const block = btn.closest(".answer-code-block") || btn.closest(".cite-source-block");
      const pre = block?.querySelector("pre");
      const code = pre?.querySelector("code");
      const topBtn = block?.querySelector(".expand-top");
      const bottomBtn = block?.querySelector(".expand-bottom");
      if (!path || !code) return;
      btn.disabled = true;
      btn.textContent = "…";
      try {
        let more = "";
        let total = 0;
        if (!fileChunksCache[path]) {
          const fcRes = await fetch(`/file-chunks?path=${encodeURIComponent(path)}`);
          if (fcRes.ok) {
            const fcData = await fcRes.json().catch(() => ({}));
            if (fcData.chunks && fcData.chunks.length) {
              fileChunksCache[path] = buildFullContentFromChunks(fcData.chunks);
            }
          }
        }
        const cached = fileChunksCache[path];
        if (cached && cached.lines) {
          total = cached.total;
          let fetchStart, fetchEnd;
          if (direction === "down") {
            fetchStart = endLine + 1;
            fetchEnd = Math.min(endLine + LINES_PER_LOAD, total);
            more = cached.lines.slice(fetchStart - 1, fetchEnd).join("\n");
          } else {
            if (startLine <= 1) {
              topBtn?.setAttribute("style", "display:none");
              btn.textContent = "↟";
              btn.disabled = false;
              return;
            }
            fetchEnd = startLine - 1;
            fetchStart = Math.max(1, startLine - LINES_PER_LOAD);
            more = cached.lines.slice(fetchStart - 1, fetchEnd).join("\n");
          }
        }
        if (!more && !cached) {
          const res = await fetch(
            `/file-content?path=${encodeURIComponent(path)}&start_line=${direction === "down" ? endLine + 1 : Math.max(1, startLine - LINES_PER_LOAD)}&end_line=${direction === "down" ? endLine + LINES_PER_LOAD : startLine - 1}`
          );
          const data = await res.json().catch(() => ({}));
          if (res.ok) {
            more = (data.content || "").trim();
            total = data.total_lines || 0;
          } else if (res.status === 404 && !btn.dataset.file404Shown) {
            btn.dataset.file404Shown = "1";
            const status = document.getElementById("query-status");
            if (status) status.textContent = `File not found: ${path}`;
          }
        }
        if (more) {
          const current = code.textContent || "";
          if (direction === "down") {
            code.textContent = current + (current ? "\n" : "") + more;
            endLine = endLine + more.split("\n").length;
            bottomBtn?.setAttribute("data-end-line", String(endLine));
            if (total > 0 && endLine >= total) bottomBtn?.setAttribute("style", "display:none");
          } else {
            code.textContent = more + (current ? "\n" : "") + current;
            startLine = startLine - more.split("\n").length;
            topBtn?.setAttribute("data-start-line", String(startLine));
            if (startLine <= 1) topBtn?.setAttribute("style", "display:none");
          }
          block?.classList.add("expanded");
          const collapseBtn = block?.querySelector(".collapse-btn");
          if (collapseBtn) collapseBtn.style.display = "";
          if (block?.classList.contains("cite-source-block")) bottomBtn?.setAttribute("data-end-line", String(endLine));
          safeHighlightCode(code);
          pre.classList.add("stream-in");
          setTimeout(() => pre.classList.remove("stream-in"), 350);
        }
        btn.textContent = direction === "down" ? "↡" : "↟";
        btn.disabled = false;
      } catch {
        btn.textContent = direction === "down" ? "↡" : "↟";
        btn.disabled = false;
      }
    });
  });
  answerEl.querySelectorAll(".collapse-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const block = btn.closest(".answer-code-block");
      const code = block?.querySelector("pre code");
      const topBtn = block?.querySelector(".expand-top");
      const bottomBtn = block?.querySelector(".expand-bottom");
      if (!code) return;
      const original = code.getAttribute("data-original-content") || "";
      code.textContent = original;
      safeHighlightCode(code);
      block?.classList.remove("expanded");
      btn.style.display = "none";
      const origStart = parseInt(block?.dataset.originalStart || "1", 10);
      const origEnd = parseInt(block?.dataset.originalEnd || "1", 10);
      if (topBtn) {
        topBtn.style.display = origStart > 1 ? "" : "none";
        topBtn.dataset.startLine = String(origStart);
        topBtn.dataset.endLine = String(origEnd);
      }
      if (bottomBtn) {
        bottomBtn.style.display = "";
        bottomBtn.dataset.startLine = String(origStart);
        bottomBtn.dataset.endLine = String(origEnd);
      }
    });
  });
  answerEl.querySelectorAll(".cite-link").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const idx = parseInt(link.dataset.cite, 10);
      const target = document.getElementById(`cite-${idx}`) || document.getElementById(`cite-source-${idx}`);
      if (target) {
        const details = target.closest("details");
        if (details && !details.open) details.open = true;
        target.scrollIntoView({ behavior: "smooth", block: "nearest" });
        target.classList.add("cite-highlight");
        setTimeout(() => target.classList.remove("cite-highlight"), 1500);
      }
    });
  });
  answerEl.querySelectorAll(".code-block-cite-link").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const href = link.getAttribute("href") || "";
      const id = href.replace("#", "");
      const target = id ? document.getElementById(id) : null;
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "nearest" });
        target.classList.add("cite-highlight");
        setTimeout(() => target.classList.remove("cite-highlight"), 1500);
      }
    });
  });
}

function onPromptHistoryClick(e) {
  const delBtn = e.target.closest(".history-item-delete");
  if (delBtn) {
    e.preventDefault();
    e.stopPropagation();
    deleteHistoryItem(parseInt(delBtn.dataset.idx, 10));
    return;
  }
  const btn = e.target.closest(".history-item");
  if (!btn) return;
  e.preventDefault();
  e.stopPropagation();
  const idx = parseInt(btn.dataset.idx, 10);
  const history = loadPromptHistory();
  const item = history[idx];
  if (item) {
    el("query-text").value = item.query;
    el("query-text").focus();
    if (item.answer != null && item.results != null) {
      el("query-status").textContent = `Cached. ${(item.results || []).length} chunks.`;
      displayCachedAnswer(item.answer, item.results);
    }
  }
}

function appendToSearchBox(text) {
  const box = el("query-text");
  if (!box) return;
  const t = (text || "").trim();
  if (!t) return;
  const current = (box.value || "").trim();
  box.value = current ? current + "\n" + t : t;
  box.focus();
}

function onAnswerClick(e) {
  if (e.target.closest(".cite-link") || e.target.closest(".answer-show-more")) return;
  const token = e.target.closest(".reprompt-token, .comment-ref, .syn-macro, .syn-function, .include-ref, .hljs-title, .hljs-name, .hljs-built_in");
  if (!token) return;
  const name = (token.dataset?.path || token.textContent || "").trim();
  if (name.length < 1) return;
  e.preventDefault();
  e.stopPropagation();
  appendToSearchBox(name);
}

function setSession(signedIn) {
  el("session-text").textContent = signedIn ? "Signed in" : "Signed out";
}

async function login() {
  const token = el("admin-token").value.trim();
  const status = el("login-status");
  if (!token) {
    status.textContent = "Enter admin token.";
    return;
  }
  status.textContent = "Signing in...";
  try {
    const res = await fetch("/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      status.textContent = data.detail || "Login failed.";
      return;
    }
    accessToken = data.access_token;
    status.textContent = "Signed in.";
    setSession(true);
    loadRuntimeConfig();
    refreshCount();
    loadCorpus();
  } catch {
    status.textContent = "Network error during login.";
  }
}

async function loadStorageInfo() {
  const elm = el("storage-info");
  if (!elm) return;
  try {
    const res = await fetch("/health/db");
    const data = await res.json().catch(() => ({}));
    if (data.status === "ok") {
      elm.textContent = `Storage: ${data.storage_mode} | ${data.point_count} points in ${data.collection}`;
    } else {
      elm.textContent = `Storage: ${data.error || "unknown"}`;
    }
  } catch {
    elm.textContent = "Storage: could not reach /health/db";
  }
}

async function refreshCount() {
  loadStorageInfo();
  const source = el("browse-source").value;
  const summary = el("db-summary");
  summary.textContent = "Loading...";
  try {
    const res = await fetch("/chunks?limit=1&source_type=" + encodeURIComponent(source));
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      summary.textContent = data.detail || "Failed to load count.";
      return;
    }
    summary.textContent = `Total chunks: ${data.total_chunks} (${source})`;
  } catch {
    summary.textContent = "Network error while loading count.";
  }
}

async function loadCorpus() {
  const source = el("browse-source").value;
  const list = el("corpus-list");
  list.textContent = "Loading...";
  try {
    const res = await fetch("/chunks?limit=40&source_type=" + encodeURIComponent(source));
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      list.textContent = data.detail || "Failed to load chunks.";
      return;
    }
    const chunks = data.chunks || [];
    if (!chunks.length) {
      list.textContent = "No chunks found for this filter.";
      return;
    }
    list.textContent = chunks
      .map((c, i) => {
        const kind = c.source_type || "code";
        const lang = c.language || "?";
        const fp = c.file_path || "?";
        const span = `L${c.start_line ?? "?"}-${c.end_line ?? "?"}`;
        return `${i + 1}. [${kind}/${lang}] ${fp} ${span}`;
      })
      .join("\n");
  } catch {
    list.textContent = "Network error while loading chunks.";
  }
}

function authHeaders(includeContentType = false) {
  const headers = {};
  if (includeContentType) {
    headers["Content-Type"] = "application/json";
  }
  if (accessToken) {
    headers["Authorization"] = "Bearer " + accessToken;
  }
  return headers;
}

async function loadRuntimeConfig() {
  const status = el("runtime-config");
  if (!accessToken) {
    status.textContent = "";
    return;
  }
  status.textContent = "Loading active config...";
  try {
    const res = await fetch("/admin/runtime-config", {
      headers: authHeaders(false),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      status.textContent = data.detail || "Could not load runtime config.";
      return;
    }
    const exts = (data.code_extensions || "").trim();
    const mode = exts === "*" ? "ALL FILES" : "FILTERED";
    status.textContent =
      `Active config -> CODE_ROOT=${data.code_root ?? "(unset)"} | ` +
      `CODE_EXTENSIONS=${exts || "(unset)"} [${mode}] | ` +
      `MAX_FILE_SIZE_MB=${data.max_file_size_mb} | ` +
      `INGEST_TEXT_ONLY=${data.ingest_text_only}`;
  } catch {
    status.textContent = "Network error loading runtime config.";
  }
}

let logsRefreshInterval = null;
async function loadLogs() {
  const placeholder = el("logs-placeholder");
  const table = el("logs-table");
  const tbody = el("logs-tbody");
  if (!placeholder || !table || !tbody) return;
  if (!accessToken) {
    placeholder.textContent = "Sign in (Admin tab) to load request logs.";
    placeholder.style.display = "block";
    table.style.display = "none";
    return;
  }
  try {
    const res = await fetch("/admin/logs", {
      headers: authHeaders(false),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      placeholder.textContent = "Failed to load logs: " + (data.detail || res.status);
      placeholder.style.display = "block";
      table.style.display = "none";
      return;
    }
    const logs = data.logs || [];
    if (!logs.length) {
      placeholder.textContent = "No request logs yet. Run a query or chat to see latency breakdown.";
      placeholder.style.display = "block";
      table.style.display = "none";
      return;
    }
    placeholder.style.display = "none";
    table.style.display = "table";
    tbody.innerHTML = logs
      .map((row) => {
        const ts = row.ts ? new Date(row.ts * 1000).toLocaleTimeString() : "—";
        const typeCls = row.type === "chat" ? "type-chat" : "type-query";
        const rerank = row.rerank_ms != null ? row.rerank_ms : "—";
        const llm = row.llm_ms != null ? row.llm_ms : "—";
        const inTok = row.input_tokens != null ? row.input_tokens : "—";
        const outTok = row.output_tokens != null ? row.output_tokens : "—";
        return `<tr>
          <td>${ts}</td>
          <td class="${typeCls}">${row.type || "—"}</td>
          <td>${row.total_ms ?? "—"}</td>
          <td>${row.embed_ms ?? "—"}</td>
          <td>${row.search_ms ?? "—"}</td>
          <td>${rerank}</td>
          <td>${llm}</td>
          <td>${inTok}</td>
          <td>${outTok}</td>
        </tr>`;
      })
      .join("");
  } catch (e) {
    placeholder.textContent = "Network error loading logs.";
    placeholder.style.display = "block";
    table.style.display = "none";
  }
}

async function resetDb() {
  const status = el("db-status");
  status.textContent = "Resetting DB...";
  try {
    const res = await fetch("/admin/reset-db", {
      method: "POST",
      headers: authHeaders(false),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      status.textContent = "Reset failed: " + (data.detail || JSON.stringify(data));
      return;
    }
    status.textContent = "Reset complete.";
    refreshCount();
    loadCorpus();
  } catch {
    status.textContent = "Network error during reset.";
  }
}

async function reingest() {
  const codeRoot = el("code-root").value.trim();
  const codeExtensions = el("code-extensions").value.trim();
  const maxFilesVal = (el("max-files")?.value || "").trim();
  const maxFiles = maxFilesVal ? parseInt(maxFilesVal, 10) : null;
  const batchSizeVal = (el("batch-size")?.value || "").trim();
  const batchSize = batchSizeVal ? parseInt(batchSizeVal, 10) : null;
  const status = el("reingest-status");
  status.textContent = "Reingesting, this can take a while...";
  try {
    const body = {};
    if (codeRoot) body.code_root = codeRoot;
    if (codeExtensions) body.code_extensions = codeExtensions;
    if (maxFiles != null && maxFiles > 0) body.max_files = maxFiles;
    if (batchSize != null && batchSize >= 10) body.batch_size = batchSize;
    const res = await fetch("/admin/reingest", {
      method: "POST",
      headers: authHeaders(true),
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      status.textContent = "Reingest failed: " + (data.detail || JSON.stringify(data));
      return;
    }
    status.textContent =
      `Done. files=${data.files_ingested ?? 0}, chunks=${data.chunks_upserted ?? 0}, ` +
      `ext=${data.code_extensions ?? "?"}. Storage: ${data.storage ?? "local"}`;
    refreshCount();
    loadCorpus();
  } catch {
    status.textContent = "Network error during reingest.";
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

const C_KEYWORDS = new Set(["void", "int", "char", "float", "double", "long", "short", "signed", "unsigned", "const", "volatile", "static", "extern", "struct", "union", "enum", "typedef", "return", "if", "else", "while", "for", "do", "switch", "case", "break", "continue", "default", "sizeof", "typeof", "inline", "register", "goto", "restrict", "bool", "true", "false"]);

function postProcessHighlightedCode(html, lang) {
  if (!html) return html;
  let out = html;
  if (lang === "c" || lang === "cpp" || lang === "csharp") {
    out = out.replace(/(#include\s+)&lt;([^&gt;]+)&gt;/g, (_, pre, path) =>
      `${pre}<span class="syn-punct">&lt;</span><span class="include-ref" data-path="${path}">${path}</span><span class="syn-punct">&gt;</span>`
    );
    out = out.replace(/(#include\s+)(&quot;|")([^"&]+)\2/g, (_, pre, q, path) =>
      `${pre}<span class="syn-punct">${q}</span><span class="include-ref" data-path="${path.trim()}">${path.trim()}</span><span class="syn-punct">${q}</span>`
    );
    out = out.replace(/#define\s+([a-zA-Z_][a-zA-Z0-9_]*)(\s*\([^)]*\))?/g, (m, name, params) => {
      const paramPart = params ? params.replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\b/g, (p) => `<span class="syn-param">${p}</span>`) : "";
      return `<span class="syn-preprocessor">#define</span> <span class="syn-macro">${name}</span>${paramPart}`;
    });
    out = out.replace(/\b(void|int|char|float|double|long|short|signed|unsigned|const|volatile|static|extern|struct|union|enum|typedef|return|if|else|while|for|do|switch|case|break|continue|default|sizeof|inline|register|goto|restrict|_Bool|bool|true|false)\b/g, (m) => `<span class="syn-keyword">${m}</span>`);
    out = out.replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g, (m, name) => {
      if (C_KEYWORDS.has(name)) return m;
      return `<span class="syn-function">${name}</span><span class="syn-punct">(</span>`;
    });
  }
  out = out.replace(
    /(<span class="hljs-comment">)([\s\S]*?)(<\/span>)/g,
    (_, open, content, close) => {
      const processed = content.replace(
        /\b([a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)|[A-Z][A-Z0-9_-]{2,})\b/g,
        (m) => `<span class="comment-ref">${m}</span>`
      );
      return open + processed + close;
    }
  );
  return out;
}

// Languages highlight.js supports. Unknown -> plaintext to avoid "Could not find language" errors.
const HLJS_SAFE_LANGS = new Set(["plaintext", "cobol", "c", "cpp", "bash", "sh", "python", "javascript", "json", "xml", "html", "css", "sql", "yaml", "ini", "markdown", "diff", "makefile"]);
function safeHighlightCode(block) {
  if (typeof hljs === "undefined") return;
  let lang = (block.className.match(/language-(\w+)/)?.[1] || "plaintext").toLowerCase();
  if (!HLJS_SAFE_LANGS.has(lang)) lang = "plaintext";
  const code = block.textContent || block.innerText || "";
  try {
    const result = hljs.highlight(code, { language: lang });
    block.innerHTML = postProcessHighlightedCode(result.value, lang);
    block.classList.add("hljs");
  } catch {
    block.textContent = code;
  }
}

function inferLanguage(filePath, lang) {
  const path = (filePath || "").toLowerCase();
  const ext = path.split(".").pop() || "";
  if (ext === "cob" || ext === "cbl") return "cobol";
  if (ext === "c" || ext === "h") return "c";
  if (ext === "sh" || ext === "bash") return "bash";
  if (ext === "ac" || ext === "in" || ext === "m4" || ext === "am" || path.includes("configure")) return "bash";
  if (ext === "at" || ext === "conf" || ext === "cfg") return "plaintext";
  if (lang) {
    const l = String(lang).toLowerCase().replace(/\s+/g, "");
    if (HLJS_SAFE_LANGS.has(l)) return l;
    if (l === "cheader" || l === "cheader") return "c";
    if (l === "text") return "plaintext";
    if (l === "markdown") return "markdown";
  }
  return "plaintext";
}

function formatAnswerWithCode(text, results) {
  const parts = [];
  let lastIndex = 0;
  const re = /```(\w*)\n?([\s\S]*?)```/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    parts.push({ type: "text", content: text.slice(lastIndex, m.index) });
    let content = m[2];
    let citeNum = null;
    const citeMatch = content.match(/^\[(\d+)\]\s*\n?/);
    if (citeMatch && parseInt(citeMatch[1], 10) >= 1 && parseInt(citeMatch[1], 10) <= results.length) {
      citeNum = parseInt(citeMatch[1], 10);
      content = content.slice(citeMatch[0].length);
    }
    parts.push({ type: "code", lang: (m[1] || "plaintext").toLowerCase(), content, citeNum });
    lastIndex = m.index + m[0].length;
  }
  parts.push({ type: "text", content: text.slice(lastIndex) });

  let body = parts
    .map((p) => {
      if (p.type === "text") {
        let html = escapeHtml(p.content).replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (match, inner) => {
          if (inner.length > 60) return match;
          const nums = inner.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n) && n >= 1 && n <= results.length);
          if (nums.length === 0) return match;
          return nums.map((idx) => `<a href="#cite-${idx}" class="cite-link" data-cite="${idx}">[${idx}]</a>`).join(", ");
        });
        html = html.replace(/`([^`\n<]+)`/g, (match, code, offset) => {
          const after = html.slice(offset + match.length);
          const nextCite = after.match(/\[(\d+)\]/);
          const citeNum = nextCite && parseInt(nextCite[1], 10) >= 1 && parseInt(nextCite[1], 10) <= results.length ? parseInt(nextCite[1], 10) : null;
          const citeAttr = citeNum ? ` data-cite="${citeNum}"` : "";
          return `<code class="reprompt-token"${citeAttr}>${escapeHtml(code.trim())}</code>`;
        });
        return html;
      }
      const chunk = p.citeNum ? results[p.citeNum - 1] : null;
      const path = chunk?.file_path || "";
      const pathDisplay = path || "(source)";
      const langClass = chunk ? inferLanguage(path, chunk.language) : (p.lang === "cobol" || p.lang === "cbl" ? "cobol" : (p.lang && p.lang !== "plaintext" ? p.lang : "bash"));
      const startLine = chunk?.start_line ?? 0;
      const endLine = chunk?.end_line ?? 0;
      let html = `<div class="answer-code-block" ${p.citeNum ? `id="cite-${p.citeNum}" data-cite="${p.citeNum}" data-original-start="${startLine}" data-original-end="${endLine}"` : ""}>`;
      if (p.citeNum) {
        html += `<a href="#cite-${p.citeNum}" class="code-block-cite-link">[${p.citeNum}] ${escapeHtml(pathDisplay)}</a>`;
      }
      const originalContent = (p.content || "").trim();
      const originalEscaped = originalContent
        .replace(/\\/g, "\\\\")
        .replace(/\$/g, "\\$")
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      html += `<div class="code-container code-lang-${langClass}">`;
      if (p.citeNum && chunk) {
        html += `<button type="button" class="collapse-btn" data-lang="${langClass}" title="Collapse to original chunk" style="display:none">↺</button>`;
      }
      if (p.citeNum && chunk && startLine > 1) {
        html += `<button type="button" class="expand-btn expand-top answer-show-more" data-path="${escapeHtml(path)}" data-start-line="${startLine}" data-end-line="${endLine}" data-direction="up" data-lang="${langClass}" data-cite="${p.citeNum}" title="Load more lines above">↟</button>`;
      }
      html += `<pre><code class="language-${langClass}" data-original-content="${originalEscaped}">${escapeHtml(originalContent)}</code></pre>`;
      if (p.citeNum && chunk) {
        html += `<button type="button" class="expand-btn expand-bottom answer-show-more" data-path="${escapeHtml(path)}" data-start-line="${startLine}" data-end-line="${endLine}" data-direction="down" data-lang="${langClass}" data-cite="${p.citeNum}" title="Load more lines below">↡</button>`;
      }
      html += `</div>`;
      html += "</div>";
      return html;
    })
    .join("");

  if (results.length > 0) {
    body += '<div class="cite-sources"><details open><summary>Sources</summary>';
    results.forEach((chunk, i) => {
      const n = i + 1;
      const path = chunk?.file_path || "?";
      const startLine = chunk?.start_line ?? 0;
      const endLine = chunk?.end_line ?? 0;
      const rrfScore = chunk?.score != null ? chunk.score.toFixed(4) : "?";
      const vecScore = chunk?.vector_score != null ? chunk.vector_score.toFixed(4) : null;
      const scoreStr = vecScore != null
        ? `RRF=${rrfScore} · cosine=${vecScore}`
        : `RRF=${rrfScore}`;
      const langClass = inferLanguage(path, chunk?.language);
      const snippet = escapeHtml((chunk?.code_snippet || "").trim());
      const pathAttr = path && path !== "?" ? ` data-path="${escapeHtml(path)}" data-start-line="${startLine}" data-end-line="${endLine}"` : "";
      body += `<div id="cite-source-${n}" class="cite-source-block" data-cite="${n}"${pathAttr}><span class="cite-source-label"><span class="cite-source-path">[${n}] ${path}</span><span class="cite-source-score" title="RRF=rank fusion score (higher=better). cosine=vector similarity 0-1.">${scoreStr}</span></span><div class="cite-source-code-container">`;
      if (path && path !== "?" && startLine > 1) {
        body += `<button type="button" class="expand-btn expand-top answer-show-more" data-path="${escapeHtml(path)}" data-start-line="${startLine}" data-end-line="${endLine}" data-direction="up" data-lang="${langClass}" data-cite="${n}" title="Load more lines above">↟</button>`;
      }
      body += `<pre><code class="language-${langClass}">${snippet}</code></pre>`;
      if (path && path !== "?") {
        body += `<button type="button" class="expand-btn expand-bottom answer-show-more" data-path="${escapeHtml(path)}" data-start-line="${startLine}" data-end-line="${endLine}" data-direction="down" data-lang="${langClass}" data-cite="${n}" title="Load more lines below">↡</button>`;
      }
      body += `</div></div>`;
    });
    body += "</details></div>";
  }
  return body;
}

if (el("login-btn")) el("login-btn").addEventListener("click", login);
if (el("refresh-count-btn")) el("refresh-count-btn").addEventListener("click", refreshCount);
if (el("load-corpus-btn")) el("load-corpus-btn").addEventListener("click", loadCorpus);
const browseSource = el("browse-source");
if (browseSource) browseSource.addEventListener("change", () => { refreshCount(); loadCorpus(); });
if (el("reset-btn")) el("reset-btn").addEventListener("click", resetDb);
if (el("reingest-btn")) el("reingest-btn").addEventListener("click", reingest);
async function runChat(chunksOnly) {
  const query = el("query-text").value.trim();
  if (!query) {
    el("query-status").textContent = "Enter a query.";
    return;
  }
  const status = el("query-status");
  const answerEl = el("chat-answer");
  status.textContent = (Array.isArray(chunksOnly) && chunksOnly.length > 0) ? "Chatting (this chunk only)…" : "Chatting...";
  answerEl.style.display = "none";
  try {
    const body = Array.isArray(chunksOnly) && chunksOnly.length > 0
      ? {
          query,
          chunks: chunksOnly.map((c) => ({
            id: c.id ?? "",
            score: c.score ?? 0,
            vector_score: c.vector_score ?? null,
            file_path: c.file_path ?? "",
            start_line: c.start_line ?? 0,
            end_line: c.end_line ?? 0,
            division: c.division ?? null,
            section_name: c.section_name ?? null,
            paragraph_name: c.paragraph_name ?? null,
            code_snippet: c.code_snippet ?? "",
            language: c.language ?? "COBOL",
            source_type: c.source_type ?? "code",
          })),
        }
      : { query };
    const res = await fetch("/query/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: (text && text.slice(0, 200)) || "Invalid response" };
    }
    if (!res.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map((e) => e.msg || JSON.stringify(e)).join("; ") : (data.detail || "Chat failed.");
      status.textContent = detail;
      return;
    }
    lastQuery = query;
    lastQueryResults = data.results || [];
    status.textContent = `Answered. ${(data.results || []).length} chunks used.`;
    savePromptHistory(query, data.answer, data.results);
    renderPromptHistory();
    displayCachedAnswer(data.answer, data.results);

  } catch (err) {
    status.textContent = "Network error: " + (err?.message || "Could not reach server.");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadStorageInfo();
  renderPromptHistory();
  const historyList = el("prompt-history-list");
  if (historyList) historyList.addEventListener("click", onPromptHistoryClick);
  const answerEl = el("chat-answer");
  if (answerEl) answerEl.addEventListener("click", onAnswerClick);

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const panel = document.getElementById(`tab-${tab}`);
      if (panel) panel.classList.add("active");
      if (tab === "logs") loadLogs();
    });
  });

  const refreshLogsBtn = el("refresh-logs-btn");
  if (refreshLogsBtn) refreshLogsBtn.addEventListener("click", loadLogs);
  const logsAutoRefresh = el("logs-auto-refresh");
  if (logsAutoRefresh) {
    logsAutoRefresh.addEventListener("change", function () {
      if (logsRefreshInterval) clearInterval(logsRefreshInterval);
      logsRefreshInterval = this.checked ? setInterval(loadLogs, 5000) : null;
    });
  }
});

el("chat-btn").addEventListener("click", () => runChat());

document.querySelectorAll(".prompt-suggestion-bubble").forEach((btn) => {
  btn.addEventListener("click", () => {
    const textarea = el("query-text");
    if (!textarea) return;
    const insert = btn.getAttribute("data-insert") || "";
    if (textarea.value.trim()) {
      textarea.value = textarea.value.trimEnd() + " " + insert;
    } else {
      textarea.value = insert;
    }
    textarea.focus();
  });
});
