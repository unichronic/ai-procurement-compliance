// Served same-origin under /ui/ by the FastAPI backend (see main.py's
// StaticFiles mount), so API calls below use relative paths -- no CORS
// config or base-URL setting needed even when this moves environments.

const textEl = document.getElementById("query-text");
const topKEl = document.getElementById("top-k");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const modeSingleBtn = document.getElementById("mode-single");
const modeBatchBtn = document.getElementById("mode-batch");
const modeLintBtn = document.getElementById("mode-lint");
const sampleBtn = document.getElementById("sample-btn");
const topKLabel = document.getElementById("top-k-label");

const SAMPLE_DRAFT = `1. Supply of hot rolled structural steel sections conforming to IS 2062:2006.
2. Packaged natural mineral water bottles, 1 litre, as per IS 14543:2004.
3. Supply of protective helmets for two-wheeler riders, 200 nos.
4. Toys for anganwadi centres conforming to IS 9873 (Part 4).
5. Supply of XLPE insulated power cable, 1.1 kV grade, 3 core.
6. Cement concrete works as per IS 99999.`;

const PLACEHOLDERS = {
  single: "e.g. supply of galvanized structural steel sections for building frame, grade E250",
  batch: "Paste a full tender document, one item per line, e.g.:\n1. Structural steel sections for building frame\n2. Gold jewellery fineness marking\n3. LPG cylinder valve assembly",
  lint: "Paste a DRAFT tender spec and it will be audited for superseded citations, omitted normative references, and missing certification requirements.",
};

let mode = "single";

modeSingleBtn.addEventListener("click", () => setMode("single"));
modeBatchBtn.addEventListener("click", () => setMode("batch"));
modeLintBtn.addEventListener("click", () => setMode("lint"));
sampleBtn.addEventListener("click", () => {
  setMode("lint");
  textEl.value = SAMPLE_DRAFT;
});

function setMode(next) {
  mode = next;
  modeSingleBtn.classList.toggle("active", mode === "single");
  modeBatchBtn.classList.toggle("active", mode === "batch");
  modeLintBtn.classList.toggle("active", mode === "lint");
  textEl.placeholder = PLACEHOLDERS[mode];

  const isLint = mode === "lint";
  topKEl.style.display = isLint ? "none" : "";
  topKLabel.style.display = isLint ? "none" : "";
  submitBtn.textContent = isLint ? "Audit this draft" : "Find applicable standards";
}

submitBtn.addEventListener("click", runSearch);
textEl.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runSearch();
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

async function runSearch() {
  const text = textEl.value.trim();
  if (!text) {
    setStatus("Enter a product description, spec, or tender document first.", true);
    return;
  }
  const topK = Math.max(1, Math.min(15, parseInt(topKEl.value, 10) || 5));

  submitBtn.disabled = true;
  resultsEl.innerHTML = "";
  setStatus({
    single: "Searching locally…",
    batch: "Splitting document and searching locally…",
    lint: "Auditing draft against the standards corpus…",
  }[mode]);

  try {
    if (mode === "single") {
      const data = await postJson("/recommend", { text, top_k: topK });
      renderSingle(data);
    } else if (mode === "batch") {
      const data = await postJson("/batch", { document_text: text, top_k_per_item: topK });
      renderBatch(data);
    } else {
      const data = await postJson("/lint", { document_text: text });
      renderLint(data, text);
    }
    setStatus("");
  } catch (err) {
    setStatus(err.message || "Something went wrong talking to the backend.", true);
  } finally {
    submitBtn.disabled = false;
  }
}

async function postJson(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const errBody = await resp.json();
      detail = errBody.detail || detail;
    } catch (_) {}
    throw new Error(`${resp.status}: ${detail}`);
  }
  return resp.json();
}

function renderSingle(data) {
  if (!data.recommendations.length) {
    resultsEl.innerHTML = `<p class="empty-state">No matching standards found.</p>`;
    return;
  }
  data.recommendations.forEach((rec) => resultsEl.appendChild(buildCard(rec)));
}

function renderBatch(data) {
  if (!data.items.length) {
    resultsEl.innerHTML = `<p class="empty-state">No line items could be extracted from that document.</p>`;
    return;
  }
  data.items.forEach((item) => {
    const heading = document.createElement("div");
    heading.className = "batch-item-heading";
    heading.textContent = `Item ${item.item_index + 1}: ${item.line}`;
    resultsEl.appendChild(heading);

    if (!item.recommendations.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "No matching standards found for this line.";
      resultsEl.appendChild(empty);
      return;
    }
    item.recommendations.forEach((rec) => resultsEl.appendChild(buildCard(rec)));
  });
}

const RULE_LABELS = {
  superseded_citation: "Superseded citation",
  missing_allied_standard: "Omitted normative reference",
  missing_certification_requirement: "Missing certification requirement",
  uncited_item: "Item cites no standard",
  unresolved_citation: "Unrecognised citation",
};

function renderLint(data, sourceText) {
  const summary = document.createElement("div");
  summary.className = "lint-summary";

  if (!data.findings.length) {
    summary.innerHTML = `<b class="clean">No defects found.</b> ${data.citations_found} citation(s) checked, all current and complete.`;
    resultsEl.appendChild(summary);
    return;
  }

  summary.innerHTML = `
    <b>${data.summary.total} finding${data.summary.total === 1 ? "" : "s"}</b>
    — <span class="sev-high">${data.summary.high} high</span>,
    <span class="sev-medium">${data.summary.medium} medium</span>.
    ${data.citations_found} citation(s) found, ${data.citations_resolved} resolved.
  `;
  resultsEl.appendChild(summary);

  data.findings.forEach((f) => {
    const card = document.createElement("article");
    card.className = `card finding sev-border-${f.severity}`;

    const head = document.createElement("div");
    head.className = "finding-head";
    const label = document.createElement("span");
    label.className = `badge ${f.severity === "high" ? "low" : "medium"}`;
    label.textContent = f.severity;
    const title = document.createElement("h3");
    title.textContent = RULE_LABELS[f.rule_id] || f.rule_id;
    head.appendChild(title);
    head.appendChild(label);
    card.appendChild(head);

    if (f.span && sourceText) {
      const quote = document.createElement("div");
      quote.className = "finding-quote";
      const before = sourceText.slice(Math.max(0, f.span[0] - 45), f.span[0]);
      const hit = sourceText.slice(f.span[0], f.span[1]);
      const after = sourceText.slice(f.span[1], f.span[1] + 45);
      quote.appendChild(document.createTextNode("…" + before));
      const mark = document.createElement("mark");
      mark.textContent = hit;
      quote.appendChild(mark);
      quote.appendChild(document.createTextNode(after + "…"));
      card.appendChild(quote);
    }

    const msg = document.createElement("p");
    msg.className = "scope";
    msg.textContent = f.message;
    card.appendChild(msg);

    if (f.suggested_fix) {
      const fix = document.createElement("div");
      fix.className = "finding-fix";
      fix.textContent = f.suggested_fix;
      card.appendChild(fix);
    }

    if (f.authority) {
      const auth = document.createElement("details");
      const sum = document.createElement("summary");
      sum.textContent = "Basis for this finding";
      const body = document.createElement("div");
      body.className = "allied-group";
      body.textContent = f.authority;
      auth.appendChild(sum);
      auth.appendChild(body);
      card.appendChild(auth);
    }

    resultsEl.appendChild(card);
  });
}

function buildCard(rec) {
  const card = document.createElement("article");
  card.className = "card";

  const band = rec.match.confidence_band;
  const simPct = (rec.match.semantic_similarity * 100).toFixed(1);

  card.innerHTML = `
    <div class="card-top">
      <div>
        <h3>${escapeHtml(rec.number)} ${rec.edition_year ? `(${rec.edition_year})` : ""} — ${escapeHtml(rec.title)}</h3>
        <div class="dept">${escapeHtml(rec.department)} · Committee ${escapeHtml(rec.committee)}</div>
      </div>
      <span class="badge ${band}">${band} confidence</span>
    </div>
    <p class="scope">${escapeHtml(rec.scope)}</p>
    <div class="meta-row">
      <span class="pill">semantic similarity ${simPct}%</span>
      <span class="pill">semantic rank #${rec.match.semantic_rank}</span>
      <span class="pill">lexical rank ${rec.match.lexical_rank !== null ? "#" + rec.match.lexical_rank : "n/a"}</span>
      <span class="pill">status: ${escapeHtml(rec.version.status)}</span>
    </div>
  `;

  if (rec.version.warning) {
    const warn = document.createElement("div");
    warn.className = "warning";
    warn.textContent = `⚠ ${rec.version.warning}`;
    card.appendChild(warn);
  }

  const cert = document.createElement("div");
  cert.className = "cert " + (rec.certification.mandatory_certification ? "mandatory" : "none");
  cert.textContent = rec.certification.message;
  card.appendChild(cert);

  card.appendChild(buildAlliedDetails(rec.allied_standards));
  card.appendChild(buildExplainRow(rec));

  return card;
}

const ALLIED_LABELS = {
  normative_reference: "Normative reference",
  test_method: "Test method",
  terminology: "Terminology",
  safety: "Safety",
  installation: "Installation",
  related_product: "Related product",
};

function buildAlliedDetails(allied) {
  const details = document.createElement("details");
  const nonEmpty = Object.entries(allied).filter(([, items]) => items.length > 0);

  const summary = document.createElement("summary");
  summary.textContent = nonEmpty.length
    ? `Allied standards (${nonEmpty.reduce((n, [, items]) => n + items.length, 0)})`
    : "Allied standards (none found)";
  details.appendChild(summary);

  nonEmpty.forEach(([type, items]) => {
    const group = document.createElement("div");
    group.className = "allied-group";
    const label = document.createElement("b");
    label.textContent = ALLIED_LABELS[type] || type;
    group.appendChild(label);

    const ul = document.createElement("ul");
    items.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = `${item.number} — ${item.title} (${item.status}${item.direction === "inverse" ? ", references this standard" : ""})`;
      ul.appendChild(li);
    });
    group.appendChild(ul);
    details.appendChild(group);
  });

  return details;
}

function buildExplainRow(rec) {
  const row = document.createElement("div");
  row.className = "explain-row";

  const btn = document.createElement("button");
  btn.className = "explain-btn";
  btn.textContent = "Why this match?";

  const out = document.createElement("div");

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Asking…";
    try {
      const data = await postJson("/explain", {
        standard_id: rec.id,
        confidence_band: rec.match.confidence_band,
        semantic_similarity: rec.match.semantic_similarity,
        lexical_rank: rec.match.lexical_rank,
        semantic_rank: rec.match.semantic_rank,
      });
      const box = document.createElement("div");
      box.className = "explanation";
      box.textContent = data.explanation;
      const tag = document.createElement("span");
      tag.className = "source-tag";
      tag.textContent = data.source === "groq" ? `Generated by ${data.model || "Groq"}` : "Generated locally (template fallback)";
      box.appendChild(tag);
      out.innerHTML = "";
      out.appendChild(box);
      btn.remove();
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "Why this match?";
      setStatus(`Explanation failed: ${err.message}`, true);
    }
  });

  row.appendChild(btn);
  row.appendChild(out);
  return row;
}

setMode("single");
