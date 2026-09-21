"use strict";
const EXAMPLES = [
    "How does Raft elect a leader?",
    "What is TrueTime in Spanner?",
    "What is a hybrid logical clock, and how does it combine NTP time with causality?",
    "What is a conflict-free replicated data type (CRDT)?",
    "What trade-off does the CAP theorem state among consistency, availability, and partition tolerance?",
];
function requireEl(id) {
    const el = document.getElementById(id);
    if (!el) {
        throw new Error(`missing #${id}`);
    }
    return el;
}
const ESCAPE = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
};
function escapeHtml(value) {
    return value.replace(/[&<>"']/g, (ch) => ESCAPE[ch] ?? ch);
}
function renderAnswer(text) {
    return escapeHtml(text).replace(/\[([^\]\s]+\.pdf)\s+p\.(\d+)\]/g, '<span class="cite">[$1 p.$2]</span>');
}
function renderSpans(spans, latencyMs, costUsd) {
    if (!spans?.length) {
        return "";
    }
    const rows = spans
        .map((span) => `
        <tr>
          <td>${escapeHtml(span.node ?? "")}</td>
          <td>${Number(span.ms ?? 0).toFixed(0)}ms</td>
          <td>${span.tokens_in ?? 0}</td>
          <td>${span.tokens_out ?? 0}</td>
          <td>$${Number(span.cost_usd ?? 0).toFixed(4)}</td>
        </tr>`)
        .join("");
    const total = `total ${Number(latencyMs ?? 0).toFixed(0)}ms · $${Number(costUsd ?? 0).toFixed(4)}`;
    return `
    <h2>Node cost <span class="plain">${escapeHtml(total)}</span></h2>
    <table class="spans">
      <thead><tr><th>node</th><th>latency</th><th>in</th><th>out</th><th>est. $</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}
function renderResult(data, liveAnswer) {
    const grounded = data.grounded;
    const badgeClass = data.abstained ? "warn" : grounded ? "ok" : "warn";
    const badgeText = data.abstained
        ? "abstained"
        : grounded === true
            ? "grounded"
            : grounded === false
                ? "ungrounded"
                : "unchecked";
    const sources = (data.sources ?? [])
        .map((src) => `<li>${escapeHtml(src.source)} p.${src.page}</li>`)
        .join("") || "<li>none</li>";
    const excerpts = (data.docs ?? [])
        .map((doc) => {
        const body = escapeHtml((doc.text ?? "").slice(0, 900));
        return `<details><summary>${escapeHtml(String(doc.source ?? ""))} p.${doc.page ?? "?"}</summary><div class="excerpt">${body}</div></details>`;
    })
        .join("");
    const trace = (data.trace ?? []).map((line) => escapeHtml(line)).join("\n");
    const answer = liveAnswer !== undefined ? liveAnswer : (data.answer ?? "");
    return `
    <div>
      <span class="badge ${badgeClass}">${badgeText}</span>
      <span class="badge">attempts ${data.attempts ?? "?"}</span>
    </div>
    <div class="answer">${renderAnswer(answer)}</div>
    <h2>Sources</h2>
    <ul>${sources}</ul>
    <h2>Excerpts</h2>
    ${excerpts || "<p class='status'>none</p>"}
    ${renderSpans(data.spans, data.latency_ms, data.cost_usd)}
    <h2>Trace</h2>
    <pre class="trace">${trace || "(none)"}</pre>
  `;
}
async function readSSE(response, onEvent) {
    if (!response.body) {
        throw new Error("no response body");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
        const { value, done } = await reader.read();
        if (done) {
            break;
        }
        buf += decoder.decode(value, { stream: true });
        let idx = buf.indexOf("\n\n");
        while (idx !== -1) {
            const raw = buf.slice(0, idx);
            buf = buf.slice(idx + 2);
            let event = "message";
            const dataLines = [];
            for (const line of raw.split("\n")) {
                if (line.startsWith("event:")) {
                    event = line.slice(6).trim();
                }
                else if (line.startsWith("data:")) {
                    dataLines.push(line.slice(5).trim());
                }
            }
            if (dataLines.length) {
                onEvent(event, JSON.parse(dataLines.join("\n")));
                if (event === "done" || event === "error") {
                    await reader.cancel().catch(() => undefined);
                    return;
                }
            }
            idx = buf.indexOf("\n\n");
        }
    }
}
function stageLabel(stage) {
    if (stage === "supervisor")
        return "supervisor is choosing a specialist…";
    if (stage === "retriever")
        return "retriever is writing a search query…";
    if (stage === "retrieve")
        return "searching the paper index…";
    if (stage === "writer")
        return "writer is drafting…";
    if (stage === "generate")
        return "writer is drafting…";
    if (stage === "critic" || stage === "check")
        return "critic is checking grounding…";
    if (stage === "retry")
        return "critic rejected the draft; supervisor is retrying…";
    if (stage === "guard")
        return "output guard is abstaining on an ungrounded draft…";
    if (stage === "abstain")
        return "supervisor is abstaining…";
    return stage;
}
async function loadStats(meta, status) {
    try {
        const response = await fetch("/api/stats");
        const stats = (await response.json());
        const config = stats.config ?? {};
        const retrieval = [
            config.use_bm25 ? "hybrid" : "dense",
            config.use_rerank ? "rerank" : null,
            config.contextual ? "contextual" : null,
            config.extractor,
        ]
            .filter((part) => Boolean(part))
            .join(" + ");
        meta.innerHTML = [
            `${stats.chunks} chunks`,
            config.llm_provider,
            config.llm_model,
            `top_k=${config.top_k}`,
            retrieval,
            config.grounding_check ? "grounding on" : "grounding off",
        ]
            .map((item) => `<span>${escapeHtml(String(item ?? ""))}</span>`)
            .join("");
        if (!stats.chunks) {
            status.textContent = "Index is empty. Run: dsqa ingest data/papers";
        }
    }
    catch {
        meta.textContent = "could not load stats";
    }
}
function bindExamples(chips, question) {
    for (const text of EXAMPLES) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = text;
        button.addEventListener("click", () => {
            question.value = text;
            question.focus();
        });
        chips.appendChild(button);
    }
}
async function askQuestion(question, go, status, out) {
    go.disabled = true;
    status.className = "status";
    status.textContent = "retrieving…";
    out.classList.remove("hidden");
    out.innerHTML = renderResult({ attempts: 0, sources: [], docs: [], trace: [], spans: [] }, "");
    let answer = "";
    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "text/event-stream",
            },
            body: JSON.stringify({ question, stream: true }),
        });
        const ctype = response.headers.get("content-type") ?? "";
        if (!response.ok && !ctype.includes("text/event-stream")) {
            const payload = (await response.json().catch(() => ({})));
            throw new Error(payload.error || response.statusText);
        }
        if (!ctype.includes("text/event-stream")) {
            const data = (await response.json());
            if (!response.ok) {
                throw new Error(data.error || response.statusText);
            }
            out.innerHTML = renderResult(data);
            status.textContent = data.reason ? data.reason : "done";
            return;
        }
        await readSSE(response, (event, data) => {
            if (event === "status") {
                status.textContent = stageLabel(data.stage ?? "");
            }
            else if (event === "token") {
                const token = data;
                if (token.replace) {
                    answer = token.text ?? "";
                }
                else {
                    answer += token.text ?? "";
                }
                const el = out.querySelector(".answer");
                if (el) {
                    el.innerHTML = renderAnswer(answer);
                }
            }
            else if (event === "done") {
                const done = data;
                out.innerHTML = renderResult(done, done.answer || answer);
                status.textContent = done.reason ? done.reason : "done";
            }
            else if (event === "error") {
                throw new Error(data.error || "stream error");
            }
        });
    }
    catch (err) {
        status.className = "status error";
        status.textContent = err instanceof Error ? err.message : String(err);
    }
    finally {
        go.disabled = false;
    }
}
function main() {
    const chips = requireEl("chips");
    const question = requireEl("q");
    const form = requireEl("form");
    const go = requireEl("go");
    const status = requireEl("status");
    const out = requireEl("out");
    const meta = requireEl("meta");
    bindExamples(chips, question);
    form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const text = question.value.trim();
        if (text) {
            void askQuestion(text, go, status, out);
        }
    });
    void loadStats(meta, status);
}
main();
