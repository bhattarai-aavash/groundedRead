const EXAMPLES: string[] = [
  "How does Raft elect a leader?",
  "What is TrueTime in Spanner?",
  "What is a hybrid logical clock, and how does it combine NTP time with causality?",
  "What is a conflict-free replicated data type (CRDT)?",
  "What trade-off does the CAP theorem state among consistency, availability, and partition tolerance?",
];

interface StatsConfig {
  llm_provider?: string;
  llm_model?: string;
  top_k?: number;
  grounding_check?: boolean;
  use_bm25?: boolean;
  use_rerank?: boolean;
  contextual?: boolean;
  extractor?: string;
}

interface StatsResponse {
  chunks: number;
  config?: StatsConfig;
}

interface SourceRef {
  source: string;
  page: number;
}

interface Excerpt {
  source?: string;
  page?: number;
  text?: string;
  distance?: number | null;
  rerank?: number | null;
}

interface Span {
  node?: string;
  ms?: number;
  tokens_in?: number;
  tokens_out?: number;
  cost_usd?: number;
}

interface AskResponse {
  answer?: string;
  grounded?: boolean | null;
  reason?: string;
  attempts?: number;
  abstained?: boolean;
  sources?: SourceRef[];
  docs?: Excerpt[];
  trace?: string[];
  spans?: Span[];
  latency_ms?: number;
  cost_usd?: number;
  error?: string;
}

interface StatusEvent {
  stage?: string;
  n_docs?: number;
}

interface TokenEvent {
  text?: string;
  replace?: boolean;
}

type SseHandler = (event: string, data: AskResponse | StatusEvent | TokenEvent) => void;

function requireEl<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`missing #${id}`);
  }
  return el as T;
}

const ESCAPE: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (ch) => ESCAPE[ch] ?? ch);
}

function renderAnswer(text: string): string {
  return escapeHtml(text).replace(
    /\[([^\]\s]+\.pdf)\s+p\.(\d+)\]/g,
    '<span class="cite">[$1 p.$2]</span>',
  );
}

function renderSpans(spans: Span[] | undefined, latencyMs?: number, costUsd?: number): string {
  if (!spans?.length) {
    return "";
  }
  const rows = spans
    .map(
      (span) => `
        <tr>
          <td>${escapeHtml(span.node ?? "")}</td>
          <td>${Number(span.ms ?? 0).toFixed(0)}ms</td>
          <td>${span.tokens_in ?? 0}</td>
          <td>${span.tokens_out ?? 0}</td>
          <td>$${Number(span.cost_usd ?? 0).toFixed(4)}</td>
        </tr>`,
    )
    .join("");
  const total = `total ${Number(latencyMs ?? 0).toFixed(0)}ms · $${Number(costUsd ?? 0).toFixed(4)}`;
  return `
    <h2>Node cost <span class="plain">${escapeHtml(total)}</span></h2>
    <table class="spans">
      <thead><tr><th>node</th><th>latency</th><th>in</th><th>out</th><th>est. $</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderResult(data: AskResponse, liveAnswer?: string): string {
  const grounded = data.grounded;
  const badgeClass = data.abstained ? "warn" : grounded ? "ok" : "warn";
  const badgeText = data.abstained
    ? "abstained"
    : grounded === true
      ? "grounded"
      : grounded === false
        ? "ungrounded"
        : "unchecked";
  const sources =
    (data.sources ?? [])
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

async function readSSE(response: Response, onEvent: SseHandler): Promise<void> {
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
      const dataLines: string[] = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) {
          event = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        }
      }
      if (dataLines.length) {
        onEvent(event, JSON.parse(dataLines.join("\n")) as AskResponse);
        if (event === "done" || event === "error") {
          await reader.cancel().catch(() => undefined);
          return;
        }
      }
      idx = buf.indexOf("\n\n");
    }
  }
}

function stageLabel(stage: string): string {
  if (stage === "supervisor") return "supervisor is choosing a specialist…";
  if (stage === "retriever") return "retriever is writing a search query…";
  if (stage === "retrieve") return "searching the paper index…";
  if (stage === "writer") return "writer is drafting…";
  if (stage === "generate") return "writer is drafting…";
  if (stage === "critic" || stage === "check") return "critic is checking grounding…";
  if (stage === "retry") return "critic rejected the draft; supervisor is retrying…";
  if (stage === "guard") return "output guard is abstaining on an ungrounded draft…";
  if (stage === "abstain") return "supervisor is abstaining…";
  return stage;
}

async function loadStats(meta: HTMLElement, status: HTMLElement): Promise<void> {
  try {
    const response = await fetch("/api/stats");
    const stats = (await response.json()) as StatsResponse;
    const config = stats.config ?? {};
    const retrieval = [
      config.use_bm25 ? "hybrid" : "dense",
      config.use_rerank ? "rerank" : null,
      config.contextual ? "contextual" : null,
      config.extractor,
    ]
      .filter((part): part is string => Boolean(part))
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
  } catch {
    meta.textContent = "could not load stats";
  }
}

function bindExamples(chips: HTMLElement, question: HTMLTextAreaElement): void {
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

async function askQuestion(
  question: string,
  go: HTMLButtonElement,
  status: HTMLElement,
  out: HTMLElement,
): Promise<void> {
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
      const payload = (await response.json().catch(() => ({}))) as AskResponse;
      throw new Error(payload.error || response.statusText);
    }
    if (!ctype.includes("text/event-stream")) {
      const data = (await response.json()) as AskResponse;
      if (!response.ok) {
        throw new Error(data.error || response.statusText);
      }
      out.innerHTML = renderResult(data);
      status.textContent = data.reason ? data.reason : "done";
      return;
    }
    await readSSE(response, (event, data) => {
      if (event === "status") {
        status.textContent = stageLabel((data as StatusEvent).stage ?? "");
      } else if (event === "token") {
        const token = data as TokenEvent;
        if (token.replace) {
          answer = token.text ?? "";
        } else {
          answer += token.text ?? "";
        }
        const el = out.querySelector(".answer");
        if (el) {
          el.innerHTML = renderAnswer(answer);
        }
      } else if (event === "done") {
        const done = data as AskResponse;
        out.innerHTML = renderResult(done, done.answer || answer);
        status.textContent = done.reason ? done.reason : "done";
      } else if (event === "error") {
        throw new Error((data as AskResponse).error || "stream error");
      }
    });
  } catch (err) {
    status.className = "status error";
    status.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    go.disabled = false;
  }
}

function main(): void {
  const chips = requireEl<HTMLElement>("chips");
  const question = requireEl<HTMLTextAreaElement>("q");
  const form = requireEl<HTMLFormElement>("form");
  const go = requireEl<HTMLButtonElement>("go");
  const status = requireEl<HTMLElement>("status");
  const out = requireEl<HTMLElement>("out");
  const meta = requireEl<HTMLElement>("meta");

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
