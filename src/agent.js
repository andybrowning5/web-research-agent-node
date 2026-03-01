import Anthropic from "@anthropic-ai/sdk";
import { createInterface } from "readline";

const BRAVE_API_KEY = process.env.BRAVE_API_KEY || "";
const BRAVE_BASE_URL = process.env.BRAVE_BASE_URL || "https://api.search.brave.com";
const BRAVE_SEARCH_URL = `${BRAVE_BASE_URL}/res/v1/web/search`;
const MODEL = process.env.ANTHROPIC_MODEL || "claude-sonnet-4-5-20250929";

const anthropic = new Anthropic();

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + "\n");
}

function log(text) {
  process.stderr.write(text + "\n");
}

// --- Brave Search ---

async function webSearch(query) {
  const url = `${BRAVE_SEARCH_URL}?${new URLSearchParams({ q: query, count: "10" })}`;
  const resp = await fetch(url, {
    headers: {
      Accept: "application/json",
      "X-Subscription-Token": BRAVE_API_KEY,
    },
    signal: AbortSignal.timeout(15000),
  });
  if (!resp.ok) throw new Error(`Brave API ${resp.status}: ${resp.statusText}`);
  const data = await resp.json();
  const results = (data.web?.results || []).map(
    (r) => `Title: ${r.title}\nURL: ${r.url}\nDescription: ${r.description || ""}`
  );
  log(`Brave API returned ${results.length} results for: ${query}`);
  return results.length ? results.join("\n\n---\n\n") : "No results found. Try a different search query.";
}

const tools = [
  {
    name: "web_search",
    description:
      "Search the web for real-time information. Use this to find current facts, news, documentation, or any topic the user asks about. You can call this multiple times with different queries to get broader coverage.",
    input_schema: {
      type: "object",
      properties: { query: { type: "string", description: "Search query" } },
      required: ["query"],
    },
  },
];

const toolHandlers = { web_search: ({ query }) => webSearch(query) };

// --- Agentic loop ---

async function research(query, messageId) {
  const today = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const systemPrompt =
    `You are Web Research Agent, an expert research assistant. Today is ${today}. ` +
    "Your job is to thoroughly research the user's question using web search. " +
    "Strategy:\n" +
    "1. Break complex questions into sub-queries and search for each\n" +
    "2. Search multiple times with different angles to get comprehensive coverage\n" +
    "3. For simple greetings or non-research questions, just respond naturally without searching\n" +
    "4. Synthesize all findings into a clear, well-structured briefing with markdown\n" +
    "5. Include inline citations [1], [2] etc. and end with a Sources section\n" +
    "Be thorough but concise. Prioritize accuracy and recency.";

  const messages = [{ role: "user", content: query }];

  while (true) {
    const resp = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 4096,
      system: systemPrompt,
      tools,
      messages,
    });

    // Emit activity for each tool use
    for (const block of resp.content) {
      if (block.type === "tool_use") {
        const q = block.input?.query || "";
        send({
          type: "activity",
          tool: block.name,
          description: q ? `${block.name}(${q})` : block.name,
          message_id: messageId,
        });
      }
    }

    // If no tool use, extract final text and return
    if (resp.stop_reason === "end_turn" || !resp.content.some((b) => b.type === "tool_use")) {
      const textBlocks = resp.content.filter((b) => b.type === "text");
      return textBlocks.map((b) => b.text).join("") || "";
    }

    // Process tool calls
    messages.push({ role: "assistant", content: resp.content });

    const toolResults = [];
    for (const block of resp.content) {
      if (block.type !== "tool_use") continue;
      let result;
      try {
        result = await toolHandlers[block.name](block.input);
      } catch (e) {
        result = `Error: ${e.message}`;
      }
      toolResults.push({ type: "tool_result", tool_use_id: block.id, content: result });
    }
    messages.push({ role: "user", content: toolResults });
  }
}

// --- Primordial Protocol ---

function main() {
  send({ type: "ready" });
  log("Web Research Agent (Node.js) ready");

  const rl = createInterface({ input: process.stdin, terminal: false });

  rl.on("line", async (line) => {
    line = line.trim();
    if (!line) return;

    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }

    if (msg.type === "shutdown") {
      log("Shutting down");
      rl.close();
      return;
    }

    if (msg.type === "message") {
      const mid = msg.message_id;
      try {
        const result = await research(msg.content, mid);
        send({ type: "response", content: result, message_id: mid, done: true });
      } catch (e) {
        log(`Error: ${e.message}`);
        send({ type: "error", error: e.message, message_id: mid });
        send({ type: "response", content: `Something went wrong: ${e.message}`, message_id: mid, done: true });
      }
    }
  });
}

main();
