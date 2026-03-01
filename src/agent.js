import Anthropic from "@anthropic-ai/sdk";
import { createInterface } from "readline";
import { mkdirSync, existsSync, writeFileSync, appendFileSync, readFileSync } from "fs";
import { join } from "path";

const BRAVE_API_KEY = process.env.BRAVE_API_KEY || "";
const BRAVE_BASE_URL = process.env.BRAVE_BASE_URL || "https://api.search.brave.com";
const BRAVE_SEARCH_URL = `${BRAVE_BASE_URL}/res/v1/web/search`;
const MODEL = process.env.ANTHROPIC_MODEL || "claude-haiku-4-5-20251001";

const anthropic = new Anthropic();

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + "\n");
}

function log(text) {
  process.stderr.write(text + "\n");
}

// --- Conversational Memory ---

const MEMORY_DIR = "/home/user/data/memory";
const CONVERSATIONS_FILE = join(MEMORY_DIR, "conversations.jsonl");
const FACTS_FILE = join(MEMORY_DIR, "facts.json");

function initMemory() {
  mkdirSync(MEMORY_DIR, { recursive: true });
  if (!existsSync(CONVERSATIONS_FILE)) writeFileSync(CONVERSATIONS_FILE, "");
  if (!existsSync(FACTS_FILE)) writeFileSync(FACTS_FILE, JSON.stringify({ facts: [], entities: {} }));
}

function saveTurn(userMsg, assistantMsg) {
  const entry = { timestamp: new Date().toISOString(), user: userMsg, assistant: assistantMsg };
  appendFileSync(CONVERSATIONS_FILE, JSON.stringify(entry) + "\n");
}

function getRecentConversations(limit = 10) {
  try {
    const lines = readFileSync(CONVERSATIONS_FILE, "utf-8").trim().split("\n").filter(Boolean);
    return lines.slice(-limit).map((l) => JSON.parse(l));
  } catch { return []; }
}

function loadFacts() {
  try { return JSON.parse(readFileSync(FACTS_FILE, "utf-8")); } catch { return { facts: [], entities: {} }; }
}

function getContext() {
  const parts = [];
  const recent = getRecentConversations();
  if (recent.length) {
    parts.push("### Recent research sessions");
    for (const c of recent) {
      parts.push(`[${c.timestamp}] Query: ${c.user.slice(0, 150)}`);
      parts.push(`Result: ${c.assistant.slice(0, 200)}`);
    }
  }
  const facts = loadFacts();
  if (facts.facts?.length) {
    parts.push("\n### Known facts");
    for (const f of facts.facts) parts.push(`- ${f}`);
  }
  if (facts.entities && Object.keys(facts.entities).length) {
    parts.push("\n### Known entities");
    for (const [name, info] of Object.entries(facts.entities)) {
      parts.push(`- ${name}: ${typeof info === "string" ? info : JSON.stringify(info)}`);
    }
  }
  return parts.join("\n") || "";
}

function searchMemory(query) {
  const q = query.toLowerCase();
  const results = [];
  for (const c of getRecentConversations(50)) {
    if ((c.user + c.assistant).toLowerCase().includes(q)) {
      results.push(`[${c.timestamp}] Query: ${c.user.slice(0, 100)} → Result: ${c.assistant.slice(0, 100)}`);
    }
  }
  const facts = loadFacts();
  for (const f of facts.facts || []) {
    if (f.toLowerCase().includes(q)) results.push(`Fact: ${f}`);
  }
  for (const [name, info] of Object.entries(facts.entities || {})) {
    const infoStr = typeof info === "string" ? info : JSON.stringify(info);
    if ((name + infoStr).toLowerCase().includes(q)) results.push(`Entity: ${name} — ${infoStr}`);
  }
  return results.length ? results.join("\n") : `No results for "${query}"`;
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
  {
    name: "remember",
    description:
      "Search your memory for past research sessions, facts, and entities. Use this when the user references something from a previous session or asks what you know about a topic.",
    input_schema: {
      type: "object",
      properties: { query: { type: "string", description: "What to search for in memory" } },
      required: ["query"],
    },
  },
];

const toolHandlers = {
  web_search: ({ query }) => webSearch(query),
  remember: ({ query }) => searchMemory(query),
};

// --- Agentic loop ---

async function research(query, messageId) {
  const today = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const memoryContext = getContext();

  const systemPrompt =
    `You are Web Research Agent, an expert research assistant. Today is ${today}. ` +
    "Your job is to thoroughly research the user's question using web search. " +
    "Strategy:\n" +
    "1. Break complex questions into sub-queries and search for each\n" +
    "2. Search multiple times with different angles to get comprehensive coverage\n" +
    "3. For simple greetings or non-research questions, just respond naturally without searching\n" +
    "4. Synthesize all findings into a clear, well-structured briefing with markdown\n" +
    "5. Include inline citations [1], [2] etc. and end with a Sources section\n" +
    "Be thorough but concise. Prioritize accuracy and recency.\n\n" +
    "## Memory\n" +
    "You have persistent memory across sessions. Previous research and facts are automatically provided below. " +
    "Use the `remember` tool to search for specific past topics when needed.\n\n" +
    "If the user references a previous session (e.g. 'what did you find about X last time?'), " +
    "check your memory context or use the remember tool.";

  const enrichedQuery = memoryContext
    ? `## Memory context from past sessions\n${memoryContext}\n\n## User message\n${query}`
    : query;

  const messages = [{ role: "user", content: enrichedQuery }];

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
      const result = textBlocks.map((b) => b.text).join("") || "";
      saveTurn(query, result);
      return result;
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
  initMemory();
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
