#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Assignment.py — Multi-Agent Chat (Ollama primary + optional ChatGPT fallback) with:
- Tool Calling (pure-JSON), DI, Hybrid Agents
- Vector Memory (TF-IDF by default; optional Ollama embeddings)
- Interactive chat by default (scenarios behind --scenarios)
- Real tool: openai_search (uses ChatGPT API if OPENAI_API_KEY is set)
- Strict stripping of <think> blocks; banner shows "[thinking hidden]"
- Title bar shows agent name(s) for every reply
- UTC-safe timestamps; extended generation length

Run (PowerShell):
  C:\Users\pc\AppData\Local\Microsoft\WindowsApps\python3.13.exe -m pip install -r requirements.txt
  ollama pull qwen3:4b
  $env:OLLAMA_MODEL="qwen3:4b"
  # optional:
  # $env:OPENAI_API_KEY="sk-proj-REGENERATED_KEY"

  # interactive chat with memory persistence + logs:
  C:\Users\pc\AppData\Local\Microsoft\WindowsApps\python3.13.exe .\Assignment.py --persist --log chat.jsonl
"""

import os
import sys
import re
import json
import uuid
import traceback
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
#   Config / Defaults
# =========================
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "")  # e.g., "nomic-embed-text" if pulled in Ollama

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # optional; enables ChatGPT fallback + openai_search tool
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

BRIEF = os.environ.get("BRIEF", "1") == "1"

DEFAULT_SYSTEM_PREFIX = (
  "You are a precise assistant.\n"
  "TOOL CALLS:\n"
  "- If you need a tool, reply with a PURE JSON object ONLY: "
  "{\"tool_name\":\"<name>\",\"arguments\":{...}}. No surrounding text, no markdown, no commentary.\n"
  "- If you are NOT calling a tool, reply with: 1 TL;DR sentence + up to 5 bullets.\n"
  "HARD RULES:\n"
  "- Never output chain-of-thought or <think> blocks. Think silently; the UI will show a 'thinking hidden' banner if needed.\n"
  "- Do not describe your intentions about tools; either call them in JSON or answer concisely.\n"
)

STYLE_GUIDE = (
  "STYLE:\n"
  "- Be brief. Use numbered or bulleted lists.\n"
  "- Keep under ~120 words unless asked for detail.\n"
  "- State the answer first, then supporting bullets.\n"
  "- Avoid speculation and hedging; be direct.\n"
)

# =========================
#   Utilities / Tracing
# =========================
def now_ts() -> str:
    # timezone-aware UTC
    return dt.datetime.now(dt.timezone.utc).isoformat()

def short_id() -> str:
    return uuid.uuid4().hex[:8]

def jprint(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def strip_think(text: str) -> str:
    """Remove any <think>...</think> blocks (case-insensitive)."""
    return re.sub(r'<think\b[^>]*>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE).strip()

def had_think(text: str) -> bool:
    return bool(re.search(r'<think\b', text, flags=re.IGNORECASE))

def looks_like_tool_intent(text: str) -> bool:
    # model talked about tools but did not produce pure JSON
    return bool(re.search(r'\b(memory_search|mock_search|openai_search|calculator|tool_name)\b', text, flags=re.I))

# =========================
#   Tools
# =========================
class ToolError(Exception):
    pass

class ToolRegistry:
    def __init__(self):
        self._tools = {}
    def register(self, name: str, func, schema: Optional[Dict] = None, desc: str = ""):
        self._tools[name] = {"func": func, "schema": schema or {}, "desc": desc}
    def list_tools(self) -> Dict[str, Dict]:
        return self._tools
    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        return self._tools[name]["func"](**arguments)

def tool_calculator(expression: str) -> Dict[str, Any]:
    try:
        allowed = set("0123456789.+-*/() ")
        if not set(expression) <= allowed:
            raise ValueError("Unsupported characters.")
        result = eval(expression, {"__builtins__": {}}, {})
        return {"ok": True, "expression": expression, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

MOCK_CORPUS = [
    {"url": "kb://ml/optimizers", "title": "ML Optimization Techniques",
     "text": "Gradient Descent, Momentum, RMSProp, Adam, AdamW. Adam and AdamW often converge faster; AdamW helps with generalization."},
    {"url": "kb://ml/transformers", "title": "Transformer Architectures",
     "text": "Transformers rely on self-attention. Efficiency variants: Performer, Linformer, Longformer, Big Bird; trade-offs include memory vs accuracy."},
    {"url": "kb://ml/neural_networks", "title": "Types of Neural Networks",
     "text": "Feedforward, CNN, RNN, LSTM, GRU, Transformer, GNN. Transformers dominate NLP and are increasingly used in vision."},
    {"url": "kb://rl/papers", "title": "Recent RL Papers",
     "text": "Model-free vs model-based; challenges include sample efficiency, exploration, stability, reward design, and sim-to-real transfer."},
]

def tool_mock_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    q = set(query.lower().split())
    scored = []
    for doc in MOCK_CORPUS:
        score = len(q.intersection(set(doc["text"].lower().split())))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [d for _, d in scored[:top_k]]
    return {"ok": True, "query": query, "results": hits}

# =========================
#   Embedding / Vector memory
# =========================
class Embedder:
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

class TFIDFEmbedder(Embedder):
    def __init__(self):
        self._vec = TfidfVectorizer()
        self._fitted = False
    def fit(self, texts: List[str]):
        self._vec.fit(texts)
        self._fitted = True
    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return self._vec.transform(texts).toarray()

class OllamaEmbedder(Embedder):
    def __init__(self, model: str, base: str = OLLAMA_BASE):
        self.model = model
        self.base = base
    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            try:
                r = requests.post(f"{self.base}/api/embeddings",
                                  json={"model": self.model, "prompt": t}, timeout=30)
                r.raise_for_status()
                data = r.json()
                vecs.append(np.array(data["embedding"], dtype=np.float32))
            except Exception:
                vecs.append(self._hash_vec(t))
        return np.vstack(vecs)
    def _hash_vec(self, s: str, dim: int = 384) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(s)) % (2**32))
        v = rng.normal(size=dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

class MemoryLayer:
    def __init__(self, embedder: Optional[Embedder] = None):
        self.conversation: List[Dict[str, Any]] = []
        self.knowledge_base: List[Dict[str, Any]] = []
        self.agent_state: List[Dict[str, Any]] = []
        self._embedder = embedder if embedder else TFIDFEmbedder()
        self._kb_index_matrix: Optional[np.ndarray] = None
        self._kb_texts_cache: List[str] = []

    def add_conv(self, role: str, content: str):
        self.conversation.append({"ts": now_ts(), "role": role, "content": content})

    def add_kb(self, topic: str, text: str, source: str, agent: str, confidence: float):
        rid = short_id()
        rec = {"id": rid, "ts": now_ts(), "topic": topic, "text": text,
               "source": source, "agent": agent, "confidence": float(confidence)}
        self.knowledge_base.append(rec)
        self._kb_index_matrix = None
        return rid

    def search_kb_keyword(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = query.lower().split()
        scored = []
        for rec in self.knowledge_base:
            score = sum(t in rec["text"].lower() or t in (rec.get("topic","").lower()) for t in q)
            scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for s, r in scored[:top_k] if s > 0]

    def search_kb_vector(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.knowledge_base:
            return []
        if self._kb_index_matrix is None:
            self._kb_texts_cache = [rec["text"] for rec in self.knowledge_base]
            self._kb_index_matrix = self._embedder.embed(self._kb_texts_cache)
        qv = self._embedder.embed([query])
        sims = cosine_similarity(qv, self._kb_index_matrix)[0]
        idx = np.argsort(-sims)[:top_k]
        return [self.knowledge_base[i] | {"similarity": float(sims[i])} for i in idx if sims[i] > 0]

    def add_agent_state(self, agent: str, task_id: str, summary: str, data: Dict[str, Any]):
        self.agent_state.append({"ts": now_ts(), "agent": agent,
                                 "task_id": task_id, "summary": summary, "data": data})

# =========================
#   LLMs: Ollama + optional ChatGPT
# =========================
class LLMResponse:
    def __init__(self, text: str, used_backend: str, tool_json: Optional[Dict[str, Any]] = None):
        self.text = text
        self.used_backend = used_backend
        self.tool_json = tool_json

class BaseLLM:
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PREFIX):
        self.system_prompt = system_prompt
    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        raise NotImplementedError
    def _maybe_parse_tool(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            txt = text.strip()
            if txt.startswith("{") and txt.endswith("}"):
                data = json.loads(txt)
                if isinstance(data, dict) and "tool_name" in data and "arguments" in data and isinstance(data["arguments"], dict):
                    return data
        except Exception:
            pass
        return None

class OllamaLLM(BaseLLM):
    def __init__(self, model: str = OLLAMA_MODEL, base: str = OLLAMA_BASE, system_prompt: str = DEFAULT_SYSTEM_PREFIX):
        super().__init__(system_prompt)
        self.model = model
        self.base = base
    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt}] + messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "repeat_penalty": 1.05,
                "num_predict": 768  # extended to avoid truncation
            }
        }
        try:
            r = requests.post(f"{self.base}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            text = data["message"]["content"]
            return LLMResponse(text, used_backend="ollama", tool_json=self._maybe_parse_tool(text))
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

class ChatGPTLLM(BaseLLM):
    """Optional fallback if OPENAI_API_KEY is present."""
    def __init__(self, model: str = OPENAI_MODEL, api_key: Optional[str] = OPENAI_API_KEY, system_prompt: str = DEFAULT_SYSTEM_PREFIX):
        super().__init__(system_prompt)
        self.model = model
        self.api_key = api_key
    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        msgs = [{"role": "system", "content": self.system_prompt}] + messages
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "temperature": 0.1, "messages": msgs},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return LLMResponse(text, used_backend="openai", tool_json=self._maybe_parse_tool(text))
        except Exception as e:
            raise RuntimeError(f"OpenAI error: {e}")

class LLMRouter(BaseLLM):
    """Try Ollama first; if it fails and OPENAI_API_KEY is set, try ChatGPT; else rule-based fallback."""
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PREFIX):
        super().__init__(system_prompt)
        self.backends: List[BaseLLM] = [OllamaLLM(system_prompt=system_prompt)]
        if OPENAI_API_KEY:
            self.backends.append(ChatGPTLLM(system_prompt=system_prompt))
    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        last_err = None
        for be in self.backends:
            try:
                return be.generate(messages)
            except Exception as e:
                last_err = e
        text = "[fallback] " + messages[-1]["content"]
        return LLMResponse(text, used_backend="fallback", tool_json=self._maybe_parse_tool(text))

# =========================
#   Agents + Hybrid
# =========================
class Agent:
    def __init__(self, name: str, role_prompt: str, llm: BaseLLM, tools: ToolRegistry, memory: MemoryLayer):
        self.name = name
        self.role_prompt = role_prompt
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def is_relevant(self, query: str) -> float:
        return 0.0

    def estimate_confidence(self, answer: str) -> float:
        l = len(answer)
        if l < 40:  return 0.55
        if l < 200: return 0.7
        return 0.8

    def run(self, task_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        messages = [{"role": "system", "content": self.role_prompt}]
        if BRIEF:
            messages.append({"role": "system", "content": STYLE_GUIDE})
        messages.append({"role": "user", "content": query})

        trace = []
        for _ in range(8):  # allow retries for tool JSON compliance and finalization
            resp = self.llm.generate(messages)

            raw_text = resp.text or ""
            cleaned = strip_think(raw_text)
            tool_req = self.llm._maybe_parse_tool(cleaned)

            # model talked about tools but didn't send pure JSON? nudge and retry
            if (tool_req is None) and looks_like_tool_intent(cleaned):
                messages.append({
                    "role": "system",
                    "content": "REMINDER: If you need a tool, respond with a PURE JSON object ONLY. No other text."
                })
                continue

            # execute tool call
            if tool_req:
                try:
                    tool_result = self.tools.call(tool_req["tool_name"], tool_req["arguments"])
                except Exception as e:
                    tool_result = {"ok": False, "error": str(e)}
                messages.append({"role": "assistant", "content": json.dumps(tool_req)})
                messages.append({"role": "tool", "content": json.dumps(tool_result)})
                trace.append({"backend": resp.used_backend, "tool": tool_req, "result": tool_result})
                continue

            # ensure we have a non-empty final answer
            if not cleaned or cleaned.strip() in ("{}", "[]"):
                messages.append({"role":"system","content":"Do not include <think>. Provide the final answer now: TL;DR + up to 5 bullets."})
                continue

            answer = cleaned
            conf = self.estimate_confidence(answer)
            self.memory.add_agent_state(self.name, task_id, f"Answered: {query}", {"trace": trace})
            return {"agent": self.name, "answer": answer, "confidence": conf, "trace": trace, "had_think": had_think(raw_text)}

        # if loop exhausted
        answer = "TL;DR: I couldn’t finalize a response.\n- Please rephrase or try again."
        conf = 0.3
        self.memory.add_agent_state(self.name, task_id, f"Partial: {query}", {"trace": trace})
        return {"agent": self.name, "answer": answer, "confidence": conf, "trace": trace, "had_think": True}

class ResearchAgent(Agent):
    def is_relevant(self, query: str) -> float:
        keys = ["find", "research", "look up", "discover", "papers", "info", "information", "recent", "news", "latest"]
        q = query.lower()
        score = 0.0
        if any(k in q for k in keys): score += 0.6
        if any(w in q for w in ["transformer","reinforcement","neural","optimizer","architecture","paper"]): score += 0.3
        return min(1.0, score)

class AnalysisAgent(Agent):
    def is_relevant(self, query: str) -> float:
        keys = ["analyze", "compare", "trade-off", "efficiency", "which is better", "recommend", "calculate", "pros", "cons"]
        q = query.lower()
        score = 0.0
        if any(k in q for k in keys): score += 0.7
        if "compare" in q or "vs" in q: score += 0.2
        return min(1.0, score)

class MemoryAgent(Agent):
    def is_relevant(self, query: str) -> float:
        keys = ["what did we learn", "earlier", "previous", "remember", "recall", "stored", "memory", "before", "summarize our"]
        q = query.lower()
        return 0.8 if any(k in q for k in keys) else 0.1

class HybridAgent(Agent):
    def __init__(self, name: str, agents: List[Agent], llm: BaseLLM):
        role_prompt = "\n\n".join([f"[{a.name} ROLE]\n{a.role_prompt}" for a in agents])
        tools = agents[0].tools
        memory = agents[0].memory
        super().__init__(name=name, role_prompt=role_prompt, llm=llm, tools=tools, memory=memory)
        self._parts = agents
    def is_relevant(self, query: str) -> float:
        return max(a.is_relevant(query) for a in self._parts)

# =========================
#   Coordinator / Planner
# =========================
class Coordinator:
    def __init__(self, agents: List[Agent], memory: MemoryLayer):
        self.agents = agents
        self.memory = memory

    def _plan(self, query: str, eligible: List[Agent]) -> List[Agent]:
        q = query.lower()
        by_name = {a.name: a for a in eligible}
        plan: List[Agent] = []

        # memory-first if explicitly recalling
        if any(isinstance(a, MemoryAgent) and a in eligible for a in self.agents) and \
           any(w in q for w in ["memory","what did we learn","earlier","previous","before","remember"]):
            plan.append(by_name.get("MemoryAgent"))

        if any(isinstance(a, ResearchAgent) and a in eligible for a in self.agents) and \
           any(w in q for w in ["research","find","papers","information","info","latest","recent","news"]):
            plan.append(by_name.get("ResearchAgent"))

        if any(isinstance(a, AnalysisAgent) and a in eligible for a in self.agents) and \
           any(w in q for w in ["analyze","compare","trade-off","recommend","which is better","efficiency","pros","cons"]):
            plan.append(by_name.get("AnalysisAgent"))

        seen = set(); final = []
        for a in plan:
            if a and a not in seen:
                final.append(a); seen.add(a)
        if final: return final

        scores = sorted(((a.is_relevant(query), a) for a in eligible), reverse=True, key=lambda x: x[0])
        return [scores[0][1]] if scores else []

    def handle_query(self, query: str) -> Dict[str, Any]:
        task_id = short_id()
        self.memory.add_conv("user", query)

        elig = [a for a in self.agents if a.is_relevant(query) >= 0.5]
        call_chain: List[Agent] = self._plan(query, elig)
        if len(call_chain) >= 2:
            used_agent = HybridAgent(
                name="HybridAgent(" + "+".join(a.name for a in call_chain) + ")",
                agents=call_chain,
                llm=LLMRouter(system_prompt=DEFAULT_SYSTEM_PREFIX)
            )
        elif len(call_chain) == 1:
            used_agent = call_chain[0]
        else:
            used_agent = HybridAgent(
                name="HybridAgent(All)",
                agents=self.agents,
                llm=LLMRouter(system_prompt=DEFAULT_SYSTEM_PREFIX)
            )

        trace = {"task_id": task_id, "selected": [a.name for a in (call_chain if call_chain else self.agents)],
                 "ts": now_ts()}

        context: Dict[str, Any] = {}
        answers: List[Dict[str, Any]] = []

        if any(isinstance(a, ResearchAgent) for a in call_chain):
            ra = next(a for a in call_chain if isinstance(a, ResearchAgent))
            ar = ra.run(task_id, query, context); answers.append(ar); context["research"] = ar["answer"]

        if any(isinstance(a, AnalysisAgent) for a in call_chain):
            aa = next(a for a in call_chain if isinstance(a, AnalysisAgent))
            q2 = query + ("\n\nContext from Research:\n" + context["research"] if "research" in context else "")
            ar = aa.run(task_id, q2, context); answers.append(ar); context["analysis"] = ar["answer"]

        if any(isinstance(a, MemoryAgent) for a in call_chain):
            ma = next(a for a in call_chain if isinstance(a, MemoryAgent))
            ar = ma.run(task_id, query, context); answers.append(ar)

        if not answers:
            ar = used_agent.run(task_id, query, context)
            answers.append(ar)

        final_answer = self._synthesize(answers)
        final_answer = strip_think(final_answer)  # final safety
        self.memory.add_conv("assistant", final_answer)

        if "analysis" in context:
            self.memory.add_kb(topic="analysis", text=context["analysis"], source="coordinator", agent="Coordinator", confidence=0.75)
        elif "research" in context:
            self.memory.add_kb(topic="research", text=context["research"], source="coordinator", agent="Coordinator", confidence=0.6)

        agent_names = "+".join(a.name for a in (call_chain if call_chain else self.agents)) if call_chain else "HybridAgent(All)"
        thinking_flag = any(a.get("had_think") for a in answers)

        return {
            "task_id": task_id,
            "trace": trace,
            "answers": answers,
            "final": final_answer,
            "agent_names": agent_names,
            "thinking_hidden": thinking_flag
        }

    def _synthesize(self, parts: List[Dict[str, Any]]) -> str:
        if not parts: return "No answer produced."
        if len(parts) == 1: return parts[0]["answer"]
        buf = ["SYNTHESIZED ANSWER (from {} agents):".format(len(parts))]
        for p in parts:
            buf.append(f"- [{p['agent']}] ({p['confidence']:.2f}): {p['answer']}")
        return strip_think("\n".join(buf))

# =========================
#   DI wiring (incl. real OpenAI tool)
# =========================
def build_system() -> Tuple['Coordinator', 'ToolRegistry', 'MemoryLayer']:
    embedder: Embedder = OllamaEmbedder(OLLAMA_EMBED_MODEL) if OLLAMA_EMBED_MODEL else TFIDFEmbedder()
    memory = MemoryLayer(embedder)

    tools = ToolRegistry()
    tools.register("calculator", tool_calculator, desc="Evaluate arithmetic expression")
    tools.register("mock_search", tool_mock_search, desc="Search a tiny internal KB")

    def tool_memory_search(query: str, top_k: int = 5, mode: str = "vector") -> Dict[str, Any]:
        if mode == "keyword":
            res = memory.search_kb_keyword(query, top_k=top_k)
        else:
            res = memory.search_kb_vector(query, top_k=top_k)
        return {"ok": True, "results": res}
    tools.register("memory_search", tool_memory_search, desc="Search memory (keyword/vector)")

    # REAL tool via ChatGPT (search/summarize)
    def tool_openai_search(query: str, max_tokens: int = 300) -> Dict[str, Any]:
        if not OPENAI_API_KEY:
            return {"ok": False, "error": "OPENAI_API_KEY not set"}
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": "You are a web research assistant. Summarize the most relevant, recent information for the given query."},
                        {"role": "user", "content": query}
                    ]
                },
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            return {"ok": True, "query": query, "answer": text}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    tools.register("openai_search", tool_openai_search, desc="Ask ChatGPT to fetch and summarize live information")

    # Agent role prompts with decision policy
    policy = "\nDecision policy: Call a tool ONLY if it’s required to compute/lookup. If you already know, answer directly."
    research_prompt = (
        "You are ResearchAgent. Gather factual points and use tools when needed:\n"
        "- openai_search(query, max_tokens)\n- mock_search(query, top_k)\n- memory_search(query, top_k, mode)\n"
    ) + policy
    analysis_prompt = (
        "You are AnalysisAgent. Compare options, compute, and recommend. Tools:\n"
        "- calculator(expression)\n- memory_search(query, top_k, mode)\n"
    ) + policy
    memory_prompt = (
        "You are MemoryAgent. Recall/summarize from memory. Tools:\n"
        "- memory_search(query, top_k, mode)\n"
    ) + policy

    llm = LLMRouter(system_prompt=DEFAULT_SYSTEM_PREFIX)
    research_agent = ResearchAgent("ResearchAgent", research_prompt, llm, tools, memory)
    analysis_agent = AnalysisAgent("AnalysisAgent", analysis_prompt, llm, tools, memory)
    memory_agent  = MemoryAgent("MemoryAgent",  memory_prompt,  llm, tools, memory)

    coord = Coordinator([research_agent, analysis_agent, memory_agent], memory)
    return coord, tools, memory

# =========================
#   Demo / Scenarios
# =========================
SCENARIOS = [
    ("Simple Query", "What are the main types of neural networks?"),
    ("Complex Query", "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs."),
    ("Memory Test", "What did we discuss about neural networks earlier?"),
    ("Multi-step", "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges."),
    ("Collaborative", "Compare two machine-learning approaches and recommend which is better for our use case."),
]

def run_scenarios():
    coord, tools, memory = build_system()
    print("=== Running Sample Scenarios ===")
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    for title, query in SCENARIOS:
        print(f"\n--- {title} ---")
        res = coord.handle_query(query)
        jprint({"title": title, "task_id": res["task_id"], "trace": res["trace"], "final": res["final"]})
        with open(os.path.join(out_dir, f"{title.lower().replace(' ', '_')}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{title}\nQuery: {query}\n\nTrace:\n")
            f.write(json.dumps(res["trace"], indent=2))
            f.write("\n\nAgent Answers:\n")
            for a in res["answers"]:
                f.write(f"\n[{a['agent']}] (conf={a['confidence']:.2f})\n{a['answer']}\n")
            f.write("\n\nFinal:\n")
            f.write(res["final"])

# =========================
#   UI / Render
# =========================
def render_boxed(agent_names: str, thinking_hidden: bool, body: str):
    banner = f"[ Agent: {agent_names} ]" + (" [thinking hidden]" if thinking_hidden else "")
    # ensure a minimum padding
    line_len = max(4, len(banner) + 1)
    line = "═" * line_len
    print(f"\n╔{line}╗")
    print(f"║ {banner.ljust(line_len)}║")
    print(f"╚{line}╝")
    print(body)

# =========================
#   Main (chat default)
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Chat (Ollama primary, optional ChatGPT fallback)")
    parser.add_argument("--scenarios", action="store_true", help="Run built-in sample scenarios")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode (default)")
    parser.add_argument("--once", type=str, default="", help="Single-shot query (no prompt loop)")
    parser.add_argument("--persist", action="store_true", help="Persist memory to ./state.json between runs")
    parser.add_argument("--log", type=str, default="", help="Write JSONL chat log to this file")
    args = parser.parse_args()

    coord, tools, memory = build_system()

    # persistence
    state_path = os.path.join(os.getcwd(), "state.json")
    if args.persist and os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            memory.conversation = state.get("conversation", [])
            memory.knowledge_base = state.get("knowledge_base", [])
            memory.agent_state = state.get("agent_state", [])
        except Exception as e:
            print("Could not load state.json:", e)

    def persist_now():
        if args.persist:
            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "conversation": memory.conversation,
                        "knowledge_base": memory.knowledge_base,
                        "agent_state": memory.agent_state
                    }, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print("Persist error:", e)

    # logging
    log_fp = open(args.log, "a", encoding="utf-8") if args.log else None
    def log_event(role, content, meta=None):
        if log_fp:
            rec = {"ts": now_ts(), "role": role, "content": content}
            if meta: rec["meta"] = meta
            log_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_fp.flush()

    # scenarios
    if args.scenarios:
        run_scenarios()
        persist_now()
        if log_fp: log_fp.close()
        sys.exit(0)

    # once
    if args.once:
        q = args.once.strip()
        res = coord.handle_query(q)
        render_boxed(res.get("agent_names","Agent"), res.get("thinking_hidden", False), res["final"])
        log_event("user", q)
        log_event("assistant", res["final"], {"trace": res["trace"], "agents": res.get("agent_names")})
        persist_now()
        if log_fp: log_fp.close()
        sys.exit(0)

    # default: interactive chat
    print("Interactive chat (memory in-session{}). Type 'exit' to quit."
          .format(" + persisted to state.json" if args.persist else ""))

    while True:
        try:
            q = input("\nYou: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            log_event("user", q)
            res = coord.handle_query(q)
            render_boxed(res.get("agent_names","Agent"), res.get("thinking_hidden", False), res["final"])
            log_event("assistant", res["final"], {"trace": res["trace"], "agents": res.get("agent_names")})
            persist_now()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)
            traceback.print_exc()

    if log_fp: log_fp.close()
