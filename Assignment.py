#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Agent Chat System (Ollama + optional Groq) with Tool Calling, DI, Hybrid Agents, and Vector Memory.

- Agents: ResearchAgent, AnalysisAgent, MemoryAgent, plus HybridAgent
- Coordinator: routes, plans, merges outputs, maintains context
- Memory: conversation history, knowledge base, agent state + keyword + vector search
- LLM backends: Ollama (local), Groq (optional). Graceful fallback to rule-based if LLMs unavailable.
- Tool calling: model returns a JSON like {"tool_name": "...", "arguments": {...}}; we execute and feed results back.

Run:
  pip install requests scikit-learn numpy
  python multi_agent_system.py

Notes:
  - By default uses Ollama model "llama3.1". Change OLLAMA_MODEL below if preferred.
  - If GROQ_API_KEY is set, we try Groq backend before falling back to Ollama; otherwise only Ollama.
  - To enable Ollama embeddings, set OLLAMA_EMBED_MODEL = "nomic-embed-text" or similar and ensure it's pulled.

Assignment reference: See “Technical Assessment — Simple Multi-Agent Chat System”. (PDF provided by user)
"""

import os
import sys
import json
import time
import math
import uuid
import queue
import traceback
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

# --- Minimal deps vector memory ---
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import requests

# =========================
#   Config / Constants
# =========================
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "")  # e.g. "nomic-embed-text", leave empty to use TF-IDF

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")  # free/dev tier-friendly

DEFAULT_SYSTEM_PREFIX = (
    "You are an assistant that may call tools by replying with a pure JSON object ONLY when needed, "
    "in the exact form: {\"tool_name\": \"<name>\", \"arguments\": { ... }}. "
    "Otherwise, reply with a helpful natural language answer. Never include extra text with the JSON."
)

# =========================
#   Utility / Tracing
# =========================
def now_ts() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"

def short_id() -> str:
    return uuid.uuid4().hex[:8]

def jprint(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

# =========================
#   Tool Registry
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

# Some basic tools (calculator, mock web/KB search, memory search)
def tool_calculator(expression: str) -> Dict[str, Any]:
    try:
        # very safe eval: implement a tiny parser for + - * / ( ) and floats
        allowed = set("0123456789.+-*/() ")
        if not set(expression) <= allowed:
            raise ValueError("Unsupported characters in expression.")
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
    # tiny scoring by token overlap
    q = set(query.lower().split())
    scored = []
    for doc in MOCK_CORPUS:
        score = len(q.intersection(set(doc["text"].lower().split())))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [d for _, d in scored[:top_k]]
    return {"ok": True, "query": query, "results": hits}

# memory tool will be wired later to MemoryLayer instance via a closure in DI

# =========================
#   Embeddings / Vectorizer
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
            # fit on the fly with provided texts
            self.fit(texts)
        return self._vec.transform(texts).toarray()

class OllamaEmbedder(Embedder):
    def __init__(self, model: str, base: str = OLLAMA_BASE):
        self.model = model
        self.base = base

    def embed(self, texts: List[str]) -> np.ndarray:
        # call /api/embeddings once per text (kept simple for clarity)
        vecs = []
        for t in texts:
            try:
                r = requests.post(f"{self.base}/api/embeddings", json={"model": self.model, "prompt": t}, timeout=30)
                r.raise_for_status()
                data = r.json()
                vecs.append(np.array(data["embedding"], dtype=np.float32))
            except Exception as e:
                # fallback to a trivial hash-based vector
                vecs.append(self._hash_vec(t))
        return np.vstack(vecs)

    def _hash_vec(self, s: str, dim: int = 384) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(s)) % (2**32))
        v = rng.normal(size=dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

# =========================
#   Memory Layer
# =========================
class MemoryLayer:
    """
    Structured memory across:
      - conversation: [{ts, role, content}]
      - knowledge_base: [{id, ts, topic, text, source, agent, confidence}]
      - agent_state: [{ts, agent, task_id, summary, data}]
    With keyword search and vector similarity.
    """
    def __init__(self, embedder: Optional[Embedder] = None):
        self.conversation: List[Dict[str, Any]] = []
        self.knowledge_base: List[Dict[str, Any]] = []
        self.agent_state: List[Dict[str, Any]] = []
        self._embedder = embedder if embedder else TFIDFEmbedder()
        self._kb_index_matrix: Optional[np.ndarray] = None
        self._kb_texts_cache: List[str] = []

    # Conversation
    def add_conv(self, role: str, content: str):
        self.conversation.append({"ts": now_ts(), "role": role, "content": content})

    # Knowledge
    def add_kb(self, topic: str, text: str, source: str, agent: str, confidence: float):
        rid = short_id()
        rec = {
            "id": rid, "ts": now_ts(),
            "topic": topic, "text": text,
            "source": source, "agent": agent, "confidence": float(confidence)
        }
        self.knowledge_base.append(rec)
        # invalidate vector index
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

    # Agent state
    def add_agent_state(self, agent: str, task_id: str, summary: str, data: Dict[str, Any]):
        self.agent_state.append({
            "ts": now_ts(),
            "agent": agent, "task_id": task_id,
            "summary": summary, "data": data
        })

# =========================
#   LLM Backends
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
            # Must be a pure JSON object to be considered a tool call.
            if (txt.startswith("{") and txt.endswith("}")):
                data = json.loads(txt)
                if "tool_name" in data and "arguments" in data and isinstance(data["arguments"], dict):
                    return data
        except Exception:
            pass
        return None

class GroqLLM(BaseLLM):
    def __init__(self, model: str = GROQ_MODEL, api_key: Optional[str] = GROQ_API_KEY, system_prompt: str = DEFAULT_SYSTEM_PREFIX):
        super().__init__(system_prompt)
        self.model = model
        self.api_key = api_key

    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        # Compose messages with system
        msgs = [{"role": "system", "content": self.system_prompt}] + messages
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": msgs, "temperature": 0.2},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return LLMResponse(text, used_backend="groq", tool_json=self._maybe_parse_tool(text))
        except Exception as e:
            raise RuntimeError(f"Groq error: {e}")

class OllamaLLM(BaseLLM):
    def __init__(self, model: str = OLLAMA_MODEL, base: str = OLLAMA_BASE, system_prompt: str = DEFAULT_SYSTEM_PREFIX):
        super().__init__(system_prompt)
        self.model = model
        self.base = base

    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        # Convert OpenAI-style messages to Ollama prompt with system
        payload = {
            "model": self.model,
            "messages": [{"role":"system","content": self.system_prompt}] + messages,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        try:
            r = requests.post(f"{self.base}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            text = data["message"]["content"]
            return LLMResponse(text, used_backend="ollama", tool_json=self._maybe_parse_tool(text))
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

class LLMRouter(BaseLLM):
    """Try Groq first (if key set), else Ollama; if both fail, degrade."""
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PREFIX):
        super().__init__(system_prompt)
        self.backends: List[BaseLLM] = []
        if GROQ_API_KEY:
            self.backends.append(GroqLLM(system_prompt=system_prompt))
        self.backends.append(OllamaLLM(system_prompt=system_prompt))

    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        last_err = None
        for be in self.backends:
            try:
                return be.generate(messages)
            except Exception as e:
                last_err = e
        # Fallback: simple rule-based echo
        text = "[Rule-based fallback] " + messages[-1]["content"]
        return LLMResponse(text, used_backend="fallback", tool_json=self._maybe_parse_tool(text))

# =========================
#   Agent Base / Hybrids
# =========================
class Agent:
    def __init__(self, name: str, role_prompt: str, llm: BaseLLM, tools: ToolRegistry, memory: MemoryLayer):
        self.name = name
        self.role_prompt = role_prompt
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def is_relevant(self, query: str) -> float:
        """
        Quick heuristic + LLM check could be implemented.
        We do a simple keyword map; subclasses may override.
        Returns a score in [0,1].
        """
        return 0.0

    def run(self, task_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent: may call tools via LLM decision.
        Returns structured dict with "answer" and "confidence".
        """
        messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": query},
        ]
        trace = []
        for _ in range(4):  # allow a few tool iterations
            resp = self.llm.generate(messages)
            trace.append({"backend": resp.used_backend, "text": resp.text})
            tool_req = resp.tool_json
            if tool_req:
                # Execute tool
                try:
                    tool_result = self.tools.call(tool_req["tool_name"], tool_req["arguments"])
                except Exception as e:
                    tool_result = {"ok": False, "error": str(e)}
                # Feed back to model
                messages.append({"role": "assistant", "content": json.dumps(tool_req)})
                messages.append({"role": "tool", "content": json.dumps(tool_result)})
                continue
            else:
                # final natural language
                answer = resp.text.strip()
                conf = self.estimate_confidence(answer)
                self.memory.add_agent_state(self.name, task_id, f"Answered: {query}", {"trace": trace})
                return {"agent": self.name, "answer": answer, "confidence": conf, "trace": trace}
        # If loop ends without NL answer:
        answer = "I attempted tools but could not finalize a natural language answer."
        conf = 0.3
        self.memory.add_agent_state(self.name, task_id, f"Partial: {query}", {"trace": trace})
        return {"agent": self.name, "answer": answer, "confidence": conf, "trace": trace}

    def estimate_confidence(self, answer: str) -> float:
        l = len(answer)
        if l < 40:
            return 0.55
        if l < 200:
            return 0.7
        return 0.8

class ResearchAgent(Agent):
    def is_relevant(self, query: str) -> float:
        keys = ["find", "research", "look up", "discover", "papers", "info", "information", "recent"]
        q = query.lower()
        score = 0.0
        if any(k in q for k in keys):
            score += 0.6
        if "transformer" in q or "reinforcement" in q or "neural" in q or "optimizer" in q:
            score += 0.3
        return min(1.0, score)

class AnalysisAgent(Agent):
    def is_relevant(self, query: str) -> float:
        keys = ["analyze", "compare", "trade-off", "efficiency", "which is better", "recommend", "calculate"]
        q = query.lower()
        score = 0.0
        if any(k in q for k in keys):
            score += 0.7
        if "compare" in q or "vs" in q:
            score += 0.2
        return min(1.0, score)

class MemoryAgent(Agent):
    def is_relevant(self, query: str) -> float:
        keys = ["what did we learn", "earlier", "previous", "remember", "recall", "stored", "memory"]
        q = query.lower()
        return 0.8 if any(k in q for k in keys) else 0.1

# Hybrid agent concatenates prompts and delegates to merged persona
class HybridAgent(Agent):
    def __init__(self, name: str, agents: List[Agent], llm: BaseLLM):
        role_prompt = "\n\n".join([f"[{a.name} ROLE]\n{a.role_prompt}" for a in agents])
        # We share tools & memory from the first agent (they all share the same instances via DI)
        tools = agents[0].tools
        memory = agents[0].memory
        super().__init__(name=name, role_prompt=role_prompt, llm=llm, tools=tools, memory=memory)
        self._parts = agents

    def is_relevant(self, query: str) -> float:
        # Combined relevance: max of parts
        return max(a.is_relevant(query) for a in self._parts)

# =========================
#   Coordinator / Planner
# =========================
class Coordinator:
    def __init__(self, agents: List[Agent], memory: MemoryLayer):
        self.agents = agents
        self.memory = memory

    def _basic_complexity(self, query: str) -> int:
        q = query.lower()
        # count verbs hinting multi-step
        hints = ["research", "analyze", "compare", "summarize", "find", "identify"]
        return sum(h in q for h in hints)

    def _plan(self, query: str, eligible: List[Agent]) -> List[Agent]:
        """
        Very small planner:
          - If memory-like query: MemoryAgent first
          - If research+analysis: ResearchAgent -> AnalysisAgent
          - Else: whichever has highest relevance
        """
        q = query.lower()
        by_name = {a.name: a for a in eligible}

        plan: List[Agent] = []
        if any(isinstance(a, MemoryAgent) and a in eligible for a in self.agents) and \
           ("memory" in q or "what did we learn" in q or "earlier" in q):
            plan.append(by_name.get("MemoryAgent"))

        if any(isinstance(a, ResearchAgent) and a in eligible for a in self.agents) and \
           ("research" in q or "find" in q or "papers" in q or "information" in q):
            plan.append(by_name.get("ResearchAgent"))
        if any(isinstance(a, AnalysisAgent) and a in eligible for a in self.agents) and \
           ("analyze" in q or "compare" in q or "trade-off" in q or "recommend" in q or "which is better" in q):
            plan.append(by_name.get("AnalysisAgent"))

        # Remove None, keep order, dedupe
        seen = set()
        final = []
        for a in plan:
            if a and a not in seen:
                final.append(a)
                seen.add(a)
        if final:
            return final

        # fallback: choose top-1 relevant
        scores = [(a.is_relevant(query), a) for a in eligible]
        scores.sort(reverse=True, key=lambda x: x[0])
        return [scores[0][1]] if scores else []

    def handle_query(self, query: str) -> Dict[str, Any]:
        task_id = short_id()
        self.memory.add_conv("user", query)

        # 1) select relevant agents
        elig = [a for a in self.agents if a.is_relevant(query) >= 0.5]
        # If multiple, compose hybrid persona for merged prompt
        used_agent: Agent
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
            # if none obvious, use a hybrid of all
            used_agent = HybridAgent(
                name="HybridAgent(All)",
                agents=self.agents,
                llm=LLMRouter(system_prompt=DEFAULT_SYSTEM_PREFIX)
            )

        # 2) Orchestrate substeps if needed
        trace = {"task_id": task_id, "selected": [a.name for a in (call_chain if call_chain else self.agents)],
                 "ts": now_ts()}

        context: Dict[str, Any] = {}

        # Run Research if in plan, then Analysis with that result, update Memory at end
        answers = []
        if any(isinstance(a, ResearchAgent) for a in call_chain):
            ra = next(a for a in call_chain if isinstance(a, ResearchAgent))
            ar = ra.run(task_id, query, context)
            answers.append(ar)
            context["research"] = ar["answer"]
        if any(isinstance(a, AnalysisAgent) for a in call_chain):
            aa = next(a for a in call_chain if isinstance(a, AnalysisAgent))
            q2 = query
            if "research" in context:
                q2 += "\n\nContext from Research:\n" + context["research"]
            ar = aa.run(task_id, q2, context)
            answers.append(ar)
            context["analysis"] = ar["answer"]
        if any(isinstance(a, MemoryAgent) for a in call_chain):
            ma = next(a for a in call_chain if isinstance(a, MemoryAgent))
            ar = ma.run(task_id, query, context)
            answers.append(ar)

        # If we used a single agent or hybrid directly (no explicit chain handled above), run it
        if not answers:
            ar = used_agent.run(task_id, query, context)
            answers.append(ar)

        # 3) Synthesize & store
        final_answer = self._synthesize(answers)
        self.memory.add_conv("assistant", final_answer)

        # Persist knowledge if it looks useful (very simple heuristic)
        if "analysis" in context:
            self.memory.add_kb(topic="analysis", text=context["analysis"], source="coordinator", agent="Coordinator", confidence=0.75)
        elif "research" in context:
            self.memory.add_kb(topic="research", text=context["research"], source="coordinator", agent="Coordinator", confidence=0.6)

        return {"task_id": task_id, "trace": trace, "answers": answers, "final": final_answer}

    def _synthesize(self, parts: List[Dict[str, Any]]) -> str:
        if not parts:
            return "No answer produced."
        if len(parts) == 1:
            return parts[0]["answer"]
        # Simple synthesis rule
        buf = []
        buf.append("SYNTHESIZED ANSWER (from {} agents):".format(len(parts)))
        for p in parts:
            buf.append(f"- [{p['agent']}] ({p['confidence']:.2f}): {p['answer']}")
        return "\n".join(buf)


# =========================
#   Dependency Injection setup
# =========================
def build_system() -> Tuple[Coordinator, ToolRegistry, MemoryLayer]:
    # Memory + embedder
    embedder: Embedder
    if OLLAMA_EMBED_MODEL:
        embedder = OllamaEmbedder(OLLAMA_EMBED_MODEL)
    else:
        embedder = TFIDFEmbedder()
    memory = MemoryLayer(embedder)

    # Tooling
    tools = ToolRegistry()
    tools.register("calculator", tool_calculator, desc="Evaluate arithmetic expression")
    tools.register("mock_search", tool_mock_search, desc="Search in a small internal corpus")

    # Memory search tool closure (wired to memory)
    def tool_memory_search(query: str, top_k: int = 5, mode: str = "vector") -> Dict[str, Any]:
        if mode == "keyword":
            res = memory.search_kb_keyword(query, top_k=top_k)
        else:
            res = memory.search_kb_vector(query, top_k=top_k)
        return {"ok": True, "results": res}
    tools.register("memory_search", tool_memory_search, desc="Search memory (keyword/vector)")

    # LLM router
    llm = LLMRouter()

    # Agent role prompts
    research_prompt = (
        "You are ResearchAgent. You gather factual points and citations (if provided) and can call tools:\n"
        "- mock_search(query, top_k): internal KB lookup\n"
        "- memory_search(query, top_k, mode): retrieve prior knowledge\n"
        "Prefer using tools to collect succinct bullet points."
    )
    analysis_prompt = (
        "You are AnalysisAgent. You compare options, analyze trade-offs, compute simple things. Tools:\n"
        "- calculator(expression)\n"
        "- memory_search(query, top_k, mode)\n"
        "Return concise, decision-focused analysis."
    )
    memory_prompt = (
        "You are MemoryAgent. You recall and summarize what we learned earlier and can search memory via:\n"
        "- memory_search(query, top_k, mode)\n"
        "If asked, summarize previously stored findings."
    )

    # Agents (DI)
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
        # print trace
        jprint({"title": title, "task_id": res["task_id"], "trace": res["trace"], "final": res["final"]})
        # save
        with open(os.path.join(out_dir, f"{title.lower().replace(' ', '_')}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{title}\nQuery: {query}\n\n")
            f.write("Trace:\n")
            f.write(json.dumps(res["trace"], indent=2))
            f.write("\n\nAgent Answers:\n")
            for a in res["answers"]:
                f.write(f"\n[{a['agent']}] (conf={a['confidence']:.2f})\n{a['answer']}\n")
            f.write("\n\nFinal:\n")
            f.write(res["final"])

def interactive_chat():
    coord, tools, memory = build_system()
    print("Interactive chat. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nYou: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            res = coord.handle_query(q)
            print("\nAssistant:\n" + res["final"])
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)
            traceback.print_exc()

if __name__ == "__main__":
    if "--chat" in sys.argv:
        interactive_chat()
    else:
        run_scenarios()
