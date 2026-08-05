"""
The fraud-detection agent: given a booking transaction, it uses an LLM with
tool-use to investigate the booking (geo mismatch, account history, velocity,
price anomaly, booking pattern, email reputation), then returns a structured
FraudVerdict.

Supports two free-tier-friendly providers, selected via LLM_PROVIDER:
  - "groq"   (default): fast open-weight models via GroqCloud, generous free tier.
  - "gemini": Google's Gemini API, also has a genuinely free tier.

Both run a real agentic loop: the model decides which tools to call, sees the
results, can call more tools, and only then produces its final structured
verdict. It is not a fixed if/else pipeline with an LLM bolted on top.
"""
from __future__ import annotations

import json
import os
import time

from models import BookingTransaction, FraudVerdict, ToolCallLog
from tools import TOOL_DEFINITIONS, FraudTools

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()  # "groq" or "gemini"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

SYSTEM_PROMPT = """You are a fraud detection analyst for an online travel booking platform.

You will be given a booking transaction. Investigate it using the available tools \
before forming a judgment — do not guess at values the tools would tell you. Call \
whichever tools are relevant; you may call several, and you may call them in any \
order. You do not need to call every tool if the transaction is clearly low-risk, \
but for anything with plausible red flags, be thorough.

Once you have enough information, produce your FINAL answer as a single JSON object \
(no markdown fences, no commentary before or after) with exactly these fields:
{
  "risk_score": <integer 0-100, 0=definitely legitimate, 100=definitely fraudulent>,
  "risk_level": "<low|medium|high|critical>",
  "flags": [<short strings naming each specific red flag found, empty list if none>],
  "explanation": "<2-4 sentence explanation of your reasoning, written for a human fraud reviewer>",
  "recommended_action": "<one of: approve | approve_with_monitoring | manual_review | hold_for_verification | decline>"
}

Risk level guidance: low = 0-24, medium = 25-54, high = 55-79, critical = 80-100.
Be calibrated: most legitimate travel bookings should score low. Reserve high/critical \
scores for transactions with multiple corroborating red flags, not a single weak signal.
IMPORTANT: your very last message must be ONLY that JSON object, nothing else.
"""


def _with_retry(fn, max_retries: int = 5, base_delay: float = 2.0):
    """Free-tier APIs have tight rate limits and occasional overload errors;
    back off and retry on 429 (rate limit) and 503 (overloaded) responses."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - provider SDKs raise different exception types
            msg = str(e).lower()
            is_transient = any(
                s in msg for s in ["429", "503", "rate", "quota", "overload", "unavailable", "high demand"]
            )
            if not is_transient or attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay)


def _parse_verdict_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to salvage a JSON object embedded in extra prose.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {
            "risk_score": 50,
            "risk_level": "medium",
            "flags": ["verdict_parse_error"],
            "explanation": f"Could not parse agent output as JSON. Raw output: {text[:500]}",
            "recommended_action": "manual_review",
        }


class FraudAgent:
    def __init__(
        self,
        dataset: list[dict] | None = None,
        provider: str = LLM_PROVIDER,
        max_tool_iterations: int = 6,
    ):
        self.provider = provider
        self.tools = FraudTools(dataset=dataset)
        self.max_tool_iterations = max_tool_iterations

        if self.provider == "groq":
            from groq import Groq

            self.client = Groq()  # reads GROQ_API_KEY from env
            self.model = GROQ_MODEL
        elif self.provider == "gemini":
            from google import genai

            # The installed google-genai SDK version looks for GOOGLE_API_KEY,
            # not GEMINI_API_KEY, so pass it through explicitly for clarity
            # and so GEMINI_API_KEY (the name Google's own docs use) works too.
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self.client = genai.Client(api_key=api_key)
            self.model = GEMINI_MODEL
        else:
            raise ValueError(f"Unknown LLM_PROVIDER '{self.provider}'. Use 'groq' or 'gemini'.")

    def analyze(self, txn: BookingTransaction) -> FraudVerdict:
        user_message = (
            "Investigate this booking transaction and return your final JSON verdict.\n\n"
            + txn.model_dump_json(indent=2)
        )
        if self.provider == "groq":
            verdict_data, tool_log = self._run_groq(user_message)
        else:
            verdict_data, tool_log = self._run_gemini(user_message)

        return FraudVerdict(
            booking_id=txn.booking_id,
            risk_score=verdict_data["risk_score"],
            risk_level=verdict_data["risk_level"],
            flags=verdict_data.get("flags", []),
            explanation=verdict_data.get("explanation", ""),
            recommended_action=verdict_data.get("recommended_action", "manual_review"),
            tool_calls=tool_log,
        )

    # ---- Groq (OpenAI-compatible tool-calling) --------------------------------

    def _groq_tool_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in TOOL_DEFINITIONS
        ]

    def _run_groq(self, user_message: str) -> tuple[dict, list[ToolCallLog]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        tool_schema = self._groq_tool_schema()
        tool_log: list[ToolCallLog] = []

        for _ in range(self.max_tool_iterations):
            response = _with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_schema,
                    tool_choice="auto",
                    max_tokens=1500,
                )
            )
            choice = response.choices[0]
            msg = choice.message

            if msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = self.tools.dispatch(tc.function.name, args)
                    tool_log.append(ToolCallLog(tool=tc.function.name, input=args, output=result))
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
                continue

            return _parse_verdict_json(msg.content or ""), tool_log

        return (
            {
                "risk_score": 50,
                "risk_level": "medium",
                "flags": ["agent_incomplete_investigation"],
                "explanation": "Agent did not converge within the tool-call budget; flagged for manual review.",
                "recommended_action": "manual_review",
            },
            tool_log,
        )

    # ---- Gemini (google-genai tool-calling) ------------------------------------

    def _gemini_tool_schema(self):
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["input_schema"],
            )
            for t in TOOL_DEFINITIONS
        ]
        return [types.Tool(function_declarations=declarations)]

    def _run_gemini(self, user_message: str) -> tuple[dict, list[ToolCallLog]]:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=self._gemini_tool_schema(),
            max_output_tokens=1500,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_message)])]
        tool_log: list[ToolCallLog] = []

        for _ in range(self.max_tool_iterations):
            response = _with_retry(
                lambda: self.client.models.generate_content(model=self.model, contents=contents, config=config)
            )
            function_calls = response.function_calls or []

            if function_calls:
                contents.append(response.candidates[0].content)  # model's turn (with function_call parts)
                response_parts = []
                for fc in function_calls:
                    args = dict(fc.args) if fc.args else {}
                    result = self.tools.dispatch(fc.name, args)
                    tool_log.append(ToolCallLog(tool=fc.name, input=args, output=result))
                    response_parts.append(types.Part.from_function_response(name=fc.name, response=result))
                contents.append(types.Content(role="user", parts=response_parts))
                continue

            return _parse_verdict_json(response.text or ""), tool_log

        return (
            {
                "risk_score": 50,
                "risk_level": "medium",
                "flags": ["agent_incomplete_investigation"],
                "explanation": "Agent did not converge within the tool-call budget; flagged for manual review.",
                "recommended_action": "manual_review",
            },
            tool_log,
        )