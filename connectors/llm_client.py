"""LLM client for probability estimation via Claude Haiku.

Uses the sync Anthropic SDK via run_in_executor to avoid blocking the event loop.
Graceful failure: returns None on any error → strategy skips the market.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Haiku pricing (per 1M tokens)
_INPUT_COST_PER_M = 0.80   # $0.80 / 1M input tokens
_OUTPUT_COST_PER_M = 4.00  # $4.00 / 1M output tokens

_SYSTEM_PROMPT = (
    "You are a calibrated prediction market analyst. "
    "Given a market question and recent news headlines, "
    "estimate the true probability of the outcome.\n\n"
    "Rules:\n"
    "- Base your estimate on the NEWS PROVIDED, not on the current market price.\n"
    "- If the headlines are irrelevant or insufficient, set confidence below 0.50.\n"
    "- Meme, joke, or unfalsifiable markets (e.g. religious prophecy, celebrity stunts) "
    "should get confidence 0.10 regardless of how obvious the answer seems.\n"
    "- confidence reflects how much the NEWS informs your estimate "
    "(0.1 = no useful info, 0.9 = strong evidence).\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"probability": <float 0-1>, "confidence": <float 0-1>, "reasoning": "<1-2 sentences>"}'
)


class LLMClient:
    def __init__(self, config: dict):
        self.model = config.get("model", "claude-haiku-4-5-20251001")
        self.max_cost_per_day = config.get("max_cost_per_day", 0.50)
        self._daily_cost = 0.0
        self._cost_reset_date: str | None = None
        self._client = None  # lazy init

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        return self._client

    def _reset_daily_cost_if_needed(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._cost_reset_date != today:
            self._daily_cost = 0.0
            self._cost_reset_date = today

    async def estimate_probability(
        self,
        market_question: str,
        outcome_name: str,
        current_price: float,
        headlines: list[str],
    ) -> dict | None:
        """Estimate probability via Claude Haiku.

        Returns {"probability": float, "confidence": float, "reasoning": str, "cost_usd": float}
        or None on failure/budget exceeded.
        """
        self._reset_daily_cost_if_needed()

        if self._daily_cost >= self.max_cost_per_day:
            logger.warning("LLM daily budget exceeded ($%.3f / $%.2f)", self._daily_cost, self.max_cost_per_day)
            return None

        headlines_text = "\n".join(f"- {h}" for h in headlines[:10])
        user_msg = (
            f"Market question: {market_question}\n"
            f"Outcome to estimate: {outcome_name}\n\n"
            f"Recent news:\n{headlines_text}\n\n"
            f"Estimate the true probability of \"{outcome_name}\"."
        )

        try:
            client = self._ensure_client()
            loop = asyncio.get_running_loop()

            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: client.messages.create(
                        model=self.model,
                        max_tokens=200,
                        system=_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": user_msg}],
                    ),
                ),
                timeout=30,
            )

            # Parse response
            text = resp.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(text)

            prob = float(result["probability"])
            conf = float(result["confidence"])
            reasoning = str(result.get("reasoning", ""))

            if not (0 <= prob <= 1) or not (0 <= conf <= 1):
                logger.warning("LLM returned out-of-range values: prob=%.2f conf=%.2f", prob, conf)
                return None

            # Track cost
            input_tokens = resp.usage.input_tokens
            output_tokens = resp.usage.output_tokens
            cost = (input_tokens * _INPUT_COST_PER_M + output_tokens * _OUTPUT_COST_PER_M) / 1_000_000
            self._daily_cost += cost

            logger.info(
                "LLM estimate: prob=%.2f conf=%.2f cost=$%.4f (daily $%.3f) — %s",
                prob, conf, cost, self._daily_cost, market_question[:60],
            )

            return {
                "probability": prob,
                "confidence": conf,
                "reasoning": reasoning,
                "cost_usd": cost,
            }

        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON: %s", text[:200] if 'text' in dir() else "?")
            return None
        except Exception:
            logger.exception("LLM call failed")
            return None

    @property
    def daily_cost(self) -> float:
        self._reset_daily_cost_if_needed()
        return self._daily_cost
