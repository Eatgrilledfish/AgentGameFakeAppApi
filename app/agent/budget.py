from __future__ import annotations

import math

from app.schemas import BudgetState


class BudgetManager:
    @staticmethod
    def estimate_slices(n_tokens: int) -> int:
        return int(math.ceil(1 + max(0, n_tokens - 1000) * 0.3))

    @staticmethod
    def can_use_llm(budget: BudgetState, estimated_prompt_tokens: int) -> bool:
        expected = BudgetManager.estimate_slices(estimated_prompt_tokens)
        return budget.used_slices + expected <= budget.limit_slices

    @staticmethod
    def record_llm_usage(budget: BudgetState, consumed_tokens: int) -> None:
        budget.llm_calls += 1
        budget.used_tokens += max(0, consumed_tokens)
        budget.used_slices += BudgetManager.estimate_slices(consumed_tokens)
