from __future__ import annotations

from app.schemas import CaseType, HouseViewModel, InvokeResponse, StructuredQuery


class OutputFormatter:
    def render(
        self,
        *,
        case_type: CaseType,
        query: StructuredQuery,
        top_houses: list[HouseViewModel],
        clarify_questions: list[str] | None = None,
        debug: dict | None = None,
    ) -> InvokeResponse:
        if clarify_questions:
            text = "为了更准确推荐，我需要再确认两点：\n" + "\n".join(
                f"{idx + 1}. {q}" for idx, q in enumerate(clarify_questions)
            )
            return InvokeResponse(text=text, clarify_questions=clarify_questions, candidates=[], debug=debug or {})

        if not top_houses:
            text = "当前条件下没有检索到合适房源。建议放宽预算、地铁距离或区域限制后再试。"
            return InvokeResponse(text=text, candidates=[], debug=debug or {})

        conditions = self._build_conditions(query)
        condition_text = "/".join(conditions) if conditions else "当前条件"
        text = f"查找到以下符合您要求的房源（查询条件：{condition_text}）"

        if case_type == CaseType.multi:
            text += "。如需进一步缩小到 1-2 套，可继续告诉我你的优先级。"

        return InvokeResponse(text=text, candidates=top_houses, debug=debug or {})

    def render_action_result(self, action: str, result: dict) -> InvokeResponse:
        messages = {
            "rent": "已提交租房操作。",
            "terminate": "已提交退租操作。",
            "offline": "已提交下架操作。",
        }
        return InvokeResponse(text=messages.get(action, "操作已提交。"), candidates=[], debug={"action_result": result})

    def _build_conditions(self, query: StructuredQuery) -> list[str]:
        conditions: list[str] = []

        if query.hard.district:
            district = query.hard.district
            conditions.append(district if district.endswith("区") else f"{district}区")

        if query.hard.layout:
            layout = query.hard.layout
            if "居" in layout and not layout.endswith("室"):
                layout = f"{layout}室"
            conditions.append(layout)

        if query.soft.prioritize_subway_distance:
            conditions.append("地铁距离")
        elif query.hard.max_subway_dist is not None:
            conditions.append(f"地铁{query.hard.max_subway_dist}米内")

        return conditions
