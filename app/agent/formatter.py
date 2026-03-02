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

        lines = [f"为你筛选到 {len(top_houses)} 套候选房源（最多展示 5 套）："]
        for idx, house in enumerate(top_houses, start=1):
            lines.append(
                f"{idx}. {house.house_id} | {house.district or '-'}·{house.community or '-'} | "
                f"{house.layout or '-'} | {house.area or '-'}㎡ | {house.rent or '-'} 元/月"
            )
            if house.pros:
                lines.append(f"   优点：{'；'.join(house.pros[:2])}")
            if house.cons:
                lines.append(f"   注意：{'；'.join(house.cons[:1])}")

        if case_type == CaseType.multi:
            lines.append("如果你要我进一步缩小到 1-2 套，我可以按你最看重的条件继续重排。")

        return InvokeResponse(text="\n".join(lines), candidates=top_houses, debug=debug or {})

    def render_action_result(self, action: str, result: dict) -> InvokeResponse:
        messages = {
            "rent": "已提交租房操作。",
            "terminate": "已提交退租操作。",
            "offline": "已提交下架操作。",
        }
        return InvokeResponse(text=messages.get(action, "操作已提交。"), candidates=[], debug={"action_result": result})
