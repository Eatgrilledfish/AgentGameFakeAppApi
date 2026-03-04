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
            text += "。已按通勤、租金、地铁与居住条件综合排序（最多5套），如需我继续对比并给出决策建议可以直接说。"

        semantic_fusion = (debug or {}).get("semantic_fusion") if isinstance(debug, dict) else None
        if isinstance(semantic_fusion, dict):
            must_confirm = semantic_fusion.get("must_confirm")
            if isinstance(must_confirm, dict) and must_confirm:
                snippets = [f"{house_id}（{reason}）" for house_id, reason in must_confirm.items()][:3]
                if snippets:
                    text += " 需确认：" + "；".join(snippets) + "。"

            decisions = semantic_fusion.get("decisions")
            if isinstance(decisions, dict) and decisions:
                hard_rejected = [
                    f"{house_id}（{item.get('reason', '与偏好冲突')}）"
                    for house_id, item in decisions.items()
                    if isinstance(item, dict) and item.get("action") == "rejected_drop"
                ][:3]
                risk_rejected = [
                    f"{house_id}（{item.get('reason', '存在偏好风险')}）"
                    for house_id, item in decisions.items()
                    if isinstance(item, dict) and item.get("action") == "rejected_penalty"
                ][:3]
                if hard_rejected:
                    text += " 已排除（明确冲突）：" + "；".join(hard_rejected) + "。"
                if risk_rejected:
                    text += " 存在风险（建议确认）：" + "；".join(risk_rejected) + "。"

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
