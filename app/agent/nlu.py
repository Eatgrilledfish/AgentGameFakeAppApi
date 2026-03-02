from __future__ import annotations

import re

from app.schemas import CaseType, HardConstraints, IntentType, Platform, SessionState, SoftPreferences, StructuredQuery

_DISTRICTS = ["海淀", "朝阳", "通州", "昌平", "大兴", "房山", "西城", "丰台", "顺义", "东城"]
_PLATFORMS = {
    "链家": Platform.lianjia,
    "安居客": Platform.anjuke,
    "58": Platform.wuba,
    "58同城": Platform.wuba,
}
_RENT_TYPE_KEYWORDS = {"合租": "合租", "单间": "合租", "整租": "整租"}
_LAYOUT_PATTERN = re.compile(r"([一二两三四五六七八九1-9](?:居|室))(?:([0-9一二三四])厅)?(?:([0-9一二三四])卫)?")
_DATE_PATTERN = re.compile(r"(20\d{2}-\d{1,2}-\d{1,2})")
_HOUSE_ID_PATTERN = re.compile(r"([A-Z]{2,4}_?\d{3,8})")


class RuleBasedNLU:
    def parse(self, text: str, state: SessionState, case_type: CaseType) -> StructuredQuery:
        hard = HardConstraints()
        soft = SoftPreferences()

        lowered = text.lower()
        intent = self._detect_intent(text, lowered)

        self._extract_house_and_platform(text, hard)
        self._extract_budget(text, hard)
        self._extract_district(text, hard)
        self._extract_rent_type(text, hard)
        self._extract_layout(text, hard)
        self._extract_area(text, hard)
        self._extract_subway(text, lowered, hard, soft)
        self._extract_commute(text, hard)
        self._extract_move_in_date(text, hard)
        self._extract_landmark_or_community(text, hard)
        self._extract_soft_preferences(text, soft)

        if intent in {IntentType.rent, IntentType.terminate, IntentType.offline}:
            questions = self._build_action_questions(hard)
        elif case_type == CaseType.multi:
            questions = self._build_clarify_questions(hard)
        else:
            questions = []

        confidence = 0.4 + 0.1 * sum(
            1
            for value in [
                hard.budget_max,
                hard.district,
                hard.community,
                hard.landmark_name,
                hard.rent_type,
                hard.max_subway_dist,
                hard.max_commute_min,
            ]
            if value is not None
        )

        return StructuredQuery(
            intent=intent,
            hard=hard,
            soft=soft,
            clarify_questions=questions,
            confidence=min(0.95, confidence),
        )

    def _detect_intent(self, text: str, lowered: str) -> IntentType:
        if any(word in text for word in ["退租", "恢复可租"]):
            return IntentType.terminate
        if "下架" in text:
            return IntentType.offline
        if any(word in text for word in ["租这个", "租这套", "帮我租", "我要租", "租房"]):
            return IntentType.rent
        if any(word in text for word in ["商场", "商超", "公园", "配套"]) and any(
            word in text for word in ["附近", "周边", "有没有"]
        ):
            return IntentType.amenities
        if any(word in text for word in ["比较", "对比"]):
            return IntentType.compare
        if any(word in lowered for word in ["你好", "hello", "hi", "在吗"]) and not any(
            w in text for w in ["预算", "地铁", "通勤", "小区", "房"]
        ):
            return IntentType.chat
        return IntentType.search

    def _extract_house_and_platform(self, text: str, hard: HardConstraints) -> None:
        m = _HOUSE_ID_PATTERN.search(text)
        if m:
            hard.house_id = m.group(1)
        for key, platform in _PLATFORMS.items():
            if key in text:
                hard.listing_platform = platform
                break

    def _extract_budget(self, text: str, hard: HardConstraints) -> None:
        values = _extract_money_values(text)
        if not values:
            return
        if len(values) >= 2 and any(token in text for token in ["到", "-", "~", "至"]):
            hard.budget_min = min(values)
            hard.budget_max = max(values)
        else:
            hard.budget_max = max(values)

    def _extract_district(self, text: str, hard: HardConstraints) -> None:
        for district in _DISTRICTS:
            if district in text:
                hard.district = district
                return

    def _extract_rent_type(self, text: str, hard: HardConstraints) -> None:
        for key, value in _RENT_TYPE_KEYWORDS.items():
            if key in text:
                hard.rent_type = value
                return

    def _extract_layout(self, text: str, hard: HardConstraints) -> None:
        m = _LAYOUT_PATTERN.search(text)
        if not m:
            return
        base = m.group(1)
        hall = m.group(2)
        bath = m.group(3)
        parts = [base]
        if hall:
            parts.append(f"{hall}厅")
        if bath:
            parts.append(f"{bath}卫")
        hard.layout = "".join(parts)

    def _extract_area(self, text: str, hard: HardConstraints) -> None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:平米|平方米|平|㎡)", text)
        if m:
            hard.area_min = float(m.group(1))

    def _extract_subway(self, text: str, lowered: str, hard: HardConstraints, soft: SoftPreferences) -> None:
        if any(keyword in text for keyword in ["近地铁", "离地铁近", "靠近地铁"]):
            hard.max_subway_dist = 800
            soft.prioritize_subway_distance = True
        if "地铁可达" in text:
            hard.max_subway_dist = 1000

        for match in re.finditer(r"(\d{2,4})\s*米", text):
            value = int(match.group(1))
            near_context = text[max(0, match.start() - 6) : match.end() + 6]
            if any(word in near_context for word in ["地铁", "站", "步行"]):
                hard.max_subway_dist = value
                return

        if "subway" in lowered and hard.max_subway_dist is None:
            hard.max_subway_dist = 1000

    def _extract_commute(self, text: str, hard: HardConstraints) -> None:
        for m in re.finditer(r"(\d{1,3})\s*分(?:钟)?", text):
            context = text[max(0, m.start() - 8) : m.end() + 8]
            if any(k in context for k in ["通勤", "西二旗", "到公司", "上班"]):
                hard.max_commute_min = int(m.group(1))
                return

    def _extract_move_in_date(self, text: str, hard: HardConstraints) -> None:
        m = _DATE_PATTERN.search(text)
        if m:
            hard.move_in_date = m.group(1)

    def _extract_landmark_or_community(self, text: str, hard: HardConstraints) -> None:
        m_business_area = re.search(r"([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,20})(?:商圈|片区)", text)
        if m_business_area:
            hard.area = m_business_area.group(1)

        named_anchors = ["西二旗", "国贸", "望京", "上地", "中关村", "三里屯", "车公庄"]
        for name in named_anchors:
            if name in text:
                hard.landmark_name = name
                return

        m_community = re.search(r"(?:小区|住在|想在)([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,20})", text)
        if m_community:
            candidate = m_community.group(1)
            if not any(d in candidate for d in _DISTRICTS):
                hard.community = candidate

    def _extract_soft_preferences(self, text: str, soft: SoftPreferences) -> None:
        for deco in ["豪华", "精装", "简装", "毛坯", "空房"]:
            if deco in text:
                soft.decoration = deco
                break

        if "电梯" in text:
            if any(word in text for word in ["不要电梯", "无电梯"]):
                soft.elevator = False
            else:
                soft.elevator = True

        for orientation in ["朝南", "朝北", "朝东", "朝西", "南北", "东西"]:
            if orientation in text:
                soft.orientation = orientation
                break

        for noise in ["安静", "临街", "吵闹", "中等"]:
            if noise in text:
                soft.noise_preference = noise
                break

        if any(k in text for k in ["商场", "商超"]):
            soft.amenities.append("商超")
        if "公园" in text:
            soft.amenities.append("公园")

        if any(k in text for k in ["性价比", "划算", "便宜又好"]):
            soft.value_for_money = True

    def _build_clarify_questions(self, hard: HardConstraints) -> list[str]:
        questions: list[str] = []
        if hard.budget_max is None:
            questions.append("你的预算上限大概是多少（元/月）？")
        if hard.district is None and hard.community is None and hard.landmark_name is None:
            questions.append("你更倾向哪个区域、小区或地标附近？")
        if hard.rent_type is None:
            questions.append("更偏好整租还是合租？")
        return questions[:2]

    def _build_action_questions(self, hard: HardConstraints) -> list[str]:
        questions: list[str] = []
        if not hard.house_id:
            questions.append("请提供要操作的房源 house_id（例如 HF_2001）。")
        if not hard.listing_platform:
            questions.append("请提供挂牌平台（链家/安居客/58同城）。")
        return questions


def _extract_money_values(text: str) -> list[int]:
    values: list[int] = []

    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([wW万kK千])", text):
        base = float(m.group(1))
        unit = m.group(2).lower()
        if unit in {"w", "万"}:
            values.append(int(base * 10000))
        elif unit in {"k", "千"}:
            values.append(int(base * 1000))

    for m in re.finditer(r"(?<!\d)(\d{3,6})(?!\d)", text):
        values.append(int(m.group(1)))

    if values:
        return sorted(set(values))

    m_cn = re.search(r"([一二三四五六七八九十]+)万([一二三四五六七八九])?", text)
    if m_cn:
        major = _cn_to_num(m_cn.group(1)) * 10000
        minor = _cn_to_num(m_cn.group(2) or "零") * 1000
        return [major + minor]

    return []


def _cn_to_num(text: str) -> int:
    table = {
        "零": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if not text:
        return 0
    if text == "十":
        return 10
    if len(text) == 2 and text[0] == "十":
        return 10 + table.get(text[1], 0)
    if len(text) == 2 and text[1] == "十":
        return table.get(text[0], 0) * 10
    if len(text) == 3 and text[1] == "十":
        return table.get(text[0], 0) * 10 + table.get(text[2], 0)
    return table.get(text, 0)
