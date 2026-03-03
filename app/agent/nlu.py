from __future__ import annotations

import re

from app.schemas import CaseType, HardConstraints, IntentType, Platform, SessionState, SoftPreferences, StructuredQuery

_PLATFORMS = {
    "链家": Platform.lianjia,
    "安居客": Platform.anjuke,
    "58": Platform.wuba,
    "58同城": Platform.wuba,
}
_RENT_TYPE_KEYWORDS = {"合租": "合租", "单间": "合租", "整租": "整租"}
_LAYOUT_PATTERN = re.compile(r"([一二两三四五六七八九1-9](?:居|室))(?:([0-9一二三四])厅)?(?:([0-9一二三四])卫)?")
_DATE_PATTERN = re.compile(r"(20\d{2}-\d{1,2}-\d{1,2})")
_HOUSE_ID_PATTERN = re.compile(r"([A-Z]{2,4}_?\d{1,8})", re.IGNORECASE)
_ADMIN_DIVISION_SUFFIXES = ("区", "县", "旗", "市", "州", "盟")
_ADMIN_DIVISION_LONG_SUFFIXES = ("新区", "开发区", "自治县", "自治州")
_ADMIN_DIVISION_PATTERN = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{1,20})(新区|开发区|自治县|自治州|区|县|旗|市|州|盟)")
_ADMIN_DIVISION_STOPWORDS = {"小区", "片区", "商圈", "区域", "地区", "社区", "园区", "校区", "学区", "站区"}
_LOCATION_PREFIXES = (
    "帮我找",
    "帮我",
    "给我找",
    "给我",
    "我想在",
    "我想",
    "想在",
    "想",
    "我要在",
    "我要",
    "在",
    "去",
    "到",
    "找",
    "换到",
    "换去",
    "换",
    "搬到",
)
_BUSINESS_AREA_PATTERN = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,20})(?:商圈|片区)")
_STATION_PATTERN = re.compile(r"(?:在|离|到|去|靠近|近)?([\u4e00-\u9fa5A-Za-z0-9·\-]{2,16})站")
_AREA_CUE_PATTERN = re.compile(r"(?:在|去|到|想在|希望在|换到|换去|搬到|住在)([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,20})(?:租|找|看|住|附近|一带)")
_COMMUNITY_PATTERN = re.compile(
    r"(?:小区|住在)([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,20}?)(?:小区|附近|一带|租|找|看|住|，|。|,|\.|\s|$)"
)
_PREFERRED_TAG_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("采光不好", "阴暗", "不朝阳"), ("采光好", "朝南")),
    (("房间小", "太小", "空间小", "拥挤"), ("大面积", "宽敞")),
    (("通勤时间长", "通勤太长", "上班太远", "通勤不方便"), ("近地铁", "地铁口", "通勤便利")),
    (("安静",), ("安静", "不临街")),
    (("性价比", "划算", "便宜又好"), ("高性价比",)),
    (("可养宠物", "养猫", "养狗"), ("可养宠物", "可养猫", "可养狗")),
)
_AVOID_TAG_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("临街", "太吵", "吵闹", "噪音", "噪声"), ("临街", "吵闹")),
    (("不要中介费", "不想中介费", "中介费太高", "免中介"), ("收中介费", "中介费一月租", "中介费半月租")),
    (("不要物业费", "免物业费"), ("物业费另付",)),
)


class RuleBasedNLU:
    def parse(self, text: str, state: SessionState, case_type: CaseType) -> StructuredQuery:
        text = _normalize_typos(text)
        hard = HardConstraints()
        soft = SoftPreferences()
        _ = state
        _ = case_type

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
        self._extract_utilities(text, hard)
        self._extract_move_in_date(text, hard)
        self._extract_landmark_or_community(text, hard)
        self._extract_soft_preferences(text, soft)

        if intent in {IntentType.rent, IntentType.terminate, IntentType.offline}:
            questions = self._build_action_questions(hard)
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
        house_ref = bool(_HOUSE_ID_PATTERN.search(text)) or any(
            word in text
            for word in ["这套", "这一套", "这个房", "它", "第一套", "第二套", "第三套", "最开始", "最初", "上一套", "刚才那套"]
        )
        if any(word in text for word in ["退租", "恢复可租", "退掉", "取消租", "不租了"]):
            return IntentType.terminate
        if "下架" in text:
            return IntentType.offline
        if any(word in text for word in ["租这个", "租这套", "租这一套", "帮我租", "我要租", "我想租", "办理租房"]):
            return IntentType.rent
        if "租房" in text and any(word in text for word in ["帮我", "办理", "我要", "我想"]):
            return IntentType.rent
        if (
            "租" in text
            and house_ref
            and not any(word in text for word in ["可租吗", "可以租吗", "能租吗", "能不能租", "租金", "多少钱", "价格"])
        ):
            return IntentType.rent
        has_platform = any(word in text for word in ["安居客", "链家", "58同城", "58"])
        if (house_ref or has_platform) and any(word in text for word in ["多少钱", "价格", "报价", "挂牌"]) and (
            has_platform or "分别" in text or "各平台" in text
        ):
            return IntentType.listings
        if house_ref and any(
            word in text for word in ["详情", "详细", "离地铁", "地铁多远", "多远", "可租吗", "可以租吗", "能租吗", "详细情况"]
        ):
            return IntentType.house_detail
        if any(word in text for word in ["商场", "商超", "公园", "配套"]) and any(
            word in text for word in ["附近", "周边", "有没有"]
        ):
            return IntentType.amenities
        if any(word in text for word in ["比较", "对比", "比价", "哪个好", "怎么选", "哪个更适合"]):
            return IntentType.compare
        if any(word in lowered for word in ["你好", "hello", "hi", "在吗"]) and not any(
            w in text for w in ["预算", "地铁", "通勤", "小区", "房"]
        ):
            return IntentType.chat
        return IntentType.search

    def _extract_house_and_platform(self, text: str, hard: HardConstraints) -> None:
        m = _HOUSE_ID_PATTERN.search(text)
        if m:
            hard.house_id = m.group(1).upper()
        for key, platform in _PLATFORMS.items():
            if key in text:
                hard.listing_platform = platform
                break

    def _extract_budget(self, text: str, hard: HardConstraints) -> None:
        values = _extract_money_values(text)
        if not values:
            return
        if not _has_budget_intent(text):
            values = [value for value in values if not _is_distance_like_number(text, value)]
        if not values:
            return
        if len(values) >= 2 and any(token in text for token in ["到", "-", "~", "至"]):
            hard.budget_min = min(values)
            hard.budget_max = max(values)
        else:
            hard.budget_max = max(values)

    def _extract_district(self, text: str, hard: HardConstraints) -> None:
        best_token: str | None = None
        best_score = -1
        for match in _ADMIN_DIVISION_PATTERN.finditer(text):
            raw = f"{match.group(1)}{match.group(2)}"
            normalized = _normalize_admin_division(raw)
            if not normalized:
                continue
            suffix = match.group(2)
            score = 3 if suffix in {"区", "县", "旗"} else 2 if suffix in _ADMIN_DIVISION_LONG_SUFFIXES else 1
            if score > best_score:
                best_token = normalized
                best_score = score
        if best_token:
            hard.district = best_token

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

    def _extract_utilities(self, text: str, hard: HardConstraints) -> None:
        if "商水商电" in text:
            hard.utilities_type = "商水商电"
            return
        if "民水民电" in text:
            hard.utilities_type = "民水民电"
            return

    def _extract_move_in_date(self, text: str, hard: HardConstraints) -> None:
        m = _DATE_PATTERN.search(text)
        if m:
            hard.move_in_date = m.group(1)

    def _extract_landmark_or_community(self, text: str, hard: HardConstraints) -> None:
        m_business_area = _BUSINESS_AREA_PATTERN.search(text)
        if m_business_area:
            hard.area = m_business_area.group(1)

        m_station = _STATION_PATTERN.search(text)
        if m_station:
            station_name = m_station.group(1).strip()
            if station_name:
                hard.landmark_name = station_name
                return

        if hard.area is None:
            m_area = _AREA_CUE_PATTERN.search(text)
            if m_area:
                area_candidate = m_area.group(1).strip()
                if area_candidate and not _contains_admin_division_token(area_candidate):
                    hard.area = area_candidate
                    # Keep a lightweight anchor so downstream planner can resolve nearby lookup if needed.
                    if hard.landmark_name is None:
                        hard.landmark_name = area_candidate

        m_community = _COMMUNITY_PATTERN.search(text)
        if m_community:
            candidate = m_community.group(1).strip("，。,. ")
            if (
                candidate
                and not _contains_admin_division_token(candidate)
                and not re.search(r"(预算|房源|通勤|地铁|电梯)", candidate)
                and not re.search(r"(一居|两居|三居|四居|\d+居|\d+室)", candidate)
            ):
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

        preferred = _derive_tags_from_rules(text, _PREFERRED_TAG_RULES)
        avoid = _derive_tags_from_rules(text, _AVOID_TAG_RULES)

        if soft.elevator is True:
            preferred.extend(["有电梯", "电梯房"])
            avoid.extend(["无电梯", "步梯", "没电梯"])
        elif soft.elevator is False:
            preferred.extend(["无电梯", "步梯"])
            avoid.extend(["有电梯", "电梯房"])

        if soft.orientation:
            preferred.append(soft.orientation)
            orient = soft.orientation.replace("朝", "").strip()
            if orient:
                preferred.append(orient)

        if soft.noise_preference == "安静":
            preferred.extend(["安静", "不临街"])
            avoid.extend(["临街", "吵闹"])
        elif soft.noise_preference in {"临街", "吵闹"}:
            preferred.extend(["临街", "吵闹"])

        for tag in preferred:
            if tag and tag not in soft.preferred_tags:
                soft.preferred_tags.append(tag)
        for tag in avoid:
            if tag and tag not in soft.avoid_tags:
                soft.avoid_tags.append(tag)

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


def _has_budget_intent(text: str) -> bool:
    signals = ["预算", "租金", "月租", "价格", "价位", "元", "块", "w", "万", "k", "千"]
    return any(token in text for token in signals)


def _is_distance_like_number(text: str, value: int) -> bool:
    for m in re.finditer(rf"(?<!\d){value}(?!\d)", text):
        context = text[max(0, m.start() - 4) : m.end() + 4]
        if any(unit in context for unit in ["米", "分钟", "分", "步行"]):
            return True
    return False


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


def _normalize_typos(text: str) -> str:
    normalized = text
    normalized = normalized.replace("两局", "两居")
    normalized = normalized.replace("一局", "一居")
    normalized = normalized.replace("三局", "三居")
    normalized = normalized.replace("四局", "四居")
    return normalized


def _normalize_admin_division(raw: str) -> str | None:
    cleaned = raw.strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"[，,。；;：:\s]+", "", cleaned)
    cleaned = _strip_location_prefix(cleaned)
    if len(cleaned) <= 1:
        return None
    if cleaned in _ADMIN_DIVISION_STOPWORDS:
        return None
    if any(marker in cleaned for marker in _ADMIN_DIVISION_STOPWORDS):
        return None
    if cleaned.endswith("区") and cleaned in {"小区", "片区", "园区", "校区", "学区", "站区"}:
        return None
    if cleaned.endswith(_ADMIN_DIVISION_LONG_SUFFIXES):
        return cleaned
    if cleaned.endswith("区"):
        token = cleaned[:-1]
        return token if token else None
    if cleaned.endswith(_ADMIN_DIVISION_SUFFIXES):
        return cleaned
    return None


def _contains_admin_division_token(text: str) -> bool:
    for match in _ADMIN_DIVISION_PATTERN.finditer(text):
        raw = f"{match.group(1)}{match.group(2)}"
        if _normalize_admin_division(raw):
            return True
    return False


def _strip_location_prefix(text: str) -> str:
    stripped = text
    changed = True
    while changed and stripped:
        changed = False
        for prefix in _LOCATION_PREFIXES:
            if stripped.startswith(prefix) and len(stripped) > len(prefix) + 1:
                stripped = stripped[len(prefix) :]
                changed = True
                break
    return stripped


def _derive_tags_from_rules(
    text: str,
    rules: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
) -> list[str]:
    found: list[str] = []
    for triggers, tags in rules:
        if any(trigger in text for trigger in triggers):
            for tag in tags:
                if tag not in found:
                    found.append(tag)
    return found
