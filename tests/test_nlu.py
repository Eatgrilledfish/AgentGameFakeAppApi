from app.agent.nlu import RuleBasedNLU
from app.schemas import CaseType, SessionState


def _state() -> SessionState:
    return SessionState(session_id="s1", user_id="u1", case_type=CaseType.single)


def test_extract_budget_subway_and_commute() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("预算1.2w以内，近地铁，通勤西二旗30分钟内，两居", _state(), CaseType.single)

    assert q.hard.budget_max == 12000
    assert q.hard.max_subway_dist == 800
    assert q.soft.prioritize_subway_distance is True
    assert q.hard.max_commute_min == 30
    assert q.hard.layout is not None


def test_extract_action_intent_and_platform() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("帮我把HF_2001在链家下架", _state(), CaseType.single)

    assert q.intent.value == "offline"
    assert q.hard.house_id == "HF_2001"
    assert q.hard.listing_platform is not None


def test_nlu_does_not_depend_on_case_type_for_search_clarify() -> None:
    nlu = RuleBasedNLU()
    q_single = nlu.parse("帮我找房", _state(), CaseType.single)
    q_multi = nlu.parse("帮我找房", _state(), CaseType.multi)

    assert q_single.clarify_questions == []
    assert q_multi.clarify_questions == []


def test_extract_subway_distance_without_priority_keyword() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("地铁500米内的两居", _state(), CaseType.single)

    assert q.hard.max_subway_dist == 500
    assert q.soft.prioritize_subway_distance is False


def test_do_not_parse_distance_as_budget() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("车公庄站 500 米内的两居", _state(), CaseType.single)

    assert q.hard.landmark_name == "车公庄"
    assert q.hard.max_subway_dist == 500
    assert q.hard.budget_max is None


def test_detect_listing_query_intent() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("HF_4这套在安居客、链家、58同城上分别多少钱？", _state(), CaseType.single)

    assert q.intent.value == "listings"
    assert q.hard.house_id == "HF_4"


def test_detect_house_detail_intent_without_triggering_rent() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("这套可以租吗", _state(), CaseType.single)

    assert q.intent.value == "rent_check"


def test_detect_compare_intent_from_bijia_phrase() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("这三套你帮我比价，哪个更合适？", _state(), CaseType.multi)

    assert q.intent.value == "compare"


def test_extract_utilities_and_typo_layout() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("通州两局商水商电房源有没有？", _state(), CaseType.single)

    assert q.intent.value == "search"
    assert q.hard.district == "通州"
    assert q.hard.layout == "两居"
    assert q.hard.utilities_type == "商水商电"


def test_extract_decoration_normalizes_jingzhuangxiu_to_jingzhuang() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("我想找精装修的两居", _state(), CaseType.single)

    assert q.soft.decoration == "精装"


def test_business_area_like_wangjing_goes_to_area_not_district() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("我想在望京租一套两居室，预算8000以内，有电梯", _state(), CaseType.single)

    assert q.hard.area == "望京"
    assert q.hard.district is None


def test_detect_rent_and_terminate_from_short_phrases() -> None:
    nlu = RuleBasedNLU()
    q1 = nlu.parse("就租第一套吧。", _state(), CaseType.single)
    q2 = nlu.parse("算了不租了，帮我退掉吧。", _state(), CaseType.single)

    assert q1.intent.value == "rent"
    assert q2.intent.value == "terminate"


def test_extract_fee_included_preference_into_tags() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("我希望网费包含在房租里，不要网费另付。", _state(), CaseType.single)

    assert "包宽带" in q.soft.preferred_tags or "免宽带费" in q.soft.preferred_tags
    assert "网费另付" in q.soft.avoid_tags
