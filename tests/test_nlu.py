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


def test_multi_case_generates_clarify_questions() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("我想租房", _state(), CaseType.multi)

    assert len(q.clarify_questions) >= 1


def test_extract_subway_distance_without_priority_keyword() -> None:
    nlu = RuleBasedNLU()
    q = nlu.parse("地铁500米内的两居", _state(), CaseType.single)

    assert q.hard.max_subway_dist == 500
    assert q.soft.prioritize_subway_distance is False
