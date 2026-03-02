from app.agent.formatter import OutputFormatter
from app.schemas import CaseType, HardConstraints, HouseViewModel, SoftPreferences, StructuredQuery


def test_formatter_returns_compact_message_and_candidates() -> None:
    formatter = OutputFormatter()
    query = StructuredQuery(
        hard=HardConstraints(district="西城", layout="一居", max_subway_dist=800),
        soft=SoftPreferences(prioritize_subway_distance=True),
    )
    response = formatter.render(
        case_type=CaseType.single,
        query=query,
        top_houses=[HouseViewModel(house_id="HF_1"), HouseViewModel(house_id="HF_2")],
    )

    assert response.text == "查找到以下符合您要求的房源（查询条件：西城区/一居室/地铁距离）"
    assert [item.house_id for item in response.candidates] == ["HF_1", "HF_2"]
