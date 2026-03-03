from __future__ import annotations

import asyncio
from dataclasses import dataclass
from statistics import pstdev

from app.clients.houses import HousesClient
from app.infra.cache import CacheManager
from app.schemas import HouseLite, HouseViewModel, Listing, NearbyLandmark, StructuredQuery
from app.settings import RankingWeights

_TAG_MATCH_WEIGHT = 8.0


@dataclass(slots=True)
class RankedHouse:
    house: HouseLite
    score: float
    listings: list[Listing]
    amenities: dict[str, list[NearbyLandmark]]


class Ranker:
    def __init__(
        self,
        houses_client: HousesClient,
        weights: RankingWeights,
        enrich_concurrency: int = 8,
        *,
        cache: CacheManager | None = None,
        listing_top_n: int = 20,
        amenities_top_n: int = 5,
    ) -> None:
        self.houses_client = houses_client
        self.weights = weights
        self.enrich_concurrency = enrich_concurrency
        self.cache = cache
        self.listing_top_n = listing_top_n
        self.amenities_top_n = amenities_top_n

    def hard_filter(self, house: HouseLite, query: StructuredQuery) -> bool:
        h = query.hard
        if house.status and not _is_rentable_status(house.status):
            return False
        if h.budget_max is not None and house.rent is not None and house.rent > h.budget_max:
            return False
        if h.budget_min is not None and house.rent is not None and house.rent < h.budget_min:
            return False
        if h.area_min is not None and house.area is not None and house.area < h.area_min:
            return False
        if h.max_subway_dist is not None and house.subway_distance is not None and house.subway_distance > h.max_subway_dist:
            return False
        if h.max_commute_min is not None and house.commute_to_xierqi_min is not None and house.commute_to_xierqi_min > h.max_commute_min:
            return False
        if h.district and house.district and h.district != house.district:
            return False
        if h.community and house.community and h.community != house.community:
            return False
        if h.rent_type and house.layout:
            if h.rent_type == "合租" and "单间" not in house.layout and "合租" not in "".join(house.tags):
                return False
            if h.rent_type == "整租" and any(k in house.layout for k in ["单间", "合租"]):
                return False
        if h.layout and house.layout and not _layout_matches(h.layout, house.layout):
            return False
        return True

    def _coarse_score(self, house: HouseLite, query: StructuredQuery) -> float:
        commute_score = _inverse_score(house.commute_to_xierqi_min, 5, 90)
        subway_score = _inverse_score(house.subway_distance, 200, 5000)
        rent_score = _rent_score(house.rent, query)
        layout_area_score = _layout_area_score(house, query)
        quality_score = _quality_score(house, query)

        total = (
            self.weights.commute * commute_score
            + self.weights.subway * subway_score
            + self.weights.rent * rent_score
            + self.weights.layout_area * layout_area_score
            + self.weights.quality * quality_score
        )
        return total

    async def rank_two_stage(self, candidates: list[HouseLite], query: StructuredQuery, max_output: int = 5) -> list[HouseViewModel]:
        filtered = [house for house in candidates if self.hard_filter(house, query)]
        relax_notes: list[str] = []
        if not filtered and query.hard.max_subway_dist == 800:
            original = query.hard.max_subway_dist
            query.hard.max_subway_dist = 1000
            filtered = [house for house in candidates if self.hard_filter(house, query)]
            query.hard.max_subway_dist = original
            if filtered:
                relax_notes.append("为避免无结果，已放宽近地铁条件到 1000 米")

        if not filtered and query.hard.budget_max is not None:
            original_budget = query.hard.budget_max
            query.hard.budget_max = int(original_budget * 1.1)
            filtered = [house for house in candidates if self.hard_filter(house, query)]
            query.hard.budget_max = original_budget
            if filtered:
                relax_notes.append("为避免无结果，已放宽预算上限 10%")

        if not filtered:
            return []

        if query.soft.prioritize_subway_distance or query.soft.prioritize_commute:
            coarse_sorted = sorted(filtered, key=lambda h: self._priority_sort_key(h, query))
        else:
            coarse_sorted = sorted(filtered, key=lambda h: self._coarse_score(h, query), reverse=True)
        top_n = coarse_sorted[: self.listing_top_n]
        enriched = await self._enrich_listings(top_n)

        top_fine = sorted(
            enriched,
            key=lambda item: self._fine_score(item.house, item.listings, query, amenities={}),
            reverse=True,
        )[:max_output]

        top_amenities = top_fine[: self.amenities_top_n]
        enriched_top = await self._enrich_amenities(top_amenities)
        combined = enriched_top + top_fine[self.amenities_top_n :]
        combined = sorted(
            combined,
            key=lambda item: self._fine_score(item.house, item.listings, query, amenities=item.amenities),
            reverse=True,
        )[:max_output]
        if query.soft.prioritize_subway_distance or query.soft.prioritize_commute:
            combined = sorted(
                combined,
                key=lambda item: self._priority_sort_key(
                    item.house,
                    query,
                    fallback_score=self._fine_score(item.house, item.listings, query, amenities=item.amenities),
                ),
            )
        views = [self._to_view_model(item, query, relax_notes=relax_notes) for item in combined]
        return views

    def _priority_sort_key(
        self,
        house: HouseLite,
        query: StructuredQuery,
        *,
        fallback_score: float | None = None,
    ) -> tuple[float, float, float, float]:
        score = fallback_score if fallback_score is not None else self._coarse_score(house, query)
        subway = house.subway_distance if house.subway_distance is not None else 10**9
        commute = house.commute_to_xierqi_min if house.commute_to_xierqi_min is not None else 10**9
        rent = house.rent if house.rent is not None else 10**9

        if query.soft.prioritize_commute and query.soft.prioritize_subway_distance:
            return (commute, subway, -score, rent)
        if query.soft.prioritize_commute:
            return (commute, -score, subway, rent)
        return (subway, -score, commute, rent)

    async def _enrich_listings(self, houses: list[HouseLite]) -> list[RankedHouse]:
        sem = asyncio.Semaphore(self.enrich_concurrency)

        async def worker(house: HouseLite) -> RankedHouse:
            if self.cache is not None and house.house_id in self.cache.house_listings:
                listings = self.cache.house_listings[house.house_id]
                return RankedHouse(house=house, score=0.0, listings=listings, amenities={})
            async with sem:
                listings_page = await self.houses_client.get_listings(house.house_id)
                listings = listings_page.get("items", [])
                if self.cache is not None:
                    self.cache.house_listings[house.house_id] = listings
                return RankedHouse(house=house, score=0.0, listings=listings, amenities={})

        return await asyncio.gather(*[worker(house) for house in houses])

    async def _enrich_amenities(self, ranked_houses: list[RankedHouse]) -> list[RankedHouse]:
        sem = asyncio.Semaphore(min(5, self.enrich_concurrency))

        async def load_one(item: RankedHouse) -> RankedHouse:
            community = item.house.community
            if not community:
                return item
            if self.cache is not None and community in self.cache.community_amenities:
                item.amenities = self.cache.community_amenities[community]
                return item
            async with sem:
                shops = await self.houses_client.nearby_landmarks(community=community, category="shopping", max_distance_m=3000)
                parks = await self.houses_client.nearby_landmarks(community=community, category="park", max_distance_m=3000)
            amenities = {"shopping": shops, "park": parks}
            item.amenities = amenities
            if self.cache is not None:
                self.cache.community_amenities[community] = amenities
            return item

        return await asyncio.gather(*[load_one(item) for item in ranked_houses])

    def _fine_score(
        self,
        house: HouseLite,
        listings: list[Listing],
        query: StructuredQuery,
        amenities: dict[str, list[NearbyLandmark]],
    ) -> float:
        base = self._coarse_score(house, query)
        listing_consistency = _listing_consistency_score(listings)
        amenities_score = _amenities_score(amenities, query)
        tag_match_score = _tag_preference_score(house, query)
        return (
            base
            + self.weights.listing_consistency * listing_consistency
            + self.weights.amenities * amenities_score
            + _TAG_MATCH_WEIGHT * tag_match_score
        )

    def _to_view_model(self, ranked: RankedHouse, query: StructuredQuery, relax_notes: list[str]) -> HouseViewModel:
        house = ranked.house
        listings = ranked.listings
        selected_platform = _choose_best_platform(house, listings)
        score = self._fine_score(house, listings, query, amenities=ranked.amenities)

        pros: list[str] = []
        cons: list[str] = []

        if house.commute_to_xierqi_min is not None:
            pros.append(f"通勤约 {house.commute_to_xierqi_min} 分钟")
        if house.subway_distance is not None:
            pros.append(f"距地铁约 {house.subway_distance} 米")
        if house.rent is not None:
            pros.append(f"月租 {house.rent} 元")

        if any(tag in (house.tags or []) for tag in ["农村房", "农村自建房", "临街"]):
            cons.append("存在潜在居住风险标签，建议实地核验")

        if listings:
            rents = [i.rent for i in listings if i.rent is not None]
            if len(rents) >= 2 and max(rents) - min(rents) >= 1500:
                cons.append("跨平台价格差异较大，建议核对房源状态")

        for note in relax_notes:
            cons.append(note)

        return HouseViewModel(
            house_id=house.house_id,
            listing_platform=selected_platform,
            rent=house.rent,
            layout=house.layout,
            area=house.area,
            district=house.district,
            community=house.community,
            nearest_subway=None,
            subway_distance=house.subway_distance,
            commute_to_xierqi_min=house.commute_to_xierqi_min,
            available_date=house.available_date,
            tags=house.tags,
            pros=pros[:4],
            cons=cons[:3],
            score=round(score, 2),
        )


def _inverse_score(value: int | float | None, low: float, high: float) -> float:
    if value is None:
        return 0.5
    clipped = min(max(float(value), low), high)
    return 1.0 - (clipped - low) / (high - low)


def _rent_score(rent: int | None, query: StructuredQuery) -> float:
    if rent is None:
        return 0.4
    budget_max = query.hard.budget_max
    budget_min = query.hard.budget_min
    if budget_max is not None:
        if rent > budget_max:
            return 0.0
        floor = budget_min if budget_min is not None else max(0, budget_max * 0.4)
        span = max(1.0, budget_max - floor)
        return 1.0 - (rent - floor) / span
    return _inverse_score(rent, 1000, 25000)


def _layout_area_score(house: HouseLite, query: StructuredQuery) -> float:
    score = 0.6
    if query.hard.layout and house.layout:
        score += 0.2 if _layout_matches(query.hard.layout, house.layout) else -0.15
    if query.hard.area_min is not None and house.area is not None:
        score += 0.2 if house.area >= query.hard.area_min else -0.2
    if query.soft.prefer_spacious:
        if house.area is None:
            score -= 0.05
        elif house.area >= 90:
            score += 0.15
        elif house.area >= 70:
            score += 0.08
        elif house.area < 45:
            score -= 0.15
    return max(0.0, min(1.0, score))


def _quality_score(house: HouseLite, query: StructuredQuery) -> float:
    score = 0.5
    if query.soft.decoration and house.decoration:
        score += 0.2 if query.soft.decoration in house.decoration else -0.1
    if query.soft.orientation and house.orientation:
        score += 0.15 if query.soft.orientation in house.orientation else -0.05
    if query.soft.elevator is True and house.elevator is True:
        score += 0.1
    if query.soft.noise_preference == "安静" and any(tag in house.tags for tag in ["临街", "吵闹"]):
        score -= 0.2
    if query.soft.value_for_money and any(tag in house.tags for tag in ["高性价比", "低价"]):
        score += 0.15
    return max(0.0, min(1.0, score))


def _listing_consistency_score(listings: list[Listing]) -> float:
    if len(listings) <= 1:
        return 0.6
    rents = [l.rent for l in listings if l.rent is not None]
    statuses = {l.status for l in listings if l.status}
    if not rents:
        return 0.5
    spread = pstdev(rents) if len(rents) > 1 else 0.0
    rent_score = max(0.0, 1.0 - min(spread, 3000) / 3000)
    status_score = 1.0 if len(statuses) <= 1 else 0.4
    return 0.7 * rent_score + 0.3 * status_score


def _choose_best_platform(house: HouseLite, listings: list[Listing]) -> str | None:
    if listings:
        rentable = [l for l in listings if l.status and _is_rentable_status(l.status)]
        pool = rentable or listings
        sorted_pool = sorted(pool, key=lambda l: (l.rent if l.rent is not None else 10**9))
        if sorted_pool and sorted_pool[0].listing_platform:
            return sorted_pool[0].listing_platform
    return house.listing_platform or "安居客"




def _layout_matches(expected_layout: str, actual_layout: str) -> bool:
    expected = _normalize_digits(expected_layout)
    actual = _normalize_digits(actual_layout)
    return expected[:1] in actual


def _normalize_digits(text: str) -> str:
    table = str.maketrans({"一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"})
    return text.translate(table)


def _is_rentable_status(status: str) -> bool:
    normalized = status.strip().lower()
    return normalized in {"可租", "available"}

def _amenities_score(amenities: dict[str, list[NearbyLandmark]], query: StructuredQuery) -> float:
    if not amenities:
        return 0.0

    preferred = set(query.soft.amenities)
    if not preferred:
        return 0.5

    shopping_count = len(amenities.get("shopping", []))
    park_count = len(amenities.get("park", []))
    score = 0.0
    if "商超" in preferred:
        score += min(shopping_count, 3) / 3
    if "公园" in preferred:
        score += min(park_count, 3) / 3
    if len(preferred) == 1:
        return min(1.0, score)
    return min(1.0, score / 2)


def _tag_preference_score(house: HouseLite, query: StructuredQuery) -> float:
    tags = [tag.strip() for tag in (house.tags or []) if isinstance(tag, str) and tag.strip()]
    if not tags:
        return 0.0

    def hit(*keywords: str) -> bool:
        lowered = [k.lower() for k in keywords]
        for tag in tags:
            norm_tag = tag.lower()
            if any(key in norm_tag for key in lowered):
                return True
        return False

    score = 0.0
    total = 0.0

    def add_check(matched: bool, weight: float = 1.0) -> None:
        nonlocal score, total
        total += weight
        if matched:
            score += weight

    if query.soft.elevator is True:
        add_check(hit("有电梯", "电梯房", "电梯"))
    elif query.soft.elevator is False:
        add_check(hit("无电梯", "步梯", "没电梯"))

    if query.soft.orientation:
        orient = query.soft.orientation.replace("朝", "").strip()
        add_check(hit(query.soft.orientation, orient))

    if query.soft.value_for_money:
        add_check(hit("高性价比", "性价比", "低价", "优惠", "急租"))

    if query.soft.noise_preference == "安静":
        add_check(hit("安静", "不临街", "远离主路", "低噪"))

    if query.hard.utilities_type:
        add_check(hit(query.hard.utilities_type))

    if query.hard.rent_type == "合租":
        add_check(hit("合租", "单间"))
    elif query.hard.rent_type == "整租":
        add_check(hit("整租"))

    if query.soft.prefer_spacious:
        add_check(hit("大户型", "宽敞", "大面积", "南北通透"))

    if query.soft.prioritize_subway_distance:
        add_check(hit("近地铁", "地铁口", "地铁沿线", "地铁"))

    if query.soft.prioritize_commute:
        add_check(hit("通勤便利", "通勤友好", "近地铁", "近公司"))

    preferred_amenities = set(query.soft.amenities or [])
    if "商超" in preferred_amenities:
        add_check(hit("商超", "商场", "商圈", "超市"))
    if "公园" in preferred_amenities:
        add_check(hit("公园", "绿地", "休闲"))

    preferred_tags = [tag.strip() for tag in (query.soft.preferred_tags or []) if isinstance(tag, str) and tag.strip()]
    if preferred_tags:
        matched = 0
        seen: set[str] = set()
        for preferred in preferred_tags:
            lowered = preferred.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            if hit(preferred):
                matched += 1
        total += float(len(seen))
        score += float(matched)

    avoid_tags = [tag.strip() for tag in (query.soft.avoid_tags or []) if isinstance(tag, str) and tag.strip()]
    if avoid_tags:
        hit_count = 0
        seen_avoid: set[str] = set()
        for avoid in avoid_tags:
            lowered = avoid.lower()
            if lowered in seen_avoid:
                continue
            seen_avoid.add(lowered)
            if hit(avoid):
                hit_count += 1
        # Avoid tags are stronger negative signals.
        weight = 1.2
        total += weight * float(len(seen_avoid))
        score += weight * float(max(0, len(seen_avoid) - hit_count))

    if total <= 0.0:
        return 0.0
    return max(0.0, min(1.0, score / total))
