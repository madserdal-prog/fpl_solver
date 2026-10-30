import math
import forecast


def _el(**kw):
    e = {"element_type": "MID", "team": "A", "status": "a", "chance_of_playing_next_round": None,
         "start_probability": 0.9, "expected_goals_per_90": 0.3, "expected_assists_per_90": 0.2,
         "ict_index_per_90": 6.0, "minutes": 900, "form": 5.0, "points_per_game": 5.0,
         "games_played_so_far": 10, "penalties_order": None}
    e.update(kw); return e


def test_fixture_difficulty_uses_own_perspective():
    # Bug: home/away difficulty was swapped. Each field is from THAT team's perspective.
    fx = [{"event": 4, "team_h": "Chelsea", "team_a": "Hull", "team_h_difficulty": 2, "team_a_difficulty": 4}]
    lk = forecast.build_fixture_lookup(fx)
    assert lk[("Chelsea", 4)] == [2] and lk[("Hull", 4)] == [4]


def test_doubt_fades_and_left_club_stays_zero():
    lk = {("A", gw): [3] for gw in range(1, 6)}
    doubtful = forecast.compute_xp_series(_el(status="d", chance_of_playing_next_round=50), lk, 1, 5)
    injured = forecast.compute_xp_series(_el(status="i", chance_of_playing_next_round=0), lk, 1, 5)
    left = forecast.compute_xp_series(_el(status="u"), lk, 1, 5)
    assert doubtful[0] < doubtful[2] and injured[0] == 0 and injured[3] > 0 and max(left) == 0


def test_penalty_bonus_immune_to_small_sample():
    lk = {("A", 1): [3]}
    gain = lambda m: (forecast.compute_xp_series(_el(minutes=m, penalties_order=1), lk, 1, 1)[0]
                      - forecast.compute_xp_series(_el(minutes=m), lk, 1, 1)[0])
    assert abs(gain(5) - gain(900)) < 0.02


def test_team_model_strong_home_beats_weak_away():
    els = []
    for t, (xg, xga) in {"Strong": (2.2, 0.8), "Weak": (0.8, 2.2), "Mid1": (1.4, 1.4), "Mid2": (1.4, 1.4)}.items():
        els.append(dict(_el(team=t, element_type="GK", expected_goals_per_90=0), web_name="gk",
                        expected_goals_total=0, expected_goals_conceded_total=xga * 10, saves_per_90=3.0))
        els.append(dict(_el(team=t, element_type="FWD"), web_name="fw", expected_goals_total=xg * 10,
                        expected_goals_conceded_total=0))
    tm = forecast.build_team_model({"elements": els})
    gf, ga = forecast.fixture_expected_goals(tm, "Strong", "Weak", True)
    assert gf > 2 and ga < 1 and math.exp(-ga) > 0.4


def test_falls_back_to_fdr_without_team_data():
    master = {"elements": [dict(_el(), id=1, web_name="x")]}
    fx = [{"event": 1, "team_h": "A", "team_a": "B", "team_h_difficulty": 3, "team_a_difficulty": 3}]
    assert forecast.build_forecast(master, fx, 1, 1)["model"] == "fdr_fallback"
