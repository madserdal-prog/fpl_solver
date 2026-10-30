import fetch_my_team as fmt
import my_team


def test_free_transfers_skip_gw1_and_handle_wildcard():
    h = fmt._self_test_history_data()
    assert [fmt.compute_free_transfers(h, gw) for gw in (2, 3, 4, 5, 6)] == [1, 2, 1, 2, 3]


def test_chips_reset_per_half_and_ignore_future():
    h = {"chips": [{"name": "wildcard", "event": 4}, {"name": "3xc", "event": 12}, {"name": "bboost", "event": 21}]}
    assert fmt.compute_chips_available(h, 9) == ["free_hit", "bench_boost", "triple_captain"]
    assert fmt.compute_chips_available(h, 19) == ["wildcard", "free_hit", "bench_boost", "triple_captain"]
    assert fmt.compute_chips_available(h, 22) == ["wildcard", "free_hit", "triple_captain"]


def test_sync_carries_purchase_prices():
    existing = {"chips_available": [], "locked_players": [], "purchase_prices": {"101": 8.5, "999": 5.0}}
    mt = fmt.build_my_team_from_entry(fmt._self_test_entry_data(), existing=existing, computed_free_transfers=1)
    assert mt["purchase_prices"] == {"101": 8.5}


def test_sell_value_rule():
    assert my_team.compute_sell_value(5.0, 6.0) == 5.5
    assert my_team.compute_sell_value(5.0, 5.3) == 5.1      # half of profit, rounded DOWN
    assert my_team.compute_sell_value(8.0, 7.5) == 7.5      # full loss passed through
