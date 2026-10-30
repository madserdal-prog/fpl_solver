import critic, critic_free

MASTER = {"elements": [{"id": 154, "web_name": "Palmer", "team": "Chelsea", "element_type": "MID"},
                       {"id": 301, "web_name": "Palmer", "team": "Ipswich", "element_type": "MID"},
                       {"id": 5, "web_name": "Isak", "team": "Liverpool", "element_type": "FWD"}]}


def test_penalties_are_proportional_shares():
    rp, blocked, _ = critic.flags_to_solver_inputs(
        {"flags": [{"player_id": 5, "player_name": "Isak", "risk_level": "medium"}]}, MASTER)
    assert rp == {5: 0.3} and blocked == {5}


def test_paid_critic_uses_ids_and_refuses_ambiguous_names():
    by_id = critic.flags_to_solver_inputs({"flags": [{"player_id": 301, "player_name": "Palmer", "risk_level": "high"}]}, MASTER)
    by_name = critic.flags_to_solver_inputs({"flags": [{"player_name": "Palmer", "risk_level": "high"}]}, MASTER)
    assert by_id[0] == {301: 0.7}
    assert by_name[0] == {} and by_name[2] == ["Palmer"]


def test_fpl_status_flags_block_captaincy_but_add_no_penalty():
    fpl_only = critic_free.merge_flags({"player_name": "Isak", "risk_level": "medium", "reason": "75%",
                                        "source": "FPL API status field", "recommendation": "monitor"}, None)
    rp, blocked, _ = critic_free.flags_to_solver_inputs([dict(fpl_only, player_id=5)], MASTER)
    assert rp == {} and blocked == {5}


def test_full_pool_baseline_never_penalises_points():
    m = {"elements": [{"id": 1, "web_name": "A", "status": "d", "chance_of_playing_next_round": 50, "news": "Knock"}]}
    rp, blocked, _ = critic_free.full_pool_solver_inputs(m)
    assert rp == {} and 1 in blocked
