from src.risk_engine import assess_order


def test_risk_engine_returns_expected_fields():
    result = assess_order(
        order_value_inr=500,
        orders_last_15m=1,
        failed_confirmations=0,
        distinct_accounts=1,
        confirmation_status="confirmed",
        payment_method="COD",
    )

    assert 0 <= result["risk_probability"] <= 1
    assert result["risk_category"] in {
        "Low",
        "Medium",
        "High",
    }
    assert "recommendation" in result