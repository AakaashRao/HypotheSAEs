from hypothesaes.batch import choose_backend


def test_choose_backend_live_and_batch():
    assert choose_backend("live", 10, 5) == "live"
    assert choose_backend("batch", 1, 100) == "batch"


def test_choose_backend_auto_threshold():
    assert choose_backend("auto", 50, 100) == "live"
    assert choose_backend("auto", 150, 100) == "batch"


def test_choose_backend_invalid():
    try:
        choose_backend("invalid", 1, 1)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for invalid backend"
