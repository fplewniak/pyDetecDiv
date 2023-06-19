from hypothesis import given, strategies as st


@given(x=st.integers())
def doubling_test(x: int):
    """
    Dummy test always returning True, to check pytest and hypothesis configuration is working
    :param x: an integer
    """
    assert x + x == 2 * x
