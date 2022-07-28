from hypothesis import given, strategies as st

@given(x=st.integers())
def doubling_test(x):
    assert x + x == 2 * x
