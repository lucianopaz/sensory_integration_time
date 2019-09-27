import pytest
import sensory_integration_time as sit


class TestCApi():
    def test_module(self):
        assert "leaky_integral_calculator" in dir(sit)

    @pytest.mark.parametrize(
        "method", ["leaky_integral_calculator", "lowergamma", "uppergamma"], ids=str
    )
    def test_api_methods(self, method):
        assert method in dir(sit.leaky_integral_calculator)

    def test_c_api_available(self):
        assert sit.model._c_api_available
