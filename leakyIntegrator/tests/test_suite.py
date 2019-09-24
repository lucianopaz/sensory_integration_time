import pytest
import leakyIntegrator


class TestCApi():
    def test_module(self):
        assert "leaky_integral_calculator" in dir(leakyIntegrator)

    @pytest.mark.parametrize(
        "method", ["leaky_integral_calculator", "lowergamma", "uppergamma"], ids=str
    )
    def test_api_methods(self, method):
        assert method in dir(leakyIntegrator.leaky_integral_calculator)

    def test_c_api_available(self):
        assert leakyIntegrator.model._c_api_available
