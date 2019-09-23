import pytest
import leakyIntegrator


def test_c_api():
    assert leakyIntegrator.model._c_api_available


def test_model():
    leakyIntegrator.model.tests()