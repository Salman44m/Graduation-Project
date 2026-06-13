import pytest
import httpx
from unittest.mock import patch, MagicMock
import os

from infra.security import probe_provider_connectivity, _set_health_probe_results, get_health_probe_results

def test_health_probe_results_getter_setter():
    _set_health_probe_results({"openai": "ok"})
    assert get_health_probe_results() == {"openai": "ok"}
    _set_health_probe_results({}) # reset

@patch("httpx.get")
@patch("httpx.post")
@patch.dict(os.environ, {
    "OPENAI_API_KEY": "valid_key",
    "ANTHROPIC_API_KEY": "valid_key"
})
def test_probe_connectivity_success(mock_post, mock_get):
    mock_resp_get = MagicMock()
    mock_resp_get.status_code = 200
    mock_get.return_value = mock_resp_get
    
    mock_resp_post = MagicMock()
    mock_resp_post.status_code = 400 # Anthropic auth success but bad request
    mock_post.return_value = mock_resp_post
    
    results = probe_provider_connectivity()
    
    assert results == {
        "openai": "ok",
        "anthropic": "ok"
    }

@patch("httpx.get")
@patch("httpx.post")
@patch.dict(os.environ, {
    "OPENAI_API_KEY": "valid_key",
    "ANTHROPIC_API_KEY": "valid_key"
})
def test_probe_connectivity_failure(mock_post, mock_get):
    mock_resp_get = MagicMock()
    mock_resp_get.status_code = 401
    mock_get.return_value = mock_resp_get
    
    mock_resp_post = MagicMock()
    mock_resp_post.status_code = 401 
    mock_post.return_value = mock_resp_post
    
    results = probe_provider_connectivity()
    
    assert results == {
        "openai": "error: 401",
        "anthropic": "error: 401"
    }

@patch.dict(os.environ, {
    "OPENAI_API_KEY": "placeholder_key",
    "ANTHROPIC_API_KEY": "sk-..."
})
def test_probe_skips_placeholders():
    # Because they are placeholders, the probe should skip them without making HTTP requests
    results = probe_provider_connectivity()
    assert results == {}

@patch("httpx.post")
@patch("httpx.get")
@patch.dict(os.environ, {
    "OPENAI_API_KEY": "valid_key",
    "ANTHROPIC_API_KEY": "valid_key"
})
def test_probe_handles_exceptions(mock_get, mock_post):
    mock_get.side_effect = Exception("Timeout")
    mock_post.side_effect = Exception("Timeout")
    
    results = probe_provider_connectivity()
    assert results == {"openai": "error: Exception", "anthropic": "error: Exception"}
