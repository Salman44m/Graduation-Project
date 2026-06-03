import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage
from adapters.base_adapter import BaseTargetAdapter, AdapterAuthError, AdapterResponse
from adapters.langchain_adapter import LangChainTargetAdapter
from adapters.ollama_adapter import OllamaTargetAdapter

def test_base_adapter_enforces_abstract():
    """1. base_adapter enforces abstract interface."""
    with pytest.raises(TypeError):
        BaseTargetAdapter()

def test_langchain_adapter_initializes_valid():
    """2. langchain_adapter initializes with valid OpenAI config."""
    mock_model = MagicMock()
    adapter = LangChainTargetAdapter(model=mock_model)
    assert adapter._model == mock_model

def test_langchain_adapter_raises_invalid_key():
    """3. langchain_adapter raises on invalid API key."""
    mock_model = MagicMock()
    del mock_model.bind
    # Mock invoke to throw Auth error
    class MockAuthException(Exception):
        status_code = 401
    mock_model.invoke.side_effect = MockAuthException("invalid api key")
    adapter = LangChainTargetAdapter(model=mock_model, max_retries=0)
    with pytest.raises(AdapterAuthError):
        adapter.invoke_full([HumanMessage(content="Hi")])

def test_ollama_adapter_initializes():
    """4. ollama_adapter initializes with local endpoint."""
    adapter = OllamaTargetAdapter(base_url="http://localhost:11434")
    assert adapter._base_url == "http://localhost:11434"

def test_adapter_send_message_expected_structure():
    """5. Adapter invoke_full returns expected response structure."""
    mock_model = MagicMock()
    del mock_model.bind
    mock_response = MagicMock()
    mock_response.content = "Test content"
    mock_model.invoke.return_value = mock_response
    
    adapter = LangChainTargetAdapter(model=mock_model)
    res = adapter.invoke_full([HumanMessage(content="test")])
    assert isinstance(res, AdapterResponse)
    assert res.content == "Test content"
