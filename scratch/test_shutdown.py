import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api import app
from infra.security import require_api_key

print("API require_api_key id:", id(require_api_key))

import api
print("api.require_api_key id:", id(api.require_api_key))

app.dependency_overrides[require_api_key] = lambda: "mocked"
print("Dependency overrides:", app.dependency_overrides)

client = TestClient(app)
response = client.post(
    "/api/v1/audit",
    json={"objective": "test objective 123", "target_model": "test-model"},
    headers={"X-PromptEvo-Key": "mykey"}
)
print("Response status:", response.status_code)
print("Response text:", response.text)
