# PromptEvo API Error Documentation

This document outlines the standard error responses returned by the PromptEvo REST API.

## Error Schema

All standard API errors return a structured JSON response matching the `ErrorResponse` schema:

```json
{
  "detail": "Human-readable description of the error",
  "error_code": "OPTIONAL_SYSTEM_ERROR_CODE"
}
```

## Standard HTTP Status Codes

### 400 Bad Request
Returned when the incoming request is syntactically invalid or missing required top-level parameters that cannot be caught by basic schema validation.

### 401 Unauthorized
Returned when authentication is required but missing or malformed.
- **Header Required**: `X-PromptEvo-Key`
- **Resolution**: Provide a valid API key configured in `PROMPTEVO_API_KEYS`.

### 403 Forbidden
Returned when the client is authenticated but lacks permission to perform the requested action.
- **Invalid Key**: The provided API key is invalid.
- **Target Model Not Allowed**: The requested `target_model` is not in the operator's configured allowlist (`PROMPTEVO_ALLOWED_TARGETS`).

### 404 Not Found
Returned when a requested resource does not exist.
- **Session Not Found**: The requested `session_id` does not exist in the active or persistent session store.

### 422 Unprocessable Entity
Returned when the request payload is well-formed JSON but fails semantic validation.
- **Validation Error**: Missing required fields (e.g., `objective`), or fields do not meet length/format constraints (e.g., `block_threshold` out of bounds).
- **CI/CD Threshold Exceeded**: If `block_threshold` is set and the final RAHS score exceeds it, the API will return HTTP 422 to trigger a failure in the CI/CD pipeline.

### 500 Internal Server Error
Returned when an unexpected error occurs during graph execution, LLM invocation, or database operations. Check the server logs for detailed tracebacks.

### 503 Service Unavailable
Returned when the system cannot handle the request.
- **Shutting Down**: The server is draining active sessions and refusing new ones (Graceful Shutdown).
- **Misconfigured**: `PROMPTEVO_API_KEYS` is empty and `PROMPTEVO_DEV_DISABLE_AUTH` is not true.
- **Graph Compilation Failed**: LangGraph failed to compile on startup.

## Common Error Scenarios

| Scenario | Status Code | Example Detail |
|---|---|---|
| Missing API Key | 401 | "Authentication required. Provide your API key in the X-PromptEvo-Key header." |
| Invalid Target Model | 403 | "Target model 'gpt-4' is not allowed." |
| Polling Non-Existent Session | 404 | "Session '1234' not found." |
| Invalid Objective Length | 422 | "String should have at least 10 characters." |
| CI/CD Pipeline Blocking | 422 | "RAHS score (8.5) exceeded threshold (7.0)." |
| Server Misconfiguration | 503 | "Server Security Misconfiguration: PROMPTEVO_API_KEYS not set..." |
