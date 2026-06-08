"""Integration tests for the FastAPI endpoints.

Tests are designed to work without actual ML models or Milvus running.
Validation tests check request schema validation. Endpoint tests verify
that routes are registered correctly (lifespan-dependent tests require
a running backend).
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app


@pytest_asyncio.fixture
async def app():
    """Create the FastAPI app for testing."""
    app = create_app(config_path="config.yaml")
    return app


@pytest_asyncio.fixture
async def client(app):
    """Async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestRequestValidation:
    """Test that request validation works correctly (Pydantic schemas).

    These tests work without a running backend because FastAPI validates
    request bodies before the lifespan-dependent route handlers execute.
    """

    @pytest.mark.asyncio
    async def test_chat_empty_question_rejected(self, client):
        """Empty question should be rejected with 422."""
        response = await client.post("/api/v1/chat", json={"question": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_empty_query_rejected(self, client):
        """Empty query should be rejected with 422."""
        response = await client.post("/api/v1/search", json={"query": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_missing_required_field(self, client):
        """Missing 'question' field should be rejected with 422."""
        response = await client.post("/api/v1/chat", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_missing_required_field(self, client):
        """Missing 'query' field should be rejected with 422."""
        response = await client.post("/api/v1/search", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_invalid_top_k(self, client):
        """top_k=0 should be rejected (ge=1)."""
        response = await client.post(
            "/api/v1/chat",
            json={"question": "test", "top_k": 0},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_invalid_top_k(self, client):
        """top_k=100 should be rejected (le=50)."""
        response = await client.post(
            "/api/v1/search",
            json={"query": "test", "top_k": 100},
        )
        assert response.status_code == 422


class TestEndpointRegistration:
    """Test that API endpoints and docs are registered."""

    @pytest.mark.asyncio
    async def test_api_docs_available(self, client):
        """Swagger docs should be accessible."""
        response = await client.get("/docs")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_redoc_available(self, client):
        """ReDoc should be accessible."""
        response = await client.get("/redoc")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_schema(self, client):
        """OpenAPI schema should be accessible."""
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        # Verify key endpoints are documented
        paths = data.get("paths", {})
        assert "/api/v1/chat" in paths
        assert "/api/v1/search" in paths
        assert "/api/v1/health" in paths
