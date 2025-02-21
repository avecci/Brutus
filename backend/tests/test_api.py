import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

if __name__ == "__main__":
    """To be filled"""
    client()
