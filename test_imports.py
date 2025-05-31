#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""


def test_imports():
    try:
        # Test core framework
        import fastapi
        import uvicorn
        import pydantic
        print("✅ FastAPI stack imported successfully")

        # Test our modules
        from src.config.settings import get_settings
        from src.utils.logger import get_logger
        from src.exceptions.base_exceptions import ChatServiceException
        print("✅ Our modules imported successfully")

        # Test databases
        import motor
        import redis
        import asyncpg
        print("✅ Database drivers imported successfully")

        # Test development tools
        import pytest
        import black
        import mypy
        print("✅ Development tools imported successfully")

        print("\n🎉 All dependencies installed correctly!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    return True


if __name__ == "__main__":
    test_imports()