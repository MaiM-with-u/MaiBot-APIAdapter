import pytest

from src import maibot_llmreq
from src.maibot_llmreq.config.config import ModuleConfig, ModelUsageConfig, APIProvider
from src.maibot_llmreq.model_client import ModelRequestHandler
from src.maibot_llmreq.model_manager import ModelManager


class TestModelManager:
    def test_retrieves_model_request_handler_for_registered_task(self):
        maibot_llmreq.init_logger()

        config = ModuleConfig(
            task_model_usage_map={"task1": ModelUsageConfig(name="task1", usage=[])}
        )
        manager = ModelManager(config)
        handler = manager["task1"]
        assert isinstance(handler, ModelRequestHandler)
        assert handler.task_name == "task1"

    def test_raises_key_error_for_unregistered_task(self):
        maibot_llmreq.init_logger()

        config = ModuleConfig(task_model_usage_map={})
        manager = ModelManager(config)
        with pytest.raises(KeyError):
            _ = manager["unregistered_task"]

    def test_registers_task_model_usage_config_successfully(self):
        maibot_llmreq.init_logger()

        config = ModuleConfig(task_model_usage_map={})
        manager = ModelManager(config)
        usage_config = ModelUsageConfig(name="new_task", usage=[])
        manager["new_task"] = usage_config
        assert "new_task" in manager
        assert manager.config.task_model_usage_map["new_task"] == usage_config

    def test_checks_if_task_is_registered_correctly(self):
        maibot_llmreq.init_logger()

        config = ModuleConfig(
            task_model_usage_map={"task1": ModelUsageConfig(name="task1", usage=[])}
        )
        manager = ModelManager(config)
        assert "task1" in manager
        assert "unregistered_task" not in manager

    def test_initializes_api_client_map_correctly(self):
        maibot_llmreq.init_logger()

        provider = APIProvider(
            name="provider1",
            base_url="https://example.com",
            api_key="ExampleKey",
        )
        config = ModuleConfig(api_providers={"provider1": provider})
        manager = ModelManager(config)
        assert "provider1" in manager.api_client_map
        assert manager.api_client_map["provider1"].api_provider == provider

    def test_raises_import_error_for_invalid_client_module(self):
        maibot_llmreq.init_logger()

        provider = APIProvider(
            name="provider1",
            base_url="https://example.com",
            api_key="ExampleKey",
            client_type="invalid_client",
        )
        config = ModuleConfig(api_providers={"provider1": provider})
        with pytest.raises(ImportError):
            ModelManager(config)
