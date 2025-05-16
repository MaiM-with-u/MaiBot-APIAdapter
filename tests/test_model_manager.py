import pytest

from src import maibot_llmreq
from src.maibot_llmreq.config.config import (
    ModuleConfig,
    ModelUsageArgConfig,
    APIProvider,
)
from src.maibot_llmreq.model_client import ModelRequestHandler
from src.maibot_llmreq.model_manager import ModelManager


class TestModelManager:
    def test_retrieves_model_request_handler_for_registered_task(self):
        manager = (
            self.test_extracted_from_test_checks_if_task_is_registered_correctly_2()
        )
        handler = manager["task1"]
        assert isinstance(handler, ModelRequestHandler)
        assert handler.task_name == "task1"

    def test_raises_key_error_for_unregistered_task(self):
        manager = self.test_extracted_from_test_registers_task_model_usage_config_successfully_2()
        with pytest.raises(KeyError):
            _ = manager["unregistered_task"]

    def test_registers_task_model_usage_config_successfully(self):
        manager = self.test_extracted_from_test_registers_task_model_usage_config_successfully_2()
        usage_config = ModelUsageArgConfig(name="new_task", usage=[])
        manager["new_task"] = usage_config
        assert "new_task" in manager
        assert manager.config.task_model_arg_map["new_task"] == usage_config

    def test_extracted_from_test_registers_task_model_usage_config_successfully_2(self):
        maibot_llmreq.init_logger()
        config = ModuleConfig(task_model_arg_map={})
        return ModelManager(config)

    def test_checks_if_task_is_registered_correctly(self):
        manager = (
            self.test_extracted_from_test_checks_if_task_is_registered_correctly_2()
        )
        assert "task1" in manager
        assert "unregistered_task" not in manager

    def test_extracted_from_test_checks_if_task_is_registered_correctly_2(self):
        maibot_llmreq.init_logger()
        config = ModuleConfig(
            task_model_arg_map={"task1": ModelUsageArgConfig(name="task1", usage=[])}
        )
        return ModelManager(config)

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
