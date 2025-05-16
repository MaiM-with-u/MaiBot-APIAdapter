import pytest
from packaging.version import InvalidVersion

from src import maibot_llmreq
from src.maibot_llmreq.config.parser import _get_config_version, load_config


class TestConfigLoad:
    def test_loads_valid_version_from_toml(self):
        maibot_llmreq.init_logger()

        toml_data = {"inner": {"version": "1.2.3"}}
        version = _get_config_version(toml_data)
        assert str(version) == "1.2.3"

    def test_handles_missing_version_key(self):
        maibot_llmreq.init_logger()

        toml_data = {}
        version = _get_config_version(toml_data)
        assert str(version) == "0.0.0"

    def test_raises_error_for_invalid_version(self):
        maibot_llmreq.init_logger()

        toml_data = {"inner": {"version": "invalid_version"}}
        with pytest.raises(InvalidVersion):
            _get_config_version(toml_data)

    def test_loads_complete_config_successfully(self, tmp_path):
        maibot_llmreq.init_logger()

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
        [inner]
        version = "0.1.0"
    
        [request_conf]
        max_retry = 5
        timeout = 10

        [[api_providers]]
        name = "provider1"
        base_url = "https://api.example.com"
        api_key = "key123"
        
        [[api_providers]]
        name = "provider2"
        base_url = "https://api.example2.com"
        api_key = "key456"
    
        [[models]]
        model_identifier = "model1"
        api_provider = "provider1"
        
        [[models]]
        model_identifier = "model2"
        api_provider = "provider2"
    
        [task_model_usage]
        task1 = { model = "model1" }
        task2 = "model1"
        task3 = [
            "model1",
            { model = "model2", temperature = 0.5 }
        ]
        """)
        config = load_config(str(config_path))
        assert config.req_conf.max_retry == 5
        assert config.req_conf.timeout == 10
        assert "provider1" in config.api_providers
        assert "model1" in config.models
        assert "task1" in config.task_model_arg_map

    def test_raises_error_for_missing_required_field(self, tmp_path):
        maibot_llmreq.init_logger()

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
        [inner]
        version = "1.0.0"
        """)
        with pytest.raises(KeyError):
            load_config(str(config_path))
