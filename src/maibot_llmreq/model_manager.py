import importlib
from typing import Dict


from .config.config import (
    ModelUsageArgConfig,
    ModuleConfig,
)

from . import _logger as logger
from .model_client import ModelRequestHandler, BaseClient


class ModelManager:
    # TODO: 添加读写锁，防止异步刷新配置时发生数据竞争

    def __init__(
        self,
        config: ModuleConfig,
    ):
        self.config: ModuleConfig = config
        """配置信息"""

        self.api_client_map: Dict[str, BaseClient] = {}
        """API客户端映射表"""

        for provider_name, api_provider in self.config.api_providers.items():
            # 初始化API客户端
            try:
                # 根据配置动态加载实现
                client_module = importlib.import_module(
                    f".model_client.{api_provider.client_type}_client", __package__
                )
                client_class = getattr(
                    client_module, f"{api_provider.client_type.capitalize()}Client"
                )
                if not issubclass(client_class, BaseClient):
                    raise TypeError(
                        f"'{client_class.__name__}' is not a subclass of 'BaseClient'"
                    )
                self.api_client_map[api_provider.name] = client_class(
                    api_provider
                )  # 实例化，放入api_client_map
            except ImportError as e:
                logger.error(f"Failed to import client module: {e}")
                raise ImportError(
                    f"Failed to import client module for '{provider_name}': {e}"
                ) from e

    def __getitem__(self, task_name: str) -> ModelRequestHandler:
        """
        获取任务所需的模型客户端（封装）
        :param task_name: 任务名称
        :return: 模型客户端
        """
        if task_name not in self.config.task_model_arg_map:
            raise KeyError(f"'{task_name}' not registered in ModelManager")

        return ModelRequestHandler(
            task_name=task_name,
            config=self.config,
            api_client_map=self.api_client_map,
        )

    def __setitem__(self, task_name: str, value: ModelUsageArgConfig):
        """
        注册任务的模型使用配置
        :param task_name: 任务名称
        :param value: 模型使用配置
        """
        self.config.task_model_arg_map[task_name] = value

    def __contains__(self, task_name: str):
        """
        判断任务是否已注册
        :param task_name: 任务名称
        :return: 是否在模型列表中
        """
        return task_name in self.config.task_model_arg_map
