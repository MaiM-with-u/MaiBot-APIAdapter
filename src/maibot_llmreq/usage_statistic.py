from datetime import datetime
from enum import Enum
from typing import Tuple

from pymongo.synchronous.database import Database

from . import _logger as logger
from .config.config import ModelInfo


class ReqType(Enum):
    """
    请求类型
    """

    CHAT = "chat"  # 对话请求
    EMBEDDING = "embedding"  # 嵌入请求


class UsageCallStatus(Enum):
    """
    任务调用状态
    """

    PROCESSING = "processing"  # 处理中
    SUCCESS = "success"  # 成功
    FAILURE = "failure"  # 失败
    CANCELED = "canceled"  # 取消


class ModelUsageStatistic:
    db: Database | None = None

    def __init__(self, db: Database):
        if db is None:
            logger.warning(
                "Warning: No database provided, ModelUsageStatistic will not work."
            )
            return
        if self._init_database(db):
            # 成功初始化
            self.db = db

    @staticmethod
    def _init_database(db: Database):
        """
        初始化数据库相关索引
        """
        try:
            db.llm_usage.create_index([("timestamp", 1)])
            db.llm_usage.create_index([("model_name", 1)])
            db.llm_usage.create_index([("task_name", 1)])
            db.llm_usage.create_index([("request_type", 1)])
            db.llm_usage.create_index([("status", 1)])
            return True
        except Exception as e:
            logger.error(f"创建数据库索引失败: {e}")
            return False

    @staticmethod
    def _calculate_cost(
        prompt_tokens: int, completion_tokens: int, model_info: ModelInfo
    ) -> float:
        """计算API调用成本
        使用模型的pri_in和pri_out价格计算输入和输出的成本

        Args:
            prompt_tokens: 输入token数量
            completion_tokens: 输出token数量

        Returns:
            float: 总成本（元）
        """
        # 使用模型的pri_in和pri_out计算成本
        input_cost = (prompt_tokens / 1000000) * model_info.price_in
        output_cost = (completion_tokens / 1000000) * model_info.price_out
        return round(input_cost + output_cost, 6)

    def create_usage(
        self,
        model_name: str,
        task_name: str = "N/A",
        request_type: ReqType = ReqType.CHAT,
    ) -> str | None:
        """
        创建模型使用情况记录
        :param model_name: 模型名
        :param task_name: 任务名称
        :param request_type: 请求类型，默认为Chat
        :return:
        """
        if self.db is None:
            return None  # 如果没有数据库连接，则不记录使用情况

        try:
            usage_data = {
                "model_name": model_name,
                "task_name": task_name,
                "request_type": request_type.value,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "status": "processing",
                "timestamp": datetime.now(),
                "ext_msg": None,
            }
            result = self.db.llm_usage.insert_one(usage_data)

            logger.trace(
                f"创建了一条模型使用情况记录 - 模型: {model_name}, "
                f"子任务: {task_name}, 类型: {request_type}"
                f"记录ID: {str(result.inserted_id)}"
            )

            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"创建模型使用情况记录失败: {str(e)}")
            return None

    def update_usage(
        self,
        record_id: str | None,
        model_info: ModelInfo,
        usage_data: Tuple[int, int, int] | None = None,
        stat: UsageCallStatus = UsageCallStatus.SUCCESS,
        ext_msg: str | None = None,
    ):
        """
        更新模型使用情况

        Args:
            record_id: 记录ID
            model_info: 模型信息
            usage_data: 使用情况数据(输入token数量, 输出token数量, 总token数量)
            stat: 任务调用状态
            ext_msg: 额外信息
        """
        if self.db is None:
            return  # 如果没有数据库连接，则不记录使用情况

        if not record_id:
            logger.error("更新模型使用情况失败: record_id不能为空")
            return

        if usage_data and len(usage_data) != 3:
            logger.error("更新模型使用情况失败: usage_data的长度不正确，应该为3个元素")
            return

        # 提取使用情况数据
        prompt_tokens = usage_data[0] if usage_data else 0
        completion_tokens = usage_data[1] if usage_data else 0
        total_tokens = usage_data[2] if usage_data else 0

        try:
            self.db.llm_usage.update_one(
                {"_id": record_id},
                {
                    "$set": {
                        "status": stat.value,
                        "ext_msg": ext_msg,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "cost": self._calculate_cost(
                            prompt_tokens, completion_tokens, model_info
                        )
                        if usage_data
                        else 0.0,
                    }
                },
            )

            logger.trace(
                f"Token使用情况 - 模型: {model_info.name}, "
                f"记录ID： {record_id}, "
                f"任务状态: {stat.value}, 额外信息： {ext_msg if ext_msg else 'N/A'}, "
                f"提示词: {prompt_tokens}, 完成: {completion_tokens}, "
                f"总计: {total_tokens}"
            )
        except Exception as e:
            logger.error(f"记录token使用情况失败: {str(e)}")
