# -*- coding: utf-8 -*-
# logger_setup.py
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


class LoggerManager:
    """
    日志管理器
    负责系统的日志记录、文件轮转及控制台输出格式化。
    """

    def __init__(
            self,
            logger_name: str = "Text2ImageEval",
            log_level: int = logging.INFO,
            log_dir: str = "logs"
    ) -> None:
        """
        初始化日志管理器。

        Args:
            logger_name (str): 日志记录器名称
            log_level (int): 日志级别 (如 logging.INFO)
            log_dir (str): 日志文件存储目录
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self.log_dir = log_dir

        # 确保日志目录存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """
        配置日志处理器 (Console Handler 和 File Handler)。
        包含日志格式的定义。
        """
        # 防止重复添加 Handler
        if self.logger.handlers:
            return

        # 定义日志格式
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(module)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 2. 文件处理器 (按大小轮转，最大5MB，保留5个备份)
        log_filename = f"{self.log_dir}/system_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """
        获取配置好的 logger 实例。

        Returns:
            logging.Logger: 日志记录器对象
        """
        return self.logger

    @staticmethod
    def log_system_info() -> None:
        """
        记录当前系统环境信息 (Python版本, OS等)。
        """
        import platform
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version}")


# 创建全局日志实例
logger_manager = LoggerManager()
sys_logger = logger_manager.get_logger()