import os
import logging

def create_run_log_dir(base_dir="logs"):
    os.makedirs(base_dir, exist_ok=True)
    run_id = 1
    while True:
        run_dir = os.path.join(base_dir, f"{run_id:04d}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_dir
        run_id += 1

def get_run_dir(base_dir="logs"):
    os.makedirs(base_dir, exist_ok=True)
    run_id = 1
    while True:
        run_dir = os.path.join(base_dir, f"{run_id:04d}")
        if not os.path.exists(run_dir):
            return run_id - 1
        run_id += 1

class AgentNameFilter(logging.Filter):
    """This filter adds a new 'agent_name' attribute to the log record."""
    def filter(self, record):
        record.agent_name = record.name.split('.')[-1]
        return True

def get_logger(run_dir=None, agent_id=None):
    def create_file_handler(level: int):
        agent_filter = AgentNameFilter()
        handler = logging.FileHandler(os.path.join(run_dir, logging.getLevelName(level)+".log"))
        handler.setLevel(level)
        handler.setFormatter(formatter)
        handler.addFilter(agent_filter)
        return handler

    logger = logging.getLogger("main")
    if not logger.handlers:
        if run_dir is None:
            run_dir = create_run_log_dir()
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("[%(levelname)-5s] [%(agent_name)-6s] %(message)s", "%H:%M:%S")


        logger.addHandler(create_file_handler(logging.DEBUG))
        logger.addHandler(create_file_handler(logging.WARNING))
        logger.addHandler(create_file_handler(logging.INFO))

    if agent_id is None:
        return logger
    
    return logging.getLogger(f"main.agent{agent_id}")
