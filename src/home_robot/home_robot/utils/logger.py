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

def get_logger(run_dir=None):
    logger = logging.getLogger("multi_level_logger")
    if not logger.handlers:
        if run_dir is None:
            run_dir = create_run_log_dir()
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")

        # DEBUG log (everything)
        debug_handler = logging.FileHandler(os.path.join(run_dir, "debug.log"))
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(formatter)
        logger.addHandler(debug_handler)

        # INFO log (info, warning, error, critical)
        info_handler = logging.FileHandler(os.path.join(run_dir, "info.log"))
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)

        # WARNING log (warning, error, critical)
        warn_handler = logging.FileHandler(os.path.join(run_dir, "warn.log"))
        warn_handler.setLevel(logging.WARNING)
        warn_handler.setFormatter(formatter)
        logger.addHandler(warn_handler)

        logger.propagate = False

    return logger
