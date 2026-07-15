import logging
import json
import os
from datetime import datetime
from app.config import SystemConfig

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)
        return json.dumps(log_record)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        c_handler.setFormatter(logging.Formatter('[%(levelname)s] %(name)s - %(message)s'))
        
        f_handler = logging.FileHandler(os.path.join(SystemConfig.LOG_DIR, "app.log"))
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(JsonFormatter())
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger