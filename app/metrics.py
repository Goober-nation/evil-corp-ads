import json
import os
import time
from app.config import SystemConfig
from app.logger import get_logger

logger = get_logger(__name__)

class MetricsTracker:
    def __init__(self):
        self.metrics_file = os.path.join(SystemConfig.LOG_DIR, "metrics.json")
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as f:
                json.dump([], f)

    def log_interaction(self, query, ad_objective, aggressiveness, stats):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "ad_objective": ad_objective,
            "aggressiveness": aggressiveness,
            "stats": stats
        }
        try:
            with open(self.metrics_file, "r+") as f:
                data = json.load(f)
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")