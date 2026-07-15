# app/metrics.py
import os
import time
import csv
from app.config import SystemConfig
from app.logger import get_logger

logger = get_logger(__name__)

class MetricsTracker:
    def __init__(self):
        self.metrics_file = os.path.join(SystemConfig.LOG_DIR, "metrics.csv")
        self.headers = [
            "timestamp", "query", "ad_objective", "aggressiveness",
            "search_fp32_ms", "search_int8_ms",
            "stage1_pure_rag_ms", "stage2_ad_rewrite_ms",
            "stage3_editor_ms", "total_pipeline_ms"
        ]

        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_interaction(self, query, ad_objective, aggressiveness, stats):
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            query,
            ad_objective,
            aggressiveness,
            stats.get("search_fp32_ms", 0),
            stats.get("search_int8_ms", 0),
            stats.get("stage1_pure_rag_ms", 0),
            stats.get("stage2_ad_rewrite_ms", 0),
            stats.get("stage3_editor_ms", 0),
            stats.get("total_pipeline_ms", 0)
        ]
        try:
            with open(self.metrics_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to write metrics to CSV: {e}")