import os

class SystemConfig:
    LLM_NAME = os.getenv("LLM_NAME", "Qwen/Qwen2.5-3B-Instruct")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DATASET_SLICE = int(os.getenv("DATASET_SLICE", 50000))
    LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 200))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.3))
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    INDEX_FP32_PATH = os.path.join(DATA_DIR, "faiss_index_fp32.bin")
    INDEX_INT8_PATH = os.path.join(DATA_DIR, "faiss_index_int8.bin")