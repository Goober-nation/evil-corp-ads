from app.engine import RAGPipeline
from app.ui import create_ui
from app.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting Application Server...")
    pipeline = RAGPipeline()
    demo = create_ui(pipeline)
    demo.launch(server_name="0.0.0.0", server_port=7860)