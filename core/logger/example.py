from logger import get_logger

# Get loggers for different modules
datasets_logger = get_logger('datasets')
retrieval_logger = get_logger('retrieval')
generation_logger = get_logger('generation')
evaluation_logger = get_logger('evaluation')

def demonstrate_logging():
    """Demonstrate different logging levels for different modules."""
    
    # Datasets module logging (DEBUG level)
    datasets_logger.debug("Processing dataset batch")
    datasets_logger.info("Dataset loading completed")
    datasets_logger.warning("Missing some optional fields")
    
    # Retrieval module logging (INFO level)
    retrieval_logger.debug("This debug message won't show up")  # Won't show up due to INFO level
    retrieval_logger.info("Retrieved documents from index")
    retrieval_logger.warning("Slow retrieval performance detected")
    
    # Generation module logging (DEBUG level)
    generation_logger.debug("Token processing details")
    generation_logger.info("Generated response completed")
    generation_logger.error("Failed to generate response")
    
    # Evaluation module logging (INFO level)
    evaluation_logger.debug("This debug message won't show up")  # Won't show up due to INFO level
    evaluation_logger.info("Evaluation metrics calculated")
    evaluation_logger.critical("Critical error in evaluation pipeline")

if __name__ == "__main__":
    demonstrate_logging() 