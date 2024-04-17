import logging
import torch
from transformers import logging as transformers_logging

# Set up basic configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set transformers logging to INFO to catch all their logs
transformers_logging.set_verbosity_info()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")


    logger.info("CUDA Available: {}".format(torch.cuda.is_available()))
    logger.info("CUDA Version: {}".format(torch.version.cuda))
    logger.info("Device Name: {}".format(torch.cuda.get_device_name(0)))



if __name__ == "__main__":
    main()
