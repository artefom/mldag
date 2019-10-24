from genn.configuration import conf

import logging
logger = logging.getLogger(__name__)


def run():
    logger.info("Executing worklfow...")

    logger.info("Mlflow url: {}".format(conf.get('tracking','mlflow_uri')))


if __name__ == "__main__":
    run()
