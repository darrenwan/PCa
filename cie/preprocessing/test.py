# -*- coding: utf-8 -*-

from cie.common import logger

logger = logger.get_logger(name=logger.get_name(__file__))


def f():
    logger.debug(f"abc")


if __name__ == "__main__":
    print("program begins")
    f()
    print("program ends")
