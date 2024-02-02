from log_utils import load_logger
import sys


log_name = sys.argv[1]
lg = load_logger(log_name)

metrics = sys.argv[2:]

lg.print_last('train', metrics)
lg.print_last('eval', metrics)
