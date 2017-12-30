#!/usr/bin/env python

import json, sys, subprocess, os, datetime
import pyschedcl as fw
import argparse
import time
import sys
from decimal import *
# logging.basicConfig(level=logging.DEBUG)
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='Partition kernels into CPU and GPU heterogeneous system')
    parser.add_argument('-f', '--file',
                        help='Input the json file',
                        required='True')

    parser.add_argument('-d', '--dataset_size',
                        help='Inputs the size of the dataset',
                        default='1024')

    parser.add_argument('-nr', '--runs',
                        help='Number of runs for executing each partitioned variant of the original program',
                        default=5)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_arg(sys.argv[1:])


    DEV_NULL = open(os.devnull, 'w')

    p_path = fw.SOURCE_DIR + 'partition/partition.py'
    i_path = fw.SOURCE_DIR + 'info/'
    span_times = []

    for p in range(11):

        execute_cmd = "python " + p_path + " -f " + i_path + args.file + " -p " + str(p) + " -d " + args.dataset_size + " > temp.log"

        count = 0
        span = 0
        for r in range(int(args.runs)):
            # print execute_cmd
            sys.stdout.write('\r')

            os.system(execute_cmd)
            sys.stdout.flush()

            time.sleep(1)
            get_span_cmd = "cat temp.log |grep span_time"
            # print get_span_cmd
            span_info = os.popen(get_span_cmd).read().strip("\n")
            if "span_time" in span_info:
                count = count + 1

                profile_time = float(span_info.split(" ")[1])
                if profile_time > 0:
                    span = span + profile_time
        avg_span = span/count
        span_times.append(avg_span)
    bar.finish()
    print "Optimal Partition Class: " + str(span_times.index(min(span_times)))
    os.system("rm temp.log")
    # print span_times
