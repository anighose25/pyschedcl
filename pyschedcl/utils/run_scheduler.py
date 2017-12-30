#!/usr/bin/env python

import json, sys, subprocess, os, datetime
import pyschedcl as fw
import argparse
import time
import sys
from decimal import *
import random
# logging.basicConfig(level=logging.DEBUG)
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='Scheduler Script for Extensive Experimentation')
    parser.add_argument('-f', '--file',
                        help='Input task file containing list of <json filename, partition class, dataset> tuples',
                        default='ALL')
    parser.add_argument('-t', '--tasks',
                        help='Number of tasks',
                        )
    parser.add_argument('-s', '--select',
                        help='Scheduling heuristic (baseline, lookahead, adbias) ',
                        default='baseline')
    parser.add_argument('-ng', '--nGPU',
                        help='Number of GPUs',
                        default='4')
    parser.add_argument('-nc', '--nCPU',
                        help='Number of CPUs',
                        default='2')
    parser.add_argument('-nr', '--runs',
                        help='Number of runs for executing each scheduling algorithm',
                        default=5)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_arg(sys.argv[1:])

    p_path = fw.SOURCE_DIR + 'scheduling/scheduler.py'
    i_path = fw.SOURCE_DIR + 'info/'

    if args.file == 'ALL':
        DEV_NULL = open(os.devnull, 'w')
        dataset =[128, 256, 512, 1024, 2048, 4096]
        partition_class = range(11)
        global_kernels_list = [f for f in os.listdir(i_path) if f.endswith('.json')]
        print len(global_kernels_list)
        print args.tasks
        kernels_list = random.sample(global_kernels_list, int(args.tasks))
        file_name = "Task_" + str(time.time())
        task_file =open(i_path + file_name, "w")
        for kernel in kernels_list:
            d = random.choice(dataset)
            p = random.choice(partition_class)
            task_file.write(kernel + " " + str(p) + " " + str(d) + "\n")
        task_file.close()
        print "Saving task file... " + file_name
        execute_cmd = "python " + p_path + " -f " + i_path + file_name + " -s " + args.select + " > temp.log"
    else:
        file_name = args.file.split("/")[-1]
        execute_cmd = "python " + p_path + " -f " + i_path + file_name + " -s " + args.select + " > temp.log"
    count = 0
    span = 0
    for r in range(int(args.runs)):
        print execute_cmd
        os.system(execute_cmd)
        time.sleep(1)
        get_span_cmd = "cat temp.log |grep span_time"
        print get_span_cmd
        span_info = os.popen(get_span_cmd).read().strip("\n")
        if "span_time" in span_info:
            count = count + 1

            profile_time = float(span_info.split(" ")[1])
            if profile_time > 0:
                span = span + profile_time
    avg_span = span/count
    print "Average Span Time: " + str(avg_span)
    #
    # os.system("rm temp.log")
    # print span_times
