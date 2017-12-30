#!/usr/bin/env python

# from framework import *
import json
import sys
import time
import threading
import datetime
import time
import numpy as np
import pyschedcl as fw
import plotly.plotly as py
# py.sign_in('anighose25', 'nrJZ4ZwpuHlTRV2zlAD1')
import logging
import argparse
# logging.basicConfig(level=logging.DEBUG)
import os
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
#log_path = os.path.join(os.path.dirname(__file__), '../logs' )

def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='Partition kernels into CPU and GPU heterogeneous system')
    parser.add_argument('-f', '--file',
                        help='Input the json file',
                        required='True')
    parser.add_argument('-p', '--partition_class',
                        help='Inputs the partition ratio')
    parser.add_argument('-d', '--dataset_size',
                        help='Inputs the size of the dataset',
                        default='512')
    parser.add_argument('-ng', '--nGPU',
                        help='Number of GPUs',
                        default='4')
    parser.add_argument('-nc', '--nCPU',
                        help='Number of CPUs',
                        default='2')
    parser.add_argument('-l', '--log',
                        help='Flag for turning on LOG',
                        action="store_true")
    parser.add_argument('-g', '--graph',
                        help='Flag for plotting GANTT chart for execution',
                        action="store_true")
    parser.add_argument('-df', '--dump_output_file',
                        help='Flag for dumping output file for a kernel',
                        action="store_true")
    return parser.parse_args(args)


if __name__ == '__main__':

    args = parse_arg(sys.argv[1:])
    src_name = args.file.split("/")[-1]
    s_name = src_name[:-5]
    cmd_qs, ctxs, gpus, cpus = fw.host_initialize(int(args.nGPU), int(args.nCPU))
    info_file = args.file
    info = json.loads(open(info_file).read())

    dataset = int(args.dataset_size)
    if args.dump_output_file:
        fw.dump_output = True
    if args.partition_class != None :
        partition = int(args.partition_class)
        kernel = fw.Kernel(info, dataset=dataset, partition=partition)
    else:
        kernel = fw.Kernel(info, dataset=dataset)
        partition = info['partition']

    name = s_name + '_' + str(partition) + '_' + str(dataset) + '_' + str(time.time()).replace(".", "")
    if args.log:
        f_path = fw.SOURCE_DIR + 'logs/' + name + '_debug.log'
        logging.basicConfig(filename=f_path, level=logging.DEBUG)
        print "LOG file is saved at %s" % f_path


    logging.debug( "Building Kernel...")
    kernel.build_kernel(gpus, cpus, ctxs)
    logging.debug( "Loading Kernel Data...")
    kernel.random_data()
    logging.debug( "Dispatching Kernel...")
    gpus = range(int(args.nGPU))
    cpus = range(int(args.nCPU))
    start_time, done_events = kernel.dispatch_multiple(gpus, cpus, ctxs, cmd_qs)
    logging.debug(fw.nGPU, fw.nCPU)
    logging.debug( "Waiting for events... \n")
    fw.host_synchronize(cmd_qs, done_events)
    dump_dev = fw.dump_device_history()
    if args.graph:
        filename = fw.SOURCE_DIR + 'gantt_charts/' + name + '.png'
        fw.plot_gantt_chart_graph(dump_dev, filename)




