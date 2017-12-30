#!/usr/bin/env python

# from fw import *
import json, sys, datetime, time, heapq
import numpy as np
import pyschedcl as fw
import logging
import argparse

import resource, sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)


def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='Schedule set of independent OpenCL kernels on CPU-GPU heterogeneous multicores')
    parser.add_argument('-f', '--file',
                        help='Input task file containing list of <json filename, partition class, dataset> tuples',
                        required='True')
    parser.add_argument('-s', '--select',
                        help='Scheduling heuristic (baseline, lookahead, adbias) ',
                        default='baseline')
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

# logging.basicConfig(level=logging.DEBUG)


def baseline_select(kernels, **kwargs):
    mixed_q = kwargs['M']
    gpu_q = kwargs['G']
    cpu_q = kwargs['C']

    now_kernel = None
    if fw.nCPU > 0 and fw.nGPU > 0 and mixed_q:
        i_p, i_num_global_work_items, i = heapq.heappop(mixed_q)
        i_num_global_work_items, i_p = kernels[i].get_num_global_work_items(), kernels[i].partition

        now_kernel = (i, i_p)
    elif fw.nCPU > 0 and cpu_q:
        c_p, c_num_global_work_items, c_i = heapq.heappop(cpu_q)
        now_kernel = (c_i, 0)
    elif fw.nGPU > 0 and gpu_q:
        g_p, g_num_global_work_items, g_i = heapq.heappop(gpu_q)
        now_kernel = (g_i, 10)
    if now_kernel is not None:
        i, p =now_kernel
        logging.debug( "Selecting kernel " + kernels[i].name + " with partition class value " + str(p))
    return now_kernel, None
    

def look_ahead_select(kernels, **kwargs):
    mixed_q = kwargs['M']
    gpu_q = kwargs['G']
    cpu_q = kwargs['C']
    now_kernel, next_kernel = None, None
    if fw.nCPU > 0 and fw.nGPU > 0 and len(mixed_q) >= 2:
        i_p, i_num_global_work_items, i = heapq.heappop(mixed_q)
        j_p, j_num_global_work_items, j = heapq.heappop(mixed_q)
        i_num_global_work_items, j_num_global_work_items = kernels[i].get_num_global_work_items(), kernels[j].get_num_global_work_items()
        i_p, j_p = kernels[i].partition, kernels[j].partition
        if i_p >= 8 and j_p <= 2:
            next_kernel = (j, 0)
            now_kernel = (i, 10)
        elif j_p >= 8 and i_p <= 2:
            next_kernel = (i, 0)
            now_kernel = (j, 10)
        elif i_num_global_work_items < j_num_global_work_items:
            now_kernel = (i, i_p)
            heapq.heappush(mixed_q, (abs(j_p-5), -j_num_global_work_items, j))
        else:
            now_kernel = (j, j_p)
            heapq.heappush(mixed_q, (abs(i_p-5), -i_num_global_work_items, i))
    elif fw.nCPU > 0 and fw.nGPU > 0 and mixed_q:
        i_p, i_num_global_work_items, i = heapq.heappop(mixed_q)
        i_num_global_work_items = kernels[i].get_num_global_work_items()
        i_p = kernels[i].partition
        now_kernel = (i, i_p)
    elif fw.nCPU > 0 and cpu_q:
        c_p, c_num_global_work_items, c_i = heapq.heappop(cpu_q)
        now_kernel = (c_i, 0)
    elif fw.nGPU > 0 and gpu_q:
        g_p, g_num_global_work_items, g_i = heapq.heappop(gpu_q)
        now_kernel = (g_i, 10)
    if now_kernel is not None:
        i, p = now_kernel
        logging.debug( "Selecting kernel " + kernels[i].name + " with partition class value " + str(p))
    return now_kernel, next_kernel    
            

def adaptive_bias_select(kernels, **kwargs):
    mixedc_q = kwargs['M1']
    mixedg_q = kwargs['M2']
    gpu_q = kwargs['G']
    cpu_q = kwargs['C']
    num_global_work_items_max =kwargs['feat_max']
    t = kwargs['threshold']
    now_kernel, next_kernel = None, None
    if fw.nCPU > 0 and fw.nGPU > 0 and mixedc_q and mixedg_q:
        is_dispatched = 0
        c_p, c_num_global_work_items, c_i = heapq.heappop(mixedc_q)
        g_p, g_num_global_work_items, g_i = heapq.heappop(mixedg_q)
        c_num_global_work_items, g_num_global_work_items = kernels[c_i].get_num_global_work_items(), kernels[g_i].get_num_global_work_items()
        c_p, g_p = kernels[c_i].partition, kernels[g_i].partition

        if c_p <= 2 and g_p >= 8:
            next_kernel = (c_i, 0)
            now_kernel = (g_i, 10)
            is_dispatched = 1

        elif c_p <= 4 and g_p >= 6:
            dispatch_cpu,  dispatch_gpu = False, False
            if c_num_global_work_items < t*num_global_work_items_max:
                now_kernel = (c_i, 0)
                is_dispatched +=1
                dispatch_cpu = True
            if g_num_global_work_items < t*num_global_work_items_max:
                if now_kernel == None:
                    now_kernel = (g_i, 10)
                else:
                    next_kernel = (g_i, 10)
                is_dispatched += 1
                dispatch_gpu = True

            if is_dispatched < 2:
                if dispatch_cpu:
                    heapq.heappush(mixedg_q, (abs(g_p - 5), -g_num_global_work_items, g_i))
                else:
                    heapq.heappush(mixedc_q, (abs(c_p - 5), -c_num_global_work_items, c_i))


        if is_dispatched == 0:
            if c_num_global_work_items < g_num_global_work_items:
                now_kernel = (c_i, c_p)
                heapq.heappush(mixedg_q, (abs(g_p-5), -g_num_global_work_items, g_i))
            else:
                now_kernel = (g_i, g_p)
                heapq.heappush(mixedc_q, (abs(c_p-5), -c_num_global_work_items, c_i))
    elif fw.nCPU > 0 and fw.nGPU > 0 and mixedc_q:
        i_p, i_num_global_work_items, i = heapq.heappop(mixedc_q)
        i_num_global_work_items = kernels[i].get_num_global_work_items()
        i_p = kernels[i].partition
        now_kernel = (i, i_p)
    elif fw.nCPU > 0 and fw.nGPU > 0 and mixedg_q:
        i_p, i_num_global_work_items, i = heapq.heappop(mixedg_q)
        i_num_global_work_items = kernels[i].get_num_global_work_items()
        i_p = kernels[i].partition
        now_kernel = (i, i_p)
    elif fw.nCPU > 0 and cpu_q:
        c_p, c_num_global_work_items, c_i = heapq.heappop(cpu_q)
        now_kernel = (c_i, 0)
    elif fw.nGPU > 0 and gpu_q:
        g_p, g_num_global_work_items, g_i = heapq.heappop(gpu_q)
        now_kernel = (g_i, 10)
    if now_kernel is not None:
        i, p = now_kernel
        logging.debug( "Selecting kernel " + kernels[i].name + " with partition class value " + str(p))
    return now_kernel, next_kernel


def select_main(kernels, select=baseline_select):
    cmd_qs, ctxs, gpus, cpus = fw.host_initialize(4, 2)
    aCPU, aGPU = fw.nCPU, fw.nGPU
    cpu_q, gpu_q, mixed_q, mixedg_q, mixedc_q = [], [], [], [], []
    num_dispatched = 0
    num_kernels = len(kernels)
    rCPU, rGPU = 0, 0
    num_global_work_items_max = 0
    for i in range(len(kernels)):
        kernels[i].build_kernel(gpus, cpus, ctxs)
        kernels[i].random_data()
        p = kernels[i].partition
        num_global_work_items = kernels[i].get_num_global_work_items()
        if num_global_work_items > num_global_work_items_max:
            num_global_work_items_max = num_global_work_items

        if p == 0:
            heapq.heappush(cpu_q, (p, -num_global_work_items, i))
            rCPU += 1
        elif p == 10:
            heapq.heappush(gpu_q, (-p, -num_global_work_items, i))
            rGPU += 1
        elif p >= 5:
            heapq.heappush(mixedg_q, (abs(p-5), -num_global_work_items, i))
            heapq.heappush(mixed_q, (abs(p-5), -num_global_work_items, i))
            rCPU += 1
            rGPU += 1
        else:
            heapq.heappush(mixedc_q, (abs(p-5), -num_global_work_items, i))
            heapq.heappush(mixed_q, (abs(p-5), -num_global_work_items, i))
            rCPU +=1
            rGPU +=1

    logging.debug( "Task Queue Stats")
    logging.debug(rCPU)
    logging.debug(rGPU)

    logging.debug( "CPU " + str(len(cpu_q)))
    logging.debug( "GPU " + str(len(gpu_q)))
    logging.debug( "Mixed " + str(len(mixed_q)))
    logging.debug( "Mixed CPU " +str(len(mixedc_q)))
    logging.debug( "Mixed GPU " +str(len(mixedg_q)))


    events = []
    now, soon = None, None
    while (len(cpu_q) > 0 or len(gpu_q) > 0 or len(mixed_q) > 0 or num_dispatched != num_kernels) and (len(cpu_q) > 0 or len(gpu_q) > 0 or
                                                                      len(mixedc_q) > 0 or len(mixedg_q) > 0 or num_dispatched != num_kernels):

        logging.debug( "READY DEVICES")
        logging.debug( "GPU " + str(len(fw.ready_queue['gpu'])))
        logging.debug( "CPU " + str(len(fw.ready_queue['cpu'])))
        logging.debug( "Number of tasks left")
        logging.debug( "Mixed Queue " + str(len(mixed_q)))
        logging.debug( "CPU Queue " + str(len(cpu_q)))
        logging.debug( "GPU Queue " + str(len(gpu_q)))
        logging.debug( "Mixed CPU " + str(len(mixedc_q)))
        logging.debug( "Mixed GPU " + str(len(mixedg_q)))

        logging.debug( "Number of available devices (CPU and GPU) " + str(fw.nCPU) + " " + str(fw.nGPU))
        if fw.nCPU > 0 or fw.nGPU > 0:
            logging.debug( "Entering selection phase")
            if soon == None:
                if select is baseline_select or select is look_ahead_select:

                    now, soon = select(kernels, M=mixed_q, G=gpu_q, C=cpu_q)
                else:

                    now, soon = select(kernels, M1=mixedc_q, M2=mixedg_q, G=gpu_q, C=cpu_q, feat_max=num_global_work_items_max, threshold=0.4)
            else:
                now, soon = soon, None

            if now == None:
                logging.debug( "Searching for available devices")
                continue
            i, p = now
            if fw.nCPU > rCPU and fw.nGPU > rGPU:
                logging.debug( "DISPATCH MULTIPLE")

                g_factor = fw.nGPU if rGPU == 0 else fw.nGPU / rGPU
                c_factor = fw.nCPU if rCPU == 0 else fw.nCPU / rCPU
                free_gpus, free_cpus = [], []
                # try:
                while fw.test_and_set(0,1):
                    pass
                if p == 0:
                    for j in range(c_factor):
                        free_cpus.append(fw.ready_queue['cpu'].popleft())
                elif p == 10:
                    for j in range(g_factor):
                        free_gpus.append(fw.ready_queue['gpu'].popleft())
                else:
                    for j in range(c_factor):
                        free_cpus.append(fw.ready_queue['cpu'].popleft())
                    for j in range(g_factor):
                        free_gpus.append(fw.ready_queue['gpu'].popleft())
                fw.rqlock[0] = 0
                # except:
                #     logging.debug( free_cpus, free_gpus, framework.ready_queue, framework.nCPU, framework.nGPU, c_factor, g_factor

                if kernels[i].partition == 0:
                    rCPU -= 1
                elif kernels[i].partition == 10:
                    rGPU -= 1
                else:
                    rCPU -= 1
                    rGPU -= 1

                kernels[i].partition = p
                # kernels[i].build_kernel(gpus, cpus, ctxs)
                # kernels[i].random_data()
                logging.debug( "Dispatching Multiple " + str(kernels[i].name))
                start_time, done_events = kernels[i].dispatch_multiple(free_gpus, free_cpus, ctxs, cmd_qs)
                events.extend(done_events)
                num_dispatched += 1
            # if False:
            #     pass
            else:
                logging.debug( "DISPATCH")
                cpu, gpu = -1, -1
                if p == 0:
                    while fw.test_and_set(0, 1):
                        pass
                    cpu = fw.ready_queue['cpu'].popleft()
                    fw.rqlock[0] = 0
                elif p == 10:
                    while fw.test_and_set(0, 1):
                        pass
                    gpu = fw.ready_queue['gpu'].popleft()
                    fw.rqlock[0] = 0
                else:
                    while fw.test_and_set(0, 1):
                        pass
                    cpu = fw.ready_queue['cpu'].popleft()
                    gpu = fw.ready_queue['gpu'].popleft()
                    fw.rqlock[0] = 0

                if kernels[i].partition == 0:
                    rCPU -= 1
                elif kernels[i].partition == 10:
                    rGPU -= 1
                else:
                    rCPU -=1
                    rGPU -=1

                kernels[i].partition = p

                logging.debug( "Dispatching " + str(kernels[i].name) + " with partition class " + str(kernels[i].partition))
                start_time, done_events = kernels[i].dispatch(gpu, cpu, ctxs, cmd_qs)

                events.extend(done_events)
                num_dispatched +=1
        else:
            logging.debug( "Devices unavailable")
    fw.host_synchronize(cmd_qs, events)

    return fw.dump_device_history()



if __name__ == '__main__':
    args = parse_arg(sys.argv[1:])
    task_files = args.file
    kernels = []
    for task in open(task_files,"r").readlines():
        task_src, partition, dataset = task.strip("\n").split(" ")
        info = json.loads(open(fw.SOURCE_DIR + "info/" + task_src).read())
        logging.debug( "Appending kernel" + task_src + " " + partition + " " + dataset)
        kernels.append(fw.Kernel(info, partition=int(partition), dataset=int(dataset)))

    name = "scheduling_" + args.select +"_" +str(time.time()).replace(".", "")
    dump_dev =None
    if args.dump_output_file:
        fw.dump_output = True
    if args.log:
        f_path = fw.SOURCE_DIR + 'logs/' + name + '_debug.log'
        logging.basicConfig(filename=f_path, level=logging.DEBUG)
    if args.select == "baseline":
        dump_dev = select_main(kernels, select=baseline_select)
    if args.select == "lookahead":
        dump_dev = select_main(kernels, select=look_ahead_select)
    if args.select == "adbias":
        dump_dev = select_main(kernels, select=adaptive_bias_select)

    if args.log:
        f_path = fw.SOURCE_DIR + 'logs/' + name + '_debug.log'
        logging.basicConfig(filename=f_path, level=logging.DEBUG)

    if args.graph:
        filename = fw.SOURCE_DIR + 'gantt_charts/' + name + '.png'
        fw.plot_gantt_chart_graph(dump_dev, filename)
