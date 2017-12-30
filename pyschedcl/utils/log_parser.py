#!/usr/bin/env python
import argparse
import sys
import os
import pyschedcl as fw


def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Parsing LOG info')
    parser.add_argument('-f', '--file',
                        help='Debug file dump',
                        default='last'
                        )
    parser.add_argument('-d', '--dispatch',
                        help='Displays information regarding dispatch events for kernel',
                        action="store_true")
    parser.add_argument('-p', '--partition_info',
                        help='Displays information regarding partitioning for CPU/GPU/both',
                        action="store_true")
    parser.add_argument('-c', '--callback',
                        help='Displays information regarding callback for PROBE, RESET, TRIGGERED depending on option',
                        default='NONE')
    parser.add_argument('-he', '--host_event',
                        help='Displays information regarding Host events for CPU/GPU/both',
                        action="store_true")
    parser.add_argument('-dvt', '--device_type',
                        help='Displays LOG information for specific device type - CPU/GPU/BOTH',
                        default='BOTH')
    parser.add_argument('-k', '--kernel_name',
                        help='Displays LOG information for the specified kernel',
                        default='ALL')

    return parser.parse_args(args)


if __name__ == '__main__':
    # print check_arg(sys.argv[1:])
    path = fw.SOURCE_DIR + 'logs/'
    l_file = ""

    args = check_arg(sys.argv[1:])
    d = args.dispatch
    p = args.partition_info
    c = args.callback
    he = args.host_event
    ke = args.kernel_event
    dvt = args.device_type
    k = args.kernel_name

    if args.file == 'last':
        l_file = path + max(os.listdir(path), key=lambda x: x.split("_")[-2])
    else:
        l_file = path + args.file.split("/")[-1]
    print l_file
    log_data = open(l_file, "r")

    if not d and not p and c == 'NONE' and not he and not ke and dvt == 'BOTH' and k == 'ALL':
        print log_data.read()
        print "Done"
    else:

        if k == 'ALL':
            k = ""
        if dvt == 'BOTH':
            dvc = ""
        elif dvt == 'CPU':
            dvc = "cpu"
        elif dvt == 'GPU':
            dvc = "gpu"
        log_data = open(l_file, "r")
        for line in log_data:
            if k in line:
                if dvc in line:
                    if d:
                        if "DISPATCH" in line:
                            print line
                    if p:
                        if "PARTITION" in line:
                            print line
                    if he:
                        if "HOST_EVENT" in line:
                            print line
                    if not d and not p and not he and c == 'ALL':
                        print line
                    if c == 'ALL':
                        if "CALLBACK" in line:
                            print line
                    if c == 'PROBE':
                        if "CALLBACK_PROBE" in line:
                            print line
                    if c == 'RESET':
                        if "CALLBACK_RESET" in line:
                            print line
                    if c == 'TRIGGERED':
                        if "CALLBACK_TRIGGERED" in line:
                            print line
