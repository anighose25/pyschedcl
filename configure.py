#!/usr/bin/env python

import os
import pyopencl as cl

p_path = os.path.join(os.path.dirname(__file__), 'pyschedcl')

def make_executables():
    folders=['scheduling', 'partition', 'utils']
    for fd in folders:
        fd_name=p_path+'/'+fd
        for f in os.listdir(fd_name):
            if f.endswith('.py'):
                execute_cmd = "chmod +x "+fd_name+'/'+f
                os.system(execute_cmd)

def populate_constants():

    cons_f = p_path + '/constant_pyschedcl.py'
    platforms = cl.get_platforms()
    cpu_platform = set()
    gpu_platform = set()
    num_cpu_devices = 0
    num_gpu_devices = 0
    for platform in platforms:
        devices = platform.get_devices()
        for dev in devices:
            if cl.device_type.to_string(dev.type) == "GPU":
                gpu_platform.add(platform.get_info(cl.platform_info.NAME))
                num_gpu_devices += 1
            if cl.device_type.to_string(dev.type) == "CPU":
                cpu_platform.add(platform.get_info(cl.platform_info.NAME))
                num_cpu_devices += 1

    f = open(cons_f, "w+")

    src_dir = "SOURCE_DIR = " + "\""+p_path + '/\"'
    print >> f, src_dir
    CPU_PLATFORM = "CPU_PLATFORM = " + "[" + ",".join("\"" + str(s) + "\"" for s in cpu_platform) + "]"
    print >> f, CPU_PLATFORM
    print >> f, "NUM_CPU_DEVICES = ", str(num_cpu_devices)
    GPU_PLATFORM = "GPU_PLATFORM = " + "[" + ",".join("\"" + str(s) + "\"" for s in gpu_platform) + "]"
    print >> f, GPU_PLATFORM
    print >> f, "NUM_GPU_DEVICES = ", str(num_gpu_devices)
    f.close()

if __name__ == '__main__':
    populate_constants()
    make_executables()

