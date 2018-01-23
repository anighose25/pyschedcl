import pyopencl as cl
import pyopencl.tools
import pyopencl.array
import datetime
import collections
import os
from copy import deepcopy
import threading
import mutex
import logging
import numpy as np
import constant_pyschedcl as cons
import time
import gc
import resource
from decimal import *

numpy_types = {
    "unsigned": np.uint32,
    "unsigned int": np.uint32,
    "uint": np.uint32,
    "int": np.int32,
    "long": np.int64,
    "long int": np.int64,
    "float": np.float32,
    "double": np.float64,
    "char": np.int8,
    "short": np.int16,
    "uchar": np.uint8,
    "unsigned char": np.uint8,
    "ulong": np.uint64,
    "unsigned long": np.uint64,
    "ushort": np.uint16,
    "unsigned short": np.uint16
}

VEC_TYPES = ['char16', 'char2', 'char3', 'char4', 'char8', 'double16', 'double2', 'double3', 'double4', 'double8',
             'float16', 'float2', 'float3', 'float4', 'float8', 'int16', 'int2', 'int3', 'int4', 'int8', 'long16',
             'long2', 'long3', 'long4', 'long8', 'short16', 'short2', 'short3', 'short4', 'short8', 'uchar16', 'uchar2',
             'uchar3', 'uchar4', 'uchar8', 'uint16', 'uint2', 'uint3', 'uint4', 'uint8', 'ulong16', 'ulong2', 'ulong3',
             'ulong4', 'ulong8', 'ushort16', 'ushort2', 'ushort3', 'ushort4', 'ushort8', ]



SOURCE_DIR = cons.SOURCE_DIR
MAX_GPU_ALLOC_SIZE = 0
MAX_CPU_ALLOC_SIZE = 0

for datatype in VEC_TYPES:
    numpy_types[datatype] = eval('cl.array.vec.{}'.format(datatype))

system_cpus, system_gpus = 0, 0
nGPU, nCPU = 0, 0
device_history = {"gpu": [], "cpu": []}
ready_queue = {"gpu": collections.deque(), "cpu": collections.deque()}
cs = mutex.mutex()
user_defined = dict()
dump_output = False

bolt = [0]
rqlock = [0]

callback_queue = {}


def compare_and_swap(testval, newval):
    """
    Software test and lock routine used for enforcing mutual exclusion across multiple callback threads.

    :param testval: List representing testing lock value
    :type testval: list
    :param newval: List representing new lock value
    :type newval: list
    :return: Returns list representing old lock value.
    :rtype: list
    """
    global bolt
    oldval = bolt[0]
    if oldval == testval:
        bolt[0] = newval
    return oldval


def test_and_set(testval, newval):
    """
    Software test and lock routine used for enforcing mutual exclusion across callback threads and main host thread while accessing ready queue and device counters.

    :param testval: List representing testing lock value
    :type testval: list
    :param newval: List representing new lock value
    :type newval: list
    :return: Returns list representing old lock value.
    :rtype: list
    """
    global rqlock
    oldval = rqlock[0]
    if oldval == testval:
        rqlock[0] = newval
    return oldval


def blank_fn(*args, **kwargs):
    """
    Does nothing. Used as dummy function for callback events.

    """
    pass


class HostEvents(object):
    """
    Class for storing timing information of various events associated with a kernel.

    :ivar dispatch_start: Start Timestamp for dispatch function
    :ivar dispatch_end:  End Timestamp for dispatch function
    :ivar create_buf_start: Start Timestamp for Creation of Buffers
    :ivar create_buf_end: End Timestamp for Creation of Buggers
    :ivar write_start: Start TimeStamp for Enqueuing Write Buffer Commands on Command Queue
    :ivar write_end: End Timestamp for Writing of Buffers to Device
    :ivar ndrange_start: Start TimeStamp for Launching Kernel
    :ivar ndrange_end: End Timestamp for when kernel execution is finished on device
    :ivar read_start:  Start TimeStamp for Enqueuing Read Buffer Commands on Command Queue
    :ivar read_end: End TimeStamp for Reading of Buffers from Device to Host
    :ivar kernel_name: Name of kernel
    :ivar kernel_id: Unique id for kernel
    :ivar dispatch_id: Dispatch id for kernel
    """

    def __init__(self, kernel_name='', kernel_id='', dispatch_id='', dispatch_start=None, dispatch_end=None,
                 create_buf_start=None, create_buf_end=None, write_start=None, write_end=None, ndrange_start=None,
                 ndrange_end=None, read_start=None, read_end=None):
        """
        Initialise attributes of HostEvents class .

        """
        self.dispatch_start = dispatch_start
        self.dispatch_end = dispatch_end
        self.create_buf_start = create_buf_start
        self.create_buf_end = create_buf_end
        self.write_start = write_start
        self.write_end = write_end
        self.ndrange_start = ndrange_start
        self.ndrange_end = ndrange_end
        self.read_start = read_start
        self.read_end = read_end
        self.kernel_name = kernel_name
        self.kernel_id = kernel_id
        self.dispatch_id = dispatch_id

    def __str__(self):

        a = deepcopy(self.__dict__)
        for i in a:
            a[i] = str(a[i])
        return str(a)

    def __repr__(self):

        return str(self)


def dump_device_history():
    """
    Dumps device history to debug.log file.
    """

    debug_strs = []
    min_timestamp = Decimal('Infinity')
    max_timestamp = 0.0
    for dev in ['gpu', 'cpu']:
        for device_id in range(len(device_history[dev])):
            for host_event in device_history[dev][device_id]:
                kernel_id = host_event.kernel_id
                kernel_name = host_event.kernel_name
                write_start = "%.20f" % host_event.write_start
                min_timestamp = min(min_timestamp, Decimal(write_start))
                write_end = "%.20f" % host_event.write_end
                ndrange_start = "%.20f" % host_event.ndrange_start
                ndrange_end = "%.20f" % host_event.ndrange_end
                read_start = "%.20f" % host_event.read_start
                read_end = "%.20f" % host_event.read_end
                max_timestamp = max(max_timestamp, Decimal(read_end))
                debug_str = "HOST_EVENT " + dev + " " + str(device_id) + " " + str(
                    kernel_id) + "," + kernel_name + " " + write_start + " " + write_end + " " + \
                            ndrange_start + " " + ndrange_end + " " + read_start + " " + read_end
                logging.debug(debug_str)
                # print debug_str
                debug_strs.append(debug_str)
    profile_time = max_timestamp - min_timestamp
    print "span_time " + str(profile_time)

    return debug_strs


def partition_round(elms, percent, exact=-1, total=100, *args, **kwargs):
    """
    Partitions dataset in a predictable way.

    :param elms: Total Number of elements
    :type elms: Integer
    :param percent: Percentage of problem space to be processed on one device
    :param type: Integer
    :param exact: Flag that states whether percentage of problem space is greater than 50 or not (0 for percent < 50, 1 for percent >= 50)
    :param type: Integer
    :param total: Percentage of total problem space (Default value: 100)
    :type total: Integer
    :return: Number of elements of partitioned dataset
    :rtype: Integer
    """
    if elms < 100:
        factor = 10
        x = elms / 10
    else:
        factor = 1
        x = elms / 100
    if exact == -1:
        exact = 0 if percent > 50 else 1
    if elms % 2 == 0:
        if percent == 50:
            logging.debug(
                "PARTITION: get_slice_values -> multiple_round -> partition_round (if percent=50) returns: %d",
                elms / 2)
            return elms / 2
        elif exact == 0:
            b = x * (total - percent) / factor
            return partition_round(elms, total) - b if total != 100 else elms - b
        elif exact == 1:
            logging.debug("PARTITION: get_slice_values -> multiple_round -> partition_round (if exact=1) returns: %d",
                          x * percent / factor)
            return x * percent / factor
    else:
        if percent > 50:
            return partition_round(elms - 1, percent, exact, total)
        else:
            return partition_round(elms - 1, percent, exact, total) + 1


part_round = partition_round


def multiple_round(elms, percent, multiples, **kwargs):
    """
    Partitions such that the partitioned datasets are multiples of given number.

    :param elms: Total number of elements of buffer
    :type elms: Integer
    :param percent: Percentage of problem space to be processed by one device
    :type percent: Integer
    :param multiples: List of integers representing partition multiples for each dimension
    :type multiples: list of integers
    :param kwargs:
    :return: Percentage of buffer space to be partitioned for one device
    :rtype: Integer
    """
    for multiple in multiples:
        if elms % multiple == 0 and elms > multiple:
            x = elms / multiple
            return partition_round(x, percent, **kwargs) * multiple


def ctype(dtype):
    """
    Convert a string datatype to corresponding Numpy datatype. User can also define new datatypes using user_defined parameter.

    :param dtype: Datatype name
    :type dtype: String
    :return: Numpy Datatype corresponding to dtype
    """
    global numpy_types
    try:
        return numpy_types[dtype]
    except:
        Exception("Data Type {} not defined".format(dtype))


def make_ctype(dtype):
    """
    Creates a vector datatype.

    :param dtype: Datatype name
    :type dtype: String
    :return: numpy datatype corresponding to dtype
    """
    global numpy_types
    if dtype in VEC_TYPES:
        return eval('cl.array.vec.make_{}'.format(dtype))
    else:
        return numpy_types[dtype]


def make_user_defined_dtype(ctxs, name, definition):

    global numpy_types
    if type(definition) is str:
        if name not in numpy_types:
            if definition not in numpy_types:
                raise Exception(
                    "Cant recognize definition {0} should be one of {1}".format(definition, numpy_types.keys()))
            else:
                numpy_types[name] = numpy_types[definition]
        else:
            if numpy_types[definition] != numpy_types[name]:
                raise Exception(
                    "Conflicting definitions {0} and {1} for {2}".format(numpy_types[definition], numpy_types[name],
                                                                         name))
    elif type(definition) is dict:
        raise NotImplementedError
        struct = np.dtype(map(lambda k, v: (k, numpy_types[v]), definition.items()))
        struct, c_decl = cl.tools.match_dtype_to_c_struct(ctxs['gpu'][0])

    else:
        raise Exception('Expected data type definition to be string or dict but got {}'.format(str(type)))


def notify_callback(kernel, device, dev_no, event_type, events, host_event_info, callback=blank_fn):
    """
    A wrapper function that generates and returns a call-back function based on parameters. This callback function is run whenever a enqueue operation finishes execution. User can suitably modify callback functions to carry out further processing run after completion of enqueue read buffers operation indicating completion of a kernel task.

    :param kernel: Kernel Object
    :type kernel:  pyschedcl.Kernel object
    :param device: Device Type (CPU or GPU)
    :type device: String
    :param dev_no: PySchedCL specific device id
    :type dev_no: Integer
    :param event_type: Event Type (Write, NDRange, Read)
    :type event_type: String
    :param events: List of Events associated with an Operation
    :type events: list of pyopencl.Event objects
    :param host_event_info: HostEvents object associated with Kernel
    :type host_event_info: HostEvents
    :param callback: Custom Callback function for carrying out further post processing if required.
    :type callback: python function
    """

    def cb(status):
        global bolt
        try:
            global callback_queue
            tid = threading.currentThread()
            callback_queue[tid] = False

            while (compare_and_swap(0, 1) == 1):
                debug_probe_string = "CALLBACK_PROBE : " + kernel.name + " " + str(device) + " " + str(
                    event_type) + " event"

                logging.debug(debug_probe_string)
            debug_trigger_string = "CALLBACK_TRIGGERED : " + kernel.name + " " + str(
                event_type) + " execution finished for device " + str(device)
            logging.debug(debug_trigger_string)

            if event_type == 'WRITE':
                host_event_info.write_end = time.time()
            elif event_type == 'READ':
                host_event_info.read_end = time.time()
                global device_history
                logging.debug("CALLBACK : " +str(host_event_info))
                logging.debug("CALLBACK : Pushing info onto " + str(device) + str(dev_no))
                device_history[device][dev_no].append(host_event_info)

                if kernel.multiple_break:
                    if device == 'cpu':
                        kernel.chunks_cpu -= 1
                    else:
                        kernel.chunks_gpu -= 1
                    if device == 'cpu' and kernel.chunks_cpu == 0:
                        kernel.release_buffers(device)

                    if device == 'gpu' and kernel.chunks_gpu == 0:
                        kernel.release_buffers(device)
                else:
                    kernel.release_buffers(device)
                kernel.chunks_left -= 1
                if kernel.chunks_left == 0:
                    global dump_output
                    if dump_output:
                        import pickle
                        filename = SOURCE_DIR + "output/" + kernel.name + "_" + str(kernel.partition) + "_" + str(kernel.dataset) + ".pickle"
                        print "Dumping Pickle"
                        with open(filename, 'wb') as handle:
                            pickle.dump(kernel.data['output'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        print "Dumped Pickle"
                    kernel.release_host_arrays()

                global ready_queue
                while (test_and_set(0, 1)):
                    pass
                ready_queue[device].append(dev_no)
                if device == 'gpu':
                    global nGPU
                    nGPU += 1
                else:
                    global nCPU
                    nCPU += 1
                global rqlock
                rqlock[0] = 0

            elif event_type == 'KERNEL':
                host_event_info.ndrange_end = time.time()
            callback_queue[tid] = True
            bolt[0] = 0
            logging.debug("CALLBACK_RESET : " + kernel.name + " Resetting bolt value by " + str(device) + " " + str(
                event_type) + " event")

        except TypeError:
            pass

    return cb


def generate_unique_id():
    """
    Generates and returns a unique id string.

    :return: Unique ID
    :rtype: String
    """
    import uuid
    return str(uuid.uuid1())


class Kernel(object):
    """
    Class to handle all operations performed on OpenCL kernel.

    :ivar dataset: An integer representing size of the data on which kernel will be dispatched.
    :ivar id: An id that is used identify a kernel uniquely.
    :ivar eco: A dictionary mapping between size of dataset and Estimated Computation Overhead
    :ivar name: Name of the Kernel
    :ivar src: Path to the Kernel source file.
    :ivar partition: An integer denoting the partition class of the kernel.
    :ivar work_dimension: Work Dimension of the Kernel.
    :ivar global_work_size: A list denoting global work dimensions along different axes.
    :ivar local_work_size: A list denoting local work dimensions along different axes.
    :ivar buffer_info: Properties of Buffers
    :ivar input_buffers: Dictionaries containing actual cl.Buf f er objects.
    :ivar output_buffers: Dictionaries containing actual cl.Buf f er objects.
    :ivar io_buffers: Dictionaries containing actual cl.Buf f er objects.
    :ivar data: Numpy Arrays maintaining the input and output data of the kernels.
    :ivar buffer_deps: Dictionary mapping containing buffer dependencies.
    :ivar variable_args: Data corresponding to Variable arguments of the kernel.
    :ivar local_args: Information regarding Local Arguments of the kernel.
    :ivar kernel objects: Dictionary mapping between devices and compiled and built pyopencl.Kernel objects.
    :ivar events: Dictionary containing pyschedcl.KEvents.
    :ivar source: String containing contents of kernel file.
    :ivar clevents: Dictionary containing pyopencl.Events.
    """

    def __init__(self, src, dataset=1024, partition=None, identifier=None):
        """
        Initialise attributes of Kernel event.

        """
        self.dataset = dataset
        if 'id' in src:
            self.id = src['id']
        else:
            self.id = generate_unique_id()
        if identifier is not None:
            self.id = identifier
        if 'ecos' in src and str(dataset) in src['ecos']:
            self.eco = src['ecos'][str(dataset)]
        elif 'eco' in src:
            self.eco = src['eco']
        else:
            self.eco = 1
        self.name = src['name']
        self.src = src['src']
        self.partition = src['partition']
        if partition is not None:
            self.partition = partition
        else:
            partition = self.partition
        self.work_dimension = src['workDimension']
        self.global_work_size = src['globalWorkSize']
        if type(self.global_work_size) in [str, unicode]:
            self.global_work_size = eval(self.global_work_size)
        if type(self.global_work_size) is int:
            self.global_work_size = [self.global_work_size]
        if 'localWorkSize' in src:
            self.local_work_size = src['localWorkSize']
        else:
            self.local_work_size = []
        if type(self.local_work_size) in [str, unicode]:
            self.local_work_size = eval(self.local_work_size)
        elif type(self.local_work_size) is int:
            self.local_work_size = [self.local_work_size]
        self.buffer_info = dict()
        if 'inputBuffers' in src:
            self.buffer_info['input'] = src['inputBuffers']
        else:
            self.buffer_info['input'] = []
        if 'outputBuffers' in src:
            self.buffer_info['output'] = src['outputBuffers']
        else:
            self.buffer_info['output'] = []
        if 'ioBuffers' in src:
            self.buffer_info['io'] = src['ioBuffers']
        else:
            self.buffer_info['io'] = []
        self.input_buffers = {'gpu': dict(), 'cpu': dict()}
        self.output_buffers = {'gpu': dict(), 'cpu': dict()}
        self.io_buffers = {'gpu': dict(), 'cpu': dict()}
        self.data = {}
        self.buffer_deps = {}
        if 'varArguments' in src:
            self.variable_args = deepcopy(src['varArguments'])
            self.vargs = src['varArguments']
        else:
            self.variable_args = []
            self.vargs = []
        if 'cpuArguments' in src:
            self.cpu_args = src['cpuArguments']
            print "Ignoring CPU Arguments"
        if 'gpuArguments' in src:
            self.gpu_args = src['gpuArguments']
            print "Ignoring GPU Arguments"
        if 'localArguments' in src:
            self.local_args = src['localArguments']
            for i in range(len(self.local_args)):
                self.local_args[i]['size'] = eval(self.local_args[i]['size'])
        else:
            self.local_args = []
            # self.buffer_info['local'] = deepcopy(self.local_args)
        self.kernel_objects = dict()
        for btype in ['input', 'output', 'io']:
            for i in range(len(self.buffer_info[btype])):
                if type(self.buffer_info[btype][i]['size']) in [str, unicode]:
                    self.buffer_info[btype][i]['size'] = eval(self.buffer_info[btype][i]['size'])
                if 'chunk' in self.buffer_info[btype][i] and type(self.buffer_info[btype][i]['chunk']) in [str,
                                                                                                           unicode]:
                    self.buffer_info[btype][i]['chunk'] = eval(self.buffer_info[btype][i]['chunk'])
                self.buffer_info[btype][i]['create'] = True
                self.buffer_info[btype][i]['enq_write'] = True
                self.buffer_info[btype][i]['enq_read'] = True
                if 'from' in self.buffer_info[btype][i]:
                    self.buffer_deps[self.buffer_info[btype][i]['pos']] = (self.buffer_info[btype][i]['from']['kernel'],
                                                                           self.buffer_info[btype][i]['from']['pos'])

        self.partition_multiples = self.get_partition_multiples()
        self.events = {'gpu': dict(), 'cpu': dict()}
        self.source = None
        # self.clevents = {'gpu': dict(), 'cpu': dict()}
        self.chunks_left = 1
        self.multiple_break = False
        self.chunks_cpu = 0
        self.chunks_gpu = 0

    def get_num_global_work_items(self):
        """
        Returns the total number of global work items based on global work size.

        :return: Total number of global work items considering all dimensions
        :rtype: Integer
        """
        i = 1
        for j in self.global_work_size:
            i *= j
        return i

    # TODO: Modify to handle dependent buffers.

    def release_host_arrays(self):
        """
        Forcefully releases all host array data after completion of a kernel task
        """
        for array_type in self.data.keys():
            for array in self.data[array_type]:
                del array
        logging.debug("Releasing host arrays")
        del self.data
        gc.collect()

    def release_buffers(self, obj):
        """
        Releases all buffers of a Kernel Object for a particular device given in obj

        :param obj: Specifies Kernel object
        :type obj: String
        """
        debug_str = "Releasing buffers of " + self.name + " on " + obj
        logging.debug(debug_str)
        for i, buff in self.input_buffers[obj].iteritems():
            if buff is not None:
                buff.release()
        for i, buff in self.output_buffers[obj].iteritems():
            if buff is not None:
                buff.release()

        for i, buff in self.io_buffers[obj].iteritems():
            if buff is not None:
                buff.release()

    def eval_vargs(self, partition=None, size_percent=0, offset_percent=0, reverse=False, exact=-1, total=100):
        """
        Method to evaluate kernel arguments. Evaluates variable kernel arguments from the specification file if they are an expression against size percent.

        :param partition: Partition Class Value
        :type partition: Integer
        :param size_percent: Percentage of problem space to be processed on device
        :type size_percent: Integer
        :param offset_percent: Offset Percentage required
        :type offset_percent: Integer
        :param reverse: Flag for inverting size and offset calculations for respective devices
        :type reverse: Boolean
        :param exact: Flag that states whether percentage of problem space is greater than 50 or not (0 for percent < 50, 1 for percent >= 50)
        :param type: Integer
        :param total: Percentage of total problem space (Default value: 100)
        :type total: Integer
        """

        def partition_round(elms, percent, exact=exact, total=total):
            return part_round(elms, percent, exact, total)

        if partition is not None:
            size_percent = partition * 10
            offset_percent = 0
            if reverse:
                offset_percent = partition * 10
                partition = 10 - partition
                size_percent = partition * 10
        dataset = self.dataset
        if self.vargs:
            for i in range(len(self.vargs)):
                if type(self.vargs[i]['value']) in [str, unicode]:
                    self.variable_args[i]['value'] = eval(self.vargs[i]['value'])

    def get_partition_multiples(self):
        """
        Determines partition multiples based on work dimension. This method returns a list of numbers based on global work size and local work size according to which the partition sizes will be determined.

        :return: List of integers representing partition multiples for each dimension

        """
        multiples = [1]
        if self.work_dimension == 1:
            if not self.local_work_size:
                multiples = [1]
            else:
                multiples = [self.local_work_size[0], 1]
        elif self.work_dimension == 2:
            if not self.local_work_size:
                multiples = [self.global_work_size[1], 1]
            else:
                multiples = [self.local_work_size[0] * self.global_work_size[1], self.global_work_size[1],
                             self.local_work_size[0], 1]
        elif self.work_dimension == 3:
            if not self.local_work_size:
                multiples = [self.global_work_size[1] * self.global_work_size[2], self.global_work_size[1], 1]
        else:
            print("Invalid Work Dimension")
        return multiples

    def build_kernel(self, gpus, cpus, ctxs):
        """
        Builds Kernels from the directory kernel_src/ for each device and stores binaries in self.kernel_objects dict.

        :param gpus: List of OpenCL GPU Devices
        :type gpus: list of pyopencl.device objects
        :param cpus: List of OpenCL GPU Devices
        :type cpus: list of pyopencl.device objects
        :param ctxs: Dictionary of contexts keyed by device types
        :type ctxs: dict
        :return: Dictionary of key: value pairs where key is device type and value is the device specific binary compiled for the kernel
        :rtype: dict
        """

        src_path = SOURCE_DIR + 'kernel_src/' + self.src
        if not os.path.exists(src_path):
            raise IOError('Kernel Source File %s not Found' % src_path)
        self.source = open(src_path).read()
        programs = dict()
        for key in ctxs.keys():
            if ctxs[key] is not None:
                programs[key] = cl.Program(ctxs[key], self.source)
        if len(gpus) != 0:
            programs['gpu'].build(devices=gpus)
        if len(cpus) != 0:
            programs['cpu'].build(devices=cpus)
        for key in programs.keys():
            self.kernel_objects[key] = cl.Kernel(programs[key], self.name)
        return programs

    def random_data(self, low=0, hi=4096):
        """
        Generates random data numpy arrays according to buffer type so that it can be enqueued to buffer. Can be used for testing. Will not generate random data for those buffers that are already enqueued. Creates empty arrays for read-only buffers. Populates values for self.data dictionary.

        :param low: Lower value of data array size
        :type low: Integer
        :param hi: Higher value of data array size
        :type hi: Integer

        """
        import numpy as np
        integers = ['int', 'uint', 'unsigned', 'long', 'unsigned int', 'long int', 'int16', 'int2', 'int3', 'int4',
                    'int8', 'long16', 'long2', 'long3', 'long4', 'long8', 'short16', 'short2', 'short3', 'short4',
                    'short8', 'uint16', 'uint2', 'uint3', 'uint4', 'uint8', 'ulong16', 'ulong2', 'ulong3',
                    'ulong4', 'ulong8', 'ushort16', 'ushort2', 'ushort3', 'ushort4', 'ushort8']
        characters = ['char16', 'char2', 'char3', 'char4', 'char8', 'uchar16', 'uchar2',
                      'uchar3', 'uchar4', 'uchar8']
        for btype in ['input', 'io']:
            self.data[btype] = []
            for i in range(len(self.buffer_info[btype])):
                if not self.buffer_info[btype][i]['enq_write']:
                    self.data[btype].append(None)
                elif self.buffer_info[btype][i]['type'] in integers:
                    self.data[btype].append(
                        np.random.randint(low, hi, size=[self.buffer_info[btype][i]['size']]).astype(
                            ctype(self.buffer_info[btype][i]['type']), order='C'))
                elif self.buffer_info[btype][i]['type'] in characters:
                    self.data[btype].append(
                        np.random.randint(low, 128, size=[self.buffer_info[btype][i]['size']]).astype(
                            ctype(self.buffer_info[btype][i]['type']), order='C'))
                else:
                    self.data[btype].append(np.random.rand(self.buffer_info[btype][i]['size']).astype(
                        ctype(self.buffer_info[btype][i]['type']), order='C'))
        self.data['output'] = []
        for i in range(len(self.buffer_info['output'])):
            self.data['output'].append(
                np.zeros(self.buffer_info['output'][i]['size'], dtype=ctype(self.buffer_info['output'][i]['type']),
                         order='C'))

    def load_data(self, data):
        """
        Populates all host input arrays with given data.

        :param data: Dictionary structure comprising user defined data for host input arrays
        :type data: dict
        """
        import numpy as np
        for key in data.keys():
            self.data[key] = []
            for i in range(len(self.buffer_info[key])):
                self.data[key].append(data[key][i])
        self.data['output'] = []
        for i in range(len(self.buffer_info['output'])):
            self.data['output'].append(
                np.zeros(self.buffer_info['output'][i]['size'], dtype=ctype(self.buffer_info['output'][i]['type']),
                         order='C'))

    def get_data(self, pos):
        """
        Returns the data of a particular kernel argument given its parameter position in the kernel.  Used to load dependent data.

        :param pos: Position of buffer argument in list of kernel arguments
        :type pos: Integer
        :return: Data stored in buffer specified by parameter position in kernel
        :rtype: Numpy array
        """
        for key in self.buffer_info.keys():
            for i in range(len(self.buffer_info[key])):
                if self.buffer_info[key][i]['pos'] == pos:
                    if key in self.data.keys():
                        return self.data[key][i]
                    else:
                        raise KeyError

    def get_buffer_info_location(self, pos):
        """
        Returns buffer_info location at given position. Used to make reusable buffers.

        :param pos: Position of buffer argument in list of kernel arguments
        :type pos: Integer
        :return: Tuple(key, i) where key represents type of buffer access and i represents the id for that buffer in self.buffer_info[key]
        :rtype: Tuple
        """
        for key in self.buffer_info.keys():
            for i in range(len(self.buffer_info[key])):
                if self.buffer_info[key][i]['pos'] == pos:
                    return key, i

    def get_buffer_info(self, pos):
        """
        Returns buffer information stored in Kernel specification file for buffer at given position in Kernel Arguments. Used to make reusable buffers.

        :param pos: Position of buffer argument in list of kernel arguments
        :type pos: Integer
        :return: Returns a dictionary of key:value pairs representing information of selected buffer in self.buffer_info
        :rtype: dict
        """
        key, i = self.get_buffer_info_location(pos)
        return self.buffer_info[key][i]

    def get_buffer(self, pos):
        """
        Returns cl.Buffer objects given its parameter position in the kernel.

        :param pos: Position of buffer argument in list of kernel arguments
        :type: Integer
        :return: PyOpenCL Buffer Object for selected buffer
        :rtype: pyopencl.Buffer
        """
        btype, i = self.get_buffer_info_location(pos)
        if btype is 'input':
            return {'gpu': self.input_buffers['gpu'].get(i, None), 'cpu': self.input_buffers['cpu'].get(i, None)}
        elif btype is 'io':
            return {'gpu': self.io_buffers['gpu'].get(i, None), 'cpu': self.io_buffers['cpu'].get(i, None)}
        elif btype is 'output':
            return {'gpu': self.output_buffers['gpu'].get(i, None), 'cpu': self.output_buffers['cpu'].get(i, None)}
        else:
            raise Exception('Expected buffer to be either input, io or output but got ' + str(btype))

    def get_slice_values(self, buffer_info, size_percent, offset_percent, **kwargs):
        """
        Returns Element offset, size based on size_percent, offset_percent.

        :param buffer_info: Dictionary of key:value pairs representing information of one buffer
        :type buffer_info: dict
        :param size_percent: Size of buffer to be processed (in percentage)
        :type size_percent: Float
        :param offset_percent: Offset for given buffer (in percentage)
        :type offset_percent: Float
        :param kwargs:
        :type kwargs:
        :return: Tuple representing element offset and number of elements
        :rtype: Tuple

        """
        logging.debug("PARTITION : %s get_slice_values -> Original Buffer Size: %s", self.name, buffer_info['size'])
        if 'chunk' in buffer_info:
            partition_multiples = [buffer_info['chunk']] + self.partition_multiples
        else:
            partition_multiples = self.partition_multiples

        if buffer_info['break'] != 1:
            eo = 0
            ne = buffer_info['size']
        else:

            if 'exact' not in kwargs:
                eo = multiple_round(buffer_info['size'], offset_percent, partition_multiples, **kwargs)
                ne = multiple_round(buffer_info['size'], size_percent, partition_multiples, **kwargs)
            else:
                eo = multiple_round(buffer_info['size'], offset_percent, partition_multiples, exact=1)
                ne = multiple_round(buffer_info['size'], size_percent, partition_multiples, **kwargs)
        return eo, ne

    def create_buffers(self, ctx, obj, size_percent=100, offset_percent=0, **kwargs):
        """
        Creates buffers for a Context while respecting partition information provided for every buffer associated with the kernel.

        :param ctx: PyOpenCL Context for either a CPU or a GPU device
        :type ctx: pyopencl.Context
        :param obj: Device Type
        :type obj: String
        :param size_percent: Size of valid buffer space for device
        :type size_percent: Float
        :param offset_percent: Offset parameter representing
        :type offset_percent:
        :param kwargs:
        :type kwargs:
        """
        logging.debug("PARTITION : Creating Input Buffers %s",obj)
        for i in range(len(self.buffer_info['input'])):
            if self.buffer_info['input'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['input'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s : %s Create_Input_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Create_Input_Buffers_number_of_elements : %s", obj, self.name, ne)
                self.input_buffers[obj][i] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                                       size=self.data['input'][i][eo:eo + ne].nbytes)

        logging.debug("PARTITION : Creating Output Buffers %s", obj)
        for i in range(len(self.buffer_info['output'])):
            if self.buffer_info['output'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['output'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s: %s Create_Output_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s: %s Create_Output_Buffers_number_of_elements : %s", obj, self.name, ne)
                self.output_buffers[obj][i] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                                        size=self.data['output'][i][eo:eo + ne].nbytes)
        logging.debug("PARTITION : Creating IO Buffers %s",obj)
        for i in range(len(self.buffer_info['io'])):
            if self.buffer_info['io'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['io'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s: %s Create_IO_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s: %s Create_IO_Buffers_number_of_elements : %s", obj, self.name, ne)
                self.io_buffers[obj][i] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                                    size=self.data['io'][i][eo:eo + ne].nbytes)

    def set_kernel_args(self, obj):
        """
        Sets Kernel Arguments (Buffers and Variable Arguments).

        :param obj: Device Type (cpu or gpu)
        :type obj: String
        """
        for i in range(len(self.input_buffers[obj])):
            self.kernel_objects[obj].set_arg(self.buffer_info['input'][i]['pos'], self.input_buffers[obj][i])
        for i in range(len(self.output_buffers[obj])):
            self.kernel_objects[obj].set_arg(self.buffer_info['output'][i]['pos'], self.output_buffers[obj][i])
        for i in range(len(self.io_buffers[obj])):
            self.kernel_objects[obj].set_arg(self.buffer_info['io'][i]['pos'], self.io_buffers[obj][i])
        for i in range(len(self.variable_args)):

            if type(self.variable_args[i]['value']) is list:
                self.kernel_objects[obj].set_arg(self.variable_args[i]['pos'],
                                                 make_ctype(self.variable_args[i]['type'])(
                                                     *self.variable_args[i]['value']))
            else:
                self.kernel_objects[obj].set_arg(self.variable_args[i]['pos'],
                                                 make_ctype(self.variable_args[i]['type'])(
                                                     self.variable_args[i]['value']))

        for i in range(len(self.local_args)):
            self.kernel_objects[obj].set_arg(self.local_args[i]['pos'],
                                             cl.LocalMemory(make_ctype(self.local_args[i]['type'])().nbytes * (
                                                 self.local_args[i]['size'])))

    def enqueue_write_buffers(self, queue, q_id, obj, size_percent=100, offset_percent=0, deps=None, **kwargs):
        """
        Enqueues list of write buffer operations to the OpenCL Runtime.

        :param queue: Command Queue for a CPU or a GPU device
        :type queue: pyopencl.CommandQueue
        :param q_id: ID of queue
        :type q_id: Integer
        :param obj: Device Type (CPU or GPU)
        :type obj: String
        :param size_percent: Percentage of problem space to be processed
        :type size_percent: Integer
        :param offset_percent: Percentage for offset required for selection of host array space to be copied to buffer
        :type offset_percent: Integer
        :param deps: Initial PyOpenCL Event on which subsequent write operations will be dependent stored in a list
        :type deps: list of pyopencl.Event Objects
        :param kwargs:
        :type kwargs:
        :return: Barrier Event signifying end of write operation
        :rtype: pyopencl.event
        """

        iev, ioev = [None] * len(self.input_buffers[obj]), [None] * len(self.io_buffers[obj])
        depends = [None] * (len(self.input_buffers[obj]) + len(self.io_buffers[obj]))
        if len(depends) == 0:
            depends = [None]
        if deps:
            depends[0] = deps
        logging.debug("PARTITION : Enqueuing Write Buffers %s",obj)
        kwargs['host_event'].write_start = time.time()
        start_barrier_event = cl.enqueue_barrier(queue, wait_for=depends[0])
        for i in range(len(self.input_buffers[obj])):
            if self.buffer_info['input'][i]['enq_write']:
                eo, ne = self.get_slice_values(self.buffer_info['input'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s : %s Enqueue_Write_Input_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Write_Input_Buffers_number_of_elements : %s", obj, self.name,
                              ne)
                iev[i] = cl.enqueue_copy(queue, self.input_buffers[obj][i], self.data['input'][i][eo:eo + ne],
                                         is_blocking=False, wait_for=depends[i])
        if self.input_buffers[obj]:
            depends = [None] * len(self.io_buffers[obj])
        j = len(self.input_buffers[obj])
        for i in range(len(self.io_buffers[obj])):
            if self.buffer_info['io'][i]['enq_write']:
                eo, ne = self.get_slice_values(self.buffer_info['io'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s : %s Enqueue_Write_IO_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Write_IO_number_of_elements : %s", obj, self.name, ne)
                ioev[i] = cl.enqueue_copy(queue, self.io_buffers[obj][i], self.data['io'][i][eo:eo + ne],
                                          is_blocking=False, wait_for=depends[i + j])
        iev.extend(ioev)
        logging.debug("PARTITION : Number of write buffers %d" % (len(iev)))
        barrier_event = cl.enqueue_barrier(queue, wait_for=iev)
        barrier_event.set_callback(cl.command_execution_status.COMPLETE,
                                   notify_callback(self, obj, q_id, 'WRITE', iev, host_event_info=kwargs['host_event']))
        return barrier_event

    def enqueue_nd_range_kernel(self, queue, q_id, obj, size_percent=100, offset_percent=0, deps=None, **kwargs):
        """
        Enqueues ND Range Kernel Operation for a kernel

        :param queue: OpenCL Command Queue
        :type queue: pyopencl.CommandQueue
        :param q_id: Framework specific id assigned to Command Queue
        :type q_id: Integer
        :param obj: OpenCL Device Type
        :type obj: String
        :param size_percent: Percentage of problem space to be processed
        :type size_percent: Integer
        :param offset_percent: Percentage for offset required for selection of host array space to be copied to buffer
        :type offset_percent: Integer
        :param deps:  PyOpenCL Event on which subsequent ndrange operations will be dependent on stored in a list
        :type deps: list of pyopencl.Event Objects
        :param kwargs:
        :type kwargs:
        :return: Barrier Event signifying end of ndrange operation
        :rtype: pyopencl.event
        """

        global_work_offset = [0] * len(self.global_work_size)
        global_work_size = deepcopy(self.global_work_size)
        global_work_size[0] = multiple_round(global_work_size[0], size_percent, self.partition_multiples, **kwargs)
        # global_work_offset[0] = multiple_round(global_work_size[0], offset_percent,self.partition_multiples, **kwargs)
        depends = [None]

        if deps:
            depends[0] = deps
        ev = None
        kwargs['host_event'].ndrange_start = time.time()
        if self.local_work_size:
            ev = cl.enqueue_nd_range_kernel(queue, self.kernel_objects[obj], global_work_size,
                                            self.local_work_size, wait_for=depends[0])
        else:
            ev = cl.enqueue_nd_range_kernel(queue, self.kernel_objects[obj], global_work_size, None,
                                            wait_for=depends[0])

        barrier_event = cl.enqueue_barrier(queue, wait_for=[ev])
        barrier_event.set_callback(cl.command_execution_status.COMPLETE,
                                   notify_callback(self, obj, q_id, 'KERNEL', [ev],
                                                   host_event_info=kwargs['host_event']))
        return barrier_event

    def enqueue_read_buffers(self, queue, q_id, obj, size_percent=100, offset_percent=0, deps=None, callback=blank_fn,
                             **kwargs):
        """
        Enqueue Read Buffer operations for a kernel.

        :param queue: OpenCL Command Queue
        :type queue: pyopencl.CommandQueue
        :param q_id: Framework specific id assigned to Command Queue
        :type q_id: Integer
        :param obj: OpenCL Device Type
        :type obj: String
        :param size_percent: Percentage of problem space to be processed
        :type size_percent: Integer
        :param offset_percent: Percentage for offset required for selection of host array space to be copied to buffer
        :type offset_percent: Integer
        :param deps:  PyOpenCL Event on which subsequent ndrange operations will be dependent on stored in a list
        :type deps: list of pyopencl.Event Objects
        :param callback: Custom callback function
        :type callback: python function
        :param kwargs:
        :type kwargs:
        :return: Barrier Event signifying end of read operation
        :rtype: pyopencl.event
        """

        oev, ioev = [None] * len(self.output_buffers[obj]), [None] * len(self.io_buffers[obj])
        logging.debug("PARTITION : Enqueuing Read Buffers %s",obj)
        depends = [None] * (len(self.output_buffers[obj]) + len(self.io_buffers[obj]))
        if len(depends) == 0:
            depends = [None]
        if deps:
            depends[0] = deps
        kwargs['host_event'].read_start = time.time()
        for i in range(len(self.output_buffers[obj])):
            if self.buffer_info['output'][i]['enq_read']:
                eo, ne = self.get_slice_values(self.buffer_info['output'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s : %s Enqueue_Read_Output_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Read_Output_Buffers_number_of_elements : %s", obj, self.name,
                              ne)
                oev[i] = cl.enqueue_copy(queue, self.data['output'][i][eo:eo + ne], self.output_buffers[obj][i],
                                         is_blocking=False, wait_for=depends[i])
        j = len(self.output_buffers[obj])
        for i in range(len(self.io_buffers[obj])):
            if self.buffer_info['io'][i]['enq_read']:
                eo, ne = self.get_slice_values(self.buffer_info['io'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s : %s Enqueue_Read_IO_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Read_IO_Buffers_number_of_elements : %s", obj, self.name, ne)
                ioev[i] = cl.enqueue_copy(queue, self.data['io'][i][eo:eo + ne], self.io_buffers[obj][i],
                                          is_blocking=False, wait_for=depends[i + j])
        oev.extend(ioev)
        logging.debug("PARTITION : Number of read buffers %d" % (len(oev)))
        barrier_event = cl.enqueue_barrier(queue, wait_for=oev)
        barrier_event.set_callback(cl.command_execution_status.COMPLETE,
                                   notify_callback(self, obj, q_id, 'READ', oev, host_event_info=kwargs['host_event'],
                                                   callback=callback))
        return barrier_event

    def dispatch(self, gpu, cpu, ctxs, cmd_qs, dep=None, partition=None, callback=blank_fn):
        """
        Dispatches Kernel with given partition class value (0,1,2,...,10). 0 is for complete CPU and 10 is for complete GPU.

        :param gpu: Denotes the index of gpu device in cmd_qs['gpu'] list or is -1 if we don't want to use device of this type.
        :type gpu: Integer
        :param cpu: Denotes the index of cpu device in cmd_qs['cpu'] list or is -1 if we don't want to use device of this type.
        :type cpu: Integer
        :param ctxs: Dictionary of Contexts for CPU and GPU devices.
        :type ctxs: dict
        :param cmd_qs: Dictionary of list of Command Queues
        :type cmd_qs: dict
        :param dep: PyOpenCL Event on which subsequent write operations will be dependent on stored in a list
        :param partition: Integer from 0 to 10 or None denoting partition class value.
        :type partition: Integer
        :param callback: A function that will run on the host side once the kernel completes execution on the device. Handle unexpected arguments.
        :return: Tuple with first element being the starting time (host side) of dispatch and second element being list of kernel events for both CPU and GPU devices
        :rtype: Tuple

        """

        dispatch_start = datetime.datetime.now()
        logging.debug("DISPATCH : Dispatch function call for %s starts at %s", self.name, dispatch_start)

        gpu_host_events = HostEvents(self.name, self.id, dispatch_start=dispatch_start)
        cpu_host_events = HostEvents(self.name, self.id, dispatch_start=dispatch_start)

        if partition is not None:
            self.partition = partition
        if dep:
            deps = dep
        else:
            deps = {key: cl.UserEvent(ctxs[key]) for key in ['cpu', 'gpu']}
        if gpu != -1 and cpu != -1:
            size_percent = self.partition * 10
        elif gpu == -1 and cpu != -1:
            size_percent = 0
            self.partition = 0
        elif cpu == -1 and gpu != -1:
            size_percent = 100
            self.partition = 10
        else:
            return None, None
        gdone, cdone = [], []
        if self.partition not in [0, 10]:
            self.chunks_left = 2
        if gpu != -1 and self.partition != 0:
            dispatch_id = generate_unique_id()

            while test_and_set(0, 1):
                pass
            global nGPU
            nGPU -= 1
            rqlock[0] = 0
            offset_percent = 0
            logging.debug("DISPATCH_gpu : Evaluation of kernel arguments for %s on GPU", self.name)
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent)
            gpu_host_events.create_buf_start = time.time()
            logging.debug("DISPATCH_gpu : Creation of buffers for %s on GPU", self.name)
            self.create_buffers(ctxs['gpu'], 'gpu', size_percent, offset_percent)
            gpu_host_events.create_buf_end = time.time()
            logging.debug("DISPATCH_gpu : Setting kernel arguments for %s on GPU", self.name)
            self.set_kernel_args('gpu')
            logging.debug("DISPATCH_gpu :Calling enqueue_write_buffers for GPU")
            gdone.append(self.enqueue_write_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent,
                                                    deps=[deps['gpu']], did=dispatch_id, host_event=gpu_host_events))
            logging.debug("DISPATCH_gpu : Calling enqueue_nd_range_kernel for GPU")
            gdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, 0, deps=[gdone[-1]],
                                             did=dispatch_id, host_event=gpu_host_events))
            logging.debug("DISPATCH_gpu : Calling enqueue_read_buffers for GPU")
            gdone.append(self.enqueue_read_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent,
                                                   deps=[gdone[-1]], callback=callback, did=dispatch_id,
                                                   host_event=gpu_host_events))

        if cpu != -1 and self.partition != 10:
            while test_and_set(0, 1):
                pass
            global nCPU
            nCPU -= 1
            rqlock[0] = 0
            dispatch_id = generate_unique_id()
            offset_percent = size_percent
            size_percent = 100 - size_percent
            logging.debug("DISPATCH_cpu : Evaluation of kernel arguments for %s on CPU", self.name)
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent)
            logging.debug("DISPATCH_cpu : Calling creating_buffers for %s on CPU", self.name)
            cpu_host_events.create_buf_start = time.time()
            self.create_buffers(ctxs['cpu'], 'cpu', size_percent, offset_percent)
            cpu_host_events.create_buf_end = time.time()
            logging.debug("DISPATCH_cpu : Calling set_kernel_args for %s on CPU", self.name)
            self.set_kernel_args('cpu')
            logging.debug("DISPATCH_cpu : Calling enqueue_write_buffers for %s on CPU", self.name)
            cdone.append(self.enqueue_write_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent,
                                                    deps=[deps['cpu']], did=dispatch_id, host_event=cpu_host_events))
            logging.debug("DISPATCH_cpu : Evaluation of enqueue_nd_range_kernel for %s on CPU", self.name)
            cdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, 0, deps=[cdone[-1]],
                                             did=dispatch_id, host_event=cpu_host_events))
            logging.debug("DISPATCH_cpu : Evaluation of enqueue_read_buffers for %s on CPU", self.name)
            cdone.append(self.enqueue_read_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent,
                                                   deps=[cdone[-1]], callback=callback, did=dispatch_id,
                                                   host_event=cpu_host_events))

        if not dep:
            for key in ['gpu', 'cpu']:
                deps[key].set_status(cl.command_execution_status.COMPLETE)

            start_time = datetime.datetime.now()
            logging.debug("DISPATCH : %s ke.dispatch_end %s ", self.name, start_time)
            logging.debug("DISPATCH : Evaluation of kernel arguments for %s ", self.name)
        logging.debug("DISPATCH : Number of events %d" % (len(gdone + cdone)))
        cmd_qs['gpu'][gpu].flush()
        cmd_qs['cpu'][cpu].flush()

        dispatch_end = datetime.datetime.now()
        logging.debug("DISPATCH : Dispatch function call for %s ends at %s", self.name, dispatch_end)
        return start_time, gdone + cdone

    def dispatch_multiple(self, gpus, cpus, ctxs, cmd_qs, dep=None, partition=None, callback=blank_fn):
        """
        Dispatch Kernel across multiple devices  with given partition class value (0,1,2,...,10). 0 is for complete CPU and 10 is for complete GPU.

        :param gpus: A list of gpu device ids
        :type gpus: list
        :param cpus: A list of cpu device ids
        :type cpus: list
        :param ctxs: Dictionary of Contexts for CPU and GPU devices.
        :type ctxs: dict
        :param cmd_qs: Dictionary of list of Command Queues
        :type cmd_qs: dict
        :param dep: PyOpenCL Event on which subsequent write operations will be dependent on stored in a list
        :param partition: Integer from 0 to 10 or None denoting partition class value.
        :type partition: Integer
        :param callback: A function that will run on the host side once the kernel completes execution on the device. Handle unexpected arguments.
        :return: Tuple with first element being the starting time (host side) of dispatch and second element being list of kernel events for both CPU and GPU devices
        :rtype: Tuple

        """
        dispatch_start = datetime.datetime.now()
        logging.debug("DISPATCH : Dispatch function call for %s starts at %s", self.name, dispatch_start)
        while test_and_set(0, 1):
            pass
        global nGPU
        nGPU -= len(gpus)
        global nCPU
        nCPU -= len(cpus)
        rqlock[0] = 0
        self.multiple_break = True
        self.chunks_cpu += len(gpus)
        self.chunks_gpu += len(cpus)
        self.chunks_left = len(gpus) + len(cpus)
        if partition is not None:
            self.partition = partition
        total = len(cpus) + len(gpus)
        size_percent = 100 / total
        if len(gpus) != 0 and len(cpus) != 0:
            gpu_percent = self.partition * 10
        elif len(gpus) == 0 and len(cpus) != 0:
            gpu_percent = 0
            self.partition = 0
        elif len(cpus) == 0 and len(gpus) != 0:
            gpu_percent = 100
            self.partition = 10
        else:
            return None, None
        if gpu_percent == 0:
            nGPU += len(gpus)
        if gpu_percent == 100:
            nCPU += len(cpus)
        cpu_percent = 100 - gpu_percent
        rqlock[0] = 0
        gdone, cdone = [], []

        if dep:
            deps = dep
        else:
            deps = dict()
            deps['gpu'] = cl.UserEvent(ctxs['gpu'])
            deps['cpu'] = cl.UserEvent(ctxs['cpu'])
        if len(gpus) != 0:
            size_percent = gpu_percent / len(gpus)
        for i in range(len(gpus)):
            dispatch_id = generate_unique_id()
            gpu_host_events = HostEvents(self.name, self.id, dispatch_start=dispatch_start)
            offset_percent = size_percent * i
            exact = 1
            if i == total - 1:
                size_percent = 100 - offset_percent
                exact = 0
            gpu = gpus[i]
            logging.debug("DISPATCH_gpu : Evaluation of kernel arguments for %s on GPU", self.name)
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent, exact=exact)
            gpu_host_events.create_buf_start = time.time()
            logging.debug("DISPATCH_gpu : Calling creating_buffer for %s on GPU", self.name)
            self.create_buffers(ctxs['gpu'], 'gpu', size_percent, offset_percent, exact=exact)
            gpu_host_events.create_buf_end = time.time()
            self.set_kernel_args('gpu')
            logging.debug("DISPATCH_gpu : %s Calling enqueue_write_buffers for GPU", self.name)
            gdone.append(self.enqueue_write_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent,
                                                    deps=[deps['gpu']], exact=exact, did=dispatch_id,
                                                    host_event=gpu_host_events))
            logging.debug("DISPATCH_gpu : %s Calling enqueue_nd_range_kernel for GPU", self.name)
            gdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, 0, deps=[gdone[-1]],
                                             exact=exact,
                                             did=dispatch_id, host_event=gpu_host_events))
            logging.debug("DISPATCH_gpu : %s Calling enqueue_read_buffers for GPU", self.name)
            gdone.append(
                self.enqueue_read_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent,
                                          deps=[gdone[-1]], exact=exact,
                                          callback=callback, did=dispatch_id, host_event=gpu_host_events))

        if len(cpus) != 0:
            size_percent = cpu_percent / len(cpus)
        for i in range(len(cpus)):
            dispatch_id = generate_unique_id()

            cpu_host_events = HostEvents(self.name, self.id, dispatch_start=dispatch_start)
            exact = 1
            offset_percent = size_percent * i + gpu_percent
            if i == total - 1 - len(gpus):
                size_percent = 100 - offset_percent
                exact = 0
            cpu = cpus[i]
            logging.debug("DISPATCH_cpu : Evaluation of kernel arguments for %s on CPU", self.name)
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent, exact=exact)
            cpu_host_events.create_buf_start = time.time()
            logging.debug("DISPATCH_cpu : Calling creating_buffer for %s on CPU", self.name)
            self.create_buffers(ctxs['cpu'], 'cpu', size_percent, offset_percent, exact=exact)
            cpu_host_events.create_buf_end = time.time()
            self.set_kernel_args('cpu')
            logging.debug("DISPATCH_cpu : Calling enqueue_write_buffers for %s on CPU", self.name)
            cdone.append(self.enqueue_write_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent,
                                                    deps=[deps['cpu']], exact=exact, did=dispatch_id,
                                                    host_event=cpu_host_events))
            logging.debug("DISPATCH_cpu : Evaluation of enqueue_nd_range_kernel for %s on CPU", self.name)
            cdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, 0, deps=[cdone[-1]],
                                             exact=exact,
                                             did=dispatch_id, host_event=cpu_host_events))
            logging.debug("DISPATCH_cpu : Evaluation of enqueue_read_buffers for %s on CPU", self.name)
            cdone.append(
                self.enqueue_read_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent,
                                          deps=[cdone[-1]], exact=exact,
                                          callback=callback, did=dispatch_id, host_event=cpu_host_events))

        if not dep:
            for key in ['gpu', 'cpu']:
                deps[key].set_status(cl.command_execution_status.COMPLETE)
            start_time = datetime.datetime.now()

        return start_time, gdone + cdone

    def get_device_requirement(self):

        req = {'gpu': 0, 'cpu': 0, 'all': 0}
        if self.partition > 0:
            req['gpu'] += 1
            req['all'] += 1
        if self.partition < 10:
            req['cpu'] += 1
            req['all'] += 1
        return req


def get_platform(vendor_name):
    """
    Gets platform given a vendor name

    :param vendor_name: Name of OpenCL Vendor
    :type vendor_name: string
    :return: OpenCL platform related with vendor name
    :rtype: PyOpenCL platform object
    """
    platforms = cl.get_platforms()
    if len(platforms):
        for pt in cl.get_platforms():
            if vendor_name in pt.name:
                return pt
        print(vendor_name + " Platform not found.")
    else:
        print("No platform found.")


def get_multiple_devices(platform, dev_type, num_devs):
    """
    Get Multiple Devices given a platform and dev type.

    :param platform: OpenCL Platform pertaining to some vendor
    :type platform: pyopencl.platform object
    :param dev_type: Device Type (CPU or GPU)
    :type dev_type: pyopencl.device_type object
    :param num_devs: Number of Devices
    :type num_devs: Integer
    :return: List of OpenCL devices
    :rtype: list of pyopencl.device objects
    """
    devs = platform.get_devices(device_type=dev_type)

    if num_devs > len(devs):
        print("Requirement: " + str(num_devs) + " greater than availability: " + str(len(devs)))
    else:
        return devs[:num_devs]


def get_single_device(platform, dev_type):
    """
    Get Single Device given a platform and dev type.

    :param platform: OpenCL platform pertaining to some vendor
    :type platform: pyopencl.platform object
    :param dev_type: Device Type (CPU or GPU)
    :type dev_type: pyopencl.device_type object
    :return: List containing one OpenCL device
    :rtype: List containing one pyopencl.device object
    """
    return get_multiple_devices(platform, dev_type, 1)


def get_sub_devices(platform, dev_type, num_devs, total_compute=16):
    """
    Get Sub Devices given a platform and dev type.

    :param platform: OpenCL platform pertaining to some vendor
    :type platform: pyopencl.platform object
    :param dev_type: Device Type (CPU or GPU)
    :type dev_type: pyopencl.device_type object
    :param num_devs: Number of devices
    :type num_devs: Integer
    :param total_compute: Total Number of Compute Units for an OpenCL device
    :type total_compute: Integer
    :return: List of OpenCL subdevices for a particular device
    :rtype: list of pyopencl.device objects
    """
    dev = get_single_device(platform, dev_type)[0]
    return dev.create_sub_devices([cl.device_partition_property.EQUALLY, total_compute / num_devs])


def create_command_queue_for_each(devs, ctx):
    """
    Creates command queue for a specified number of devices belonging to a context provided as argument

    :param devs: List of OpenCL devices
    :type devs: List of pyopencl.device objects
    :param ctx: OpenCL Context
    :rtype ctx: pyopencl.context object
    :return: List of OpenCL Command Queues
    :rtype: list of pyopencl.CommandQueue objects
    """
    cmd_qs = [cl.CommandQueue(ctx, device=dev, properties=cl.command_queue_properties.PROFILING_ENABLE) for dev in devs]
    return cmd_qs


def host_initialize(num_gpus=cons.NUM_GPU_DEVICES, num_cpus=cons.NUM_CPU_DEVICES, local=False,
                    cpu_platform=cons.CPU_PLATFORM[0], gpu_platform=cons.GPU_PLATFORM[0]):
    """
    Set local=True if your device doesn't support GPU. But you still
    want to pretend as if you have one.

    :param num_gpus: Number of GPU Devices
    :type num_gpus: Integer
    :param num_cpus: Number of CPU Devices
    :type num_cpus: Integer
    :param local: Flag for Specifying whether platform supports GPU or not
    :type local: Boolean
    :param cpu_platform: CPU Platform Name
    :type cpu_platform: String
    :param gpu_platform: GPU Platform Name
    :type gpu_platform: String
    :return: Returns a tuple comprising command queue dictionary, context dictionary and list of OpenCL CPU and GPU devices.
    :rtype: Tuple


    """
    global nGPU
    global nCPU
    if local:
        gpus = None
        cpu_platform = get_platform(cpu_platform)
        if num_cpus > 1 and cl.get_cl_header_version() >= (1, 2):
            cpus = get_sub_devices(cpu_platform, cl.device_type.CPU, num_cpus, 4)
        else:
            cpus = get_single_device(cpu_platform, cl.device_type.CPU)
        ctx = cl.Context(devices=cpus)
        ctxs = {"gpu": ctx, "cpu": ctx}
        cmd_q = create_command_queue_for_each(cpus, ctxs['cpu'])
        # cmd_qs = {"gpu":cmd_q, "cpu": cmd_q}
        cmd_qs = {"gpu": [cmd_q[0]], "cpu": [cmd_q[1]]}
        ready_queue['gpu'].append(0)
        ready_queue['cpu'].append(0)
        device_history['gpu'].append([])
        device_history['cpu'].append([])
        gpus = [cpus[0]]
        cpus = [cpus[1]]
        nGPU = 1
        nCPU = 1

    else:
        gpus, cpus = [], []
        ctxs = {"gpu": None, "cpu": None}
        cmd_qs = {
            "gpu": [],
            "cpu": []
        }
        if num_gpus > 0:
            gpu_platform = get_platform(gpu_platform)
            gpus = get_multiple_devices(gpu_platform, cl.device_type.GPU, num_gpus)
            # print gpus
            global MAX_GPU_ALLOC_SIZE
            MAX_GPU_ALLOC_SIZE = gpus[0].max_mem_alloc_size
            ctxs['gpu'] = cl.Context(devices=gpus)
            cmd_qs['gpu'] = create_command_queue_for_each(gpus, ctxs['gpu'])
        if num_cpus > 0:
            # cpu_platform = get_platform("AMD Accelerated Parallel Processing")
            cpu_platform = get_platform(cpu_platform)
            if num_cpus > 1 and cl.get_cl_header_version() >= (1, 2):
                cpus = get_sub_devices(cpu_platform, cl.device_type.CPU, num_cpus)
            else:
                cpus = get_single_device(cpu_platform, cl.device_type.CPU)
            # print cpus
            global MAX_CPU_ALLOC_SIZE
            MAX_CPU_ALLOC_SIZE = cpus[0].max_mem_alloc_size
            ctxs['cpu'] = cl.Context(devices=cpus)
            cmd_qs['cpu'] = create_command_queue_for_each(cpus, ctxs['cpu'])
        nGPU = len(cmd_qs['gpu'])
        nCPU = len(cmd_qs['cpu'])
        for key in cmd_qs.keys():
            ready_queue[key].extend(range(len(cmd_qs[key])))
            device_history[key].extend([[] for i in range(len(cmd_qs[key]))])
    global system_cpus
    global system_gpus
    system_cpus = nCPU
    system_gpus = nGPU
    return cmd_qs, ctxs, gpus, cpus


def host_synchronize(cmd_qs, events):
    """
    Ensures that all operations in all command queues and associated events have finished execution

    :param cmd_qs: Dictionary of list of Command Queues
    :type cmd_qs: dict
    :param events: List of OpenCL Events
    :type events: list of pyopencl.events
    """
    global nCPU, nGPU, system_cpus, system_gpus
    while nCPU < system_cpus or nGPU < system_gpus:
        pass
    global callback_queue
    while any(callback_queue[key] == False for key in callback_queue.keys()):
        pass
    for event in events:
        event.wait()
    for key in cmd_qs:
        for q in cmd_qs[key]:
            q.finish()


def build_kernel_from_info(info_file_name, gpus, cpus, ctxs):
    """
    Create Kernel object from info.

    :param info_file_name: Name of OpenCL Kernel Specification File (JSON)
    :type info_file_name: String
    :param gpus: List of OpenCL GPU devices
    :type gpus: list of pyopencl.device objects
    :param cpus: List of OpenCL CPU devices
    :type cpus: list of pyopencl.device objects
    :param ctxs: Dictionary of contexts keyed by device types
    :type ctxs: dict
    :return: Dictionary of key: value pairs where key is device type and value is the device specific binary compiled for the kernel
    :rtype: dict


    """
    import json
    info = json.loads(open(info_file_name).read())
    ki = Kernel(info)
    ki.build_kernel(gpus, cpus, ctxs)
    return ki


ALL_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)',
              'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
              'rgb(217,217,217)', 'rgb(240,2,127)', 'rgb(253,205,172)', 'rgb(179,205,227)', 'rgb(166,86,40)',
              'rgb(51,160,44)', 'rgb(247,129,191)', 'rgb(253,191,111)', 'rgb(190,186,218)', 'rgb(231,41,138)',
              'rgb(166,216,84)', 'rgb(153,153,153)', 'rgb(166,118,29)', 'rgb(230,245,201)', 'rgb(255,255,204)',
              'rgb(102,102,102)', 'rgb(77,175,74)', 'rgb(228,26,28)', 'rgb(217,95,2)', 'rgb(255,255,179)',
              'rgb(178,223,138)', 'rgb(190,174,212)', 'rgb(253,180,98)', 'rgb(255,217,47)', 'rgb(31,120,180)',
              'rgb(56,108,176)', 'rgb(229,216,189)', 'rgb(251,154,153)', 'rgb(222,203,228)', 'rgb(203,213,232)',
              'rgb(188,128,189)', 'rgb(55,126,184)', 'rgb(231,138,195)', 'rgb(244,202,228)', 'rgb(191,91,23)',
              'rgb(128,177,211)', 'rgb(27,158,119)', 'rgb(229,196,148)', 'rgb(253,218,236)', 'rgb(102,166,30)',
              'rgb(241,226,204)', 'rgb(255,127,0)', 'rgb(252,141,98)', 'rgb(227,26,28)', 'rgb(254,217,166)',
              'rgb(141,160,203)', 'rgb(204,235,197)', 'rgb(117,112,179)', 'rgb(152,78,163)', 'rgb(202,178,214)',
              'rgb(141,211,199)', 'rgb(106,61,154)', 'rgb(253,192,134)', 'rgb(255,255,51)', 'rgb(179,226,205)',
              'rgb(127,201,127)', 'rgb(251,128,114)', 'rgb(255,242,174)', 'rgb(230,171,2)', 'rgb(102,194,165)',
              'rgb(255,255,153)', 'rgb(179,179,179)', 'rgb(179,222,105)', 'rgb(252,205,229)', 'rgb(204,204,204)',
              'rgb(242,242,242)', 'rgb(166,206,227)', 'rgb(251,180,174)']

AC = ALL_COLORS


def plot_gantt_chart_graph(device_history, filename):
    """
    Plots Gantt Chart and Saves as png.

    :param device_history: Dictionary Structure containing timestamps of every kernel on every device
    :type device_history: dict
    :param filename: Name of file where the gantt chart is saved. The plot is saved in gantt_charts folder.
    :type filename: String
    """
    import random
    import colorsys
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def save_png(fig, filename):
        fig.savefig(filename)
        print "GANTT chart is saved at %s" % filename

    def get_N_HexCol(N=5):

        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in xrange(N)]
        hex_out = []
        for rgb in HSV_tuples:
            rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
            hex_out.append("".join(map(lambda x: chr(x).encode('hex'), rgb)))
        return hex_out

    def get_N_random_HexColor(N=5):
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in xrange(N)]
        hex_out = []
        'rgb(31, 119, 180)'
        indexs = random.sample(range(0, 77), N)
        for i in indexs:
            r = int(ALL_COLORS[i][4:-1].split(",")[0])
            g = int(ALL_COLORS[i][4:-1].split(",")[1])
            b = int(ALL_COLORS[i][4:-1].split(",")[2])
            hex_out.append('#%02x%02x%02x' % (r, g, b))
        return hex_out

    def list_from_file(file):
        device_info_list = []
        dev_data = open(file, "r")
        for line in dev_data:
            if "HOST_EVENT" in line:
                d_list = line.split(" ")[1:]
                device_info_list.append(d_list)
        return device_info_list

    def list_from_dev_history(dev_history):
        device_info_list = []
        for his in dev_history:
            device_info_list.append(his.split(" ")[1:])
        return device_info_list

    def get_min(device_info_list):
        g_min = Decimal('Infinity')
        for item in device_info_list:
            n = Decimal(min(item[3:], key=lambda x: Decimal(x)))
            if g_min > n:
                g_min = n
        return g_min

    def get_max(device_info_list):
        g_max = -1
        for item in device_info_list:
            x = Decimal(max(item[3:], key=lambda x: Decimal(x)))
            if g_max < x:
                g_max = x
        return g_max

    def normalise_timestamp(device_info_list):
        min_t = get_min(device_info_list)
        for item in device_info_list:
            for i in range(len(item) - 3):
                item[i + 3] = Decimal(item[i + 3]) - min_t
        return device_info_list

    device_info_list = normalise_timestamp(list_from_dev_history(device_history))

    colourMap = {}
    # colors = get_N_HexCol(len(device_info_list))
    colors = get_N_random_HexColor(len(device_info_list))

    c = 0
    dev_time = {}
    for k in device_info_list:
        kn = k[0] + "_" + k[1]

        kernel_times = [k[2], k[3], k[-1]]
        if kn not in dev_time:
            dev_time[kn] = []
        if kn in dev_time:
            dev_time[kn].append(kernel_times)

    for k in device_info_list:
        colourMap[k[2]] = colors[c]
        c = c + 1

    # legend_patches = []
    # for kn in colourMap:
    #     patch_color = "#" + colourMap[kn]
    #     legend_patches.append(patches.Patch(color=patch_color, label=str(k[2])))

    fig, ax = plt.subplots(figsize=(20, 10))
    device = 0
    #print dev_time
    for dev in dev_time:
        for k in dev_time[dev]:
            kname = k[0]
            # patch_color = "#" + colourMap[kname]
            patch_color = colourMap[kname]
            start = k[1]
            finish = k[2]
            y = 5 + device * 5
            x = start
            height = 5
            width = finish - start
            # print kname.split(",")[-1] + " : " + str(x) + "," + str(y) + "," + str(width) + "," + str(height)
            ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=patch_color, edgecolor="#000000",
                                           label=kname.split(",")[-1]))
        device = device + 1
    plt.legend(loc=1)
    ax.autoscale(True)
    x_length = float(get_max(device_info_list))
    ax.set_xlim(0, 1.2 * x_length)
    ax.set_ylim(0, len(dev_time) * 10, True, True)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = ""
    i = 1
    for dev in dev_time:
        labels[i] = (dev)
        i = i + 1

    y_ticks = np.arange(2.5, 2.5 + 5 * (1 + len(dev_time)), 5)

    plt.yticks(y_ticks.tolist(), labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('time ( in second )')
    ax.set_ylabel('devices')
    ax.set_yticklabels(labels)

    save_png(fig, filename)
