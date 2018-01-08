# pyschedcl
A Python Based OpenCL Scheduling Framework

Overview
============

PySchedCL is a python based scheduling framework for OpenCL applications. The framework heavily relies on the PyOpenCL package which provides easy access to the full power of the OpenCL API to the end user. We present only the dependencies of the package and a one line description for every important folder and file present in the project. Detailed usage and documentation regarding the project may be found here.

Dependencies
------------------

+ Python 2.7
+ PIP
+ OpenCL Runtime Device Drivers (Minimum 1.2)
  - Intel
  - AMD
  - NVIDIA
+ PyOpenCL



Project Hierarchy
-----------------

<pre>
<code>
.
├── <b>setup.py</b> (Downloads python packages required by the base package)
├── <b>configure.py</b> (Configures certain parameters required by the framework)
├── <b>docs</b> (HTML sources for basic API Documentation)
├── <b>examples</b> (Example Scripts)
└── <b>pyschedcl</b> (Base Package Folder)
    ├── <b>pyschedcl.py</b> (Base Package API)
    ├── <b>partition</b> (Folder containing scripts for partitioning)
    ├── <b>scheduling</b> (Folder containing scripts for scheduling)
    ├── <b>utils</b> (Folder containing additional utility scripts)
    ├── <b>kernel_src</b> (Folder containing kernel source files used by framework)
    ├── <b>info</b> (Folder containing sample JSON Kernel Specification Files)
    ├── <b>logs</b> (Folder containing log files generated as a result of running partitioning and scheduling scripts)
    ├── <b>output</b> (Folder containing data dump outputs after execution of individual OpenCL kernels )
    ├── <b>gantt_charts</b> (Folder containing gantt charts generated as a results of running partitioning and scheduling scripts)  
  </code>
  </pre>
    
