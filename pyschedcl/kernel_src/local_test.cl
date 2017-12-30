// Matrix multiplication kernel called by MatrixMul()
// The following defines are set during runtime compilation, see reduction.cpp
#define BLOCK_DIM 16
//#define blockSize 16

__kernel void local_kernel(__global int* a,
                           __global int* b,
                           __global int* c,
                           __global int* d,
                           __global int* e
                           )
{

        int i = get_global_id(0);
        int j = get_global_id(1);
        a[i] = get_group_id(0);
        b[i] = get_num_groups(0);
        c[i] = get_local_id(0);
        d[j] = get_local_id(1);
        e[i*1024 + j] = get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0);
}