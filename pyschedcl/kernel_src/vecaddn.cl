__kernel
void vecadd(__global int *A,
            __global int *B,
            __global int *C,
            int iNumElements)
{

   // Get the work-item’s unique ID
   int idx = get_global_id(0);
   if (idx >= iNumElements)
    {   
        return; 
    }
   // Add the corresponding locations of
   // 'A' and 'B', and store the result in 'C'.
   C[idx] = A[idx] + B[idx];
}