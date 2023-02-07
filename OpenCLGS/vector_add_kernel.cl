__kernel void vector_add(__global int *A, __global int *B, __global int *C) {
    
    // Get the index of the current element
    int i = get_global_id(0);

    // Do the operation
    for(int j = 0; j < 1000; ++j)
    C[i] += A[i] + B[i];
}
