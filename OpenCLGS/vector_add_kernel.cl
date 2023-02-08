__kernel void vector_add(__global int *A, __global int *B, __global int *C) {
    
    // Get the index of the current element
    int i = get_global_id(0);

    int c = 0;

    // Do the operation
    for(int j = 0; j < 100000; ++j)
        c += A[i] + B[i];

    C[i] = c;
}


__kernel void vector_add_double(__global double *A, __global double *B, __global double *C) {
    
    // Get the index of the current element
    int i = get_global_id(0);

    double c = 0.0;

    // Do the operation
    for(int j = 0; j < 10000; ++j)
        c += A[i] + B[i];

    C[i] = c;
}


__kernel void vector_add_float(__global float *A, __global float *B, __global float *C) {
    
    // Get the index of the current element
    int i = get_global_id(0);

    float c = 0.0;

    // Do the operation
    for(int j = 0; j < 100000; ++j)
        c += A[i] + B[i];

    C[i] = c;
}
