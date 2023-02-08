#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

#define CHECK_CL_ERR(__code) __code; if (ret != CL_SUCCESS) printf("CL error: '%d' on line %u\n", ret, __LINE__);

#define TYPE float
#define KERNEL "vector_add_float"
#define DEVICE 0

int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 10240000;
    TYPE *A = (TYPE*)malloc(sizeof(TYPE)*LIST_SIZE);
    TYPE *B = (TYPE*)malloc(sizeof(TYPE)*LIST_SIZE);
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("vector_add_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
    #define  PARAM_SIZE 128
    cl_platform_id * platform_ids = NULL;
    cl_device_id** device_ids = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    platform_ids = (cl_platform_id*)malloc(sizeof(cl_platform_id) * ret_num_platforms);
    device_ids = (cl_device_id**)malloc(sizeof(cl_device_id*) * ret_num_platforms);

    ret = clGetPlatformIDs(ret_num_platforms, platform_ids, NULL);
char vendor_name[PARAM_SIZE] = {0};
char platform_name[PARAM_SIZE] = {0};
    for (cl_uint ui=0; ui< ret_num_platforms; ++ui)
    {
        ret = clGetPlatformInfo(platform_ids[ui],
                              CL_PLATFORM_VENDOR, 
                              PARAM_SIZE * sizeof(char), 
                              vendor_name, 
                              NULL);
        ret = clGetPlatformInfo(platform_ids[ui],
                              CL_PLATFORM_NAME, 
                              PARAM_SIZE * sizeof(char), 
                              platform_name, 
                              NULL);
        if (CL_SUCCESS != ret) 
        {
                        // handle error
        }
        if (vendor_name != NULL)
        {
            printf("Platform %u: %s by %s\n", ui, platform_name, vendor_name);
        }

        ret = clGetDeviceIDs( platform_ids[0], CL_DEVICE_TYPE_ALL, 1, 
            NULL, &ret_num_devices);

        printf("    %u devices:\n", ret_num_devices);
        device_ids[ui] = (cl_device_id*)malloc(sizeof(cl_device_id) * ret_num_devices);

        ret = clGetDeviceIDs( platform_ids[0], CL_DEVICE_TYPE_ALL, ret_num_devices, 
            device_ids[ui], NULL);

        for (cl_uint dev = 0; dev < ret_num_devices; ++dev)
        {
            char device_info[PARAM_SIZE] = {0};
            size_t device_info2 = {0};
            size_t global_mem_info = {0};
            CHECK_CL_ERR(ret = clGetDeviceInfo(device_ids[ui][dev], CL_DEVICE_NAME, PARAM_SIZE * sizeof(char), device_info, NULL));
            CHECK_CL_ERR(ret = clGetDeviceInfo(device_ids[ui][dev], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &device_info2, &global_mem_info));
        //     CHECK_CL_ERR(ret = clGetDeviceInfo(device_ids[ui][dev], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_uint), &global_mem_info, global_mem_info));
            if (ret == CL_SUCCESS)
            {
                printf("        Device %u name: %32s having %6lu MiB global mem\n", dev, device_info, device_info2 / 1024 / 1024);
            }
        }
    }

    cl_device_id device_id = device_ids[0][DEVICE];

    // Create an OpenCL context
    CHECK_CL_ERR(cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret));
printf("Created context\n");
    // Create a command queue
    const cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    CHECK_CL_ERR(cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, properties, &ret));

printf("Created queue\n");
    // Create memory buffers on the device for each vector 
    CHECK_CL_ERR(cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            LIST_SIZE * sizeof(TYPE), NULL, &ret));
    CHECK_CL_ERR(cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(TYPE), NULL, &ret));
    CHECK_CL_ERR(cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            LIST_SIZE * sizeof(TYPE), NULL, &ret));

printf("Created buffers\n");
    // Copy the lists A and B to their respective memory buffers
    CHECK_CL_ERR(ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            LIST_SIZE * sizeof(TYPE), A, 0, NULL, NULL));
    CHECK_CL_ERR(ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
            LIST_SIZE * sizeof(TYPE), B, 0, NULL, NULL));

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    CHECK_CL_ERR(ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

    // Create the OpenCL kernel
    printf("Creating kernel from %s\n", KERNEL);
    CHECK_CL_ERR(cl_kernel kernel = clCreateKernel(program, KERNEL, &ret));

    // Set the arguments of the kernel
    CHECK_CL_ERR(ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj));
    CHECK_CL_ERR(ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj));
    CHECK_CL_ERR(ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj));
    
    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Process in groups of 64
    cl_event event;
    CHECK_CL_ERR(ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, &event));

    clWaitForEvents(1, &event);
    clFinish(command_queue);
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double nanoSeconds = time_end - time_start;
    printf("OpenCl device %u of %u platforms. Execution time is: %0.3f milliseconds (%f ns) \n", ret_num_devices, ret_num_platforms, nanoSeconds / 1000000.0, nanoSeconds);

    // Read the memory buffer C on the device to the local variable C
    TYPE *C = (TYPE*)malloc(sizeof(TYPE)*LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
            LIST_SIZE * sizeof(TYPE), C, 0, NULL, NULL);

    // Display the result to the screen
//     for(i = 0; i < LIST_SIZE; i++)
//         printf("%d + %d = %d\n", A[i], B[i], C[i]);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    return 0;
}

