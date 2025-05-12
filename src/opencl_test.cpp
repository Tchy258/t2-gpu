#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <iostream>

int main() {
    cl_uint platformCount;
    cl_platform_id platform;
    cl_int err = clGetPlatformIDs(1, &platform, &platformCount);
    if (err != CL_SUCCESS || platformCount == 0) {
        std::cerr << "No OpenCL platform found.\n";
        return 1;
    }

    cl_uint deviceCount;
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &deviceCount);
    if (err != CL_SUCCESS || deviceCount == 0) {
        std::cerr << "No OpenCL devices found.\n";
        return 1;
    }

    char deviceName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "OpenCL Device: " << deviceName << "\n";

    return 0;
}
