#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdarg>
#include <map>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#define N 16
#define BLOCK_SIZE 8
#define RADIUS 3

class Queue
{
private:
    cl::Platform _platform;
    cl::Device _device;
    cl::Context _context;
    cl::CommandQueue _queue;
    std::vector<cl::Buffer> _buffers;
    cl::Kernel _kernel;
    cl::Program _program;
    void setKernelArgs(int idx) {} // Base case for recursion

    template <typename Last>
    void setKernelArgs(int idx, Last last)
    {
        _kernel.setArg(idx, last);
    };

    template <typename First, typename... Rest>
    void setKernelArgs(int idx, First first, Rest... rest)
    {
        _kernel.setArg(idx, first);
        setKernelArgs(idx + 1, rest...);
    };

public:
    Queue()
    {
        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        _platform = platforms.front();
        std::cout << "Platform: " << _platform.getInfo<CL_PLATFORM_NAME>()
                  << std::endl;

        // Get a list of devices on this platform
        std::vector<cl::Device> devices;
        // Select the platform.
        _platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        _device = devices.front();
        std::cout << "Device: " << _device.getInfo<CL_DEVICE_NAME>()
                  << std::endl
                  << std::endl;

        // Create a context
        _context = cl::Context(devices);

        // Create a command queue
        // Select the device.
        _queue = cl::CommandQueue(_context, _device);
    }

    template <typename T>
    int addBuffer(std::vector<T> &data, cl_mem_flags flags = CL_MEM_READ_WRITE)
    {
        cl::Buffer buffer(_context, flags, data.size() * sizeof(T));
        _queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, data.size() * sizeof(T),
                                  data.data());
        _buffers.push_back(buffer);
        return _buffers.size() - 1;
    }

    void setKernel(const std::string &file, const std::string &kernelName)
    {
        std::ifstream sourceFile(file);
        std::stringstream sourceCode;
        sourceCode << sourceFile.rdbuf();

        // Make and build program from the source code
        _program = cl::Program(_context, sourceCode.str(), true);

        // Make kernel
        _kernel = cl::Kernel(_program, kernelName.c_str());
    }

    template <typename T>
    void readBuffer(std::vector<T> &data, int index = 0)
    {
        _queue.enqueueReadBuffer(_buffers[index], CL_TRUE, 0,
                                 data.size() * sizeof(T), data.data());
    }

    template <typename... Args>
    cl::Event operator()(cl::NDRange globalSize, cl::NDRange localSize, Args... args)
    {
        // Set the kernel arguments
        for (size_t i = 0; i < _buffers.size(); ++i)
        {
            _kernel.setArg(i, _buffers[i]);
        }
        setKernelArgs(_buffers.size(), args...);

        cl::Event event;
        _queue.enqueueNDRangeKernel(_kernel, cl::NullRange, globalSize, localSize,
                                    nullptr, &event);
        event.wait();
        return event;
    }
};

using std::chrono::microseconds;

int main(int argc, char const *argv[])
{
    try
    {
        Queue q;
        const int size = N * sizeof(int);
        std::vector<float> a(N), b(N), c(N);

        // Assign values to host variables
        auto t_start = std::chrono::high_resolution_clock::now();
        for (auto &v : a)
            v = 1;
        for (auto &v : b)
            v = 2;
        auto t_end = std::chrono::high_resolution_clock::now();
        auto t_create_data =
            std::chrono::duration_cast<microseconds>(t_end - t_start).count();

        // Copy values from host variables to device
        t_start = std::chrono::high_resolution_clock::now();
        q.addBuffer(a, CL_MEM_READ_ONLY);
        q.addBuffer(b, CL_MEM_READ_ONLY);
        q.addBuffer(c, CL_MEM_WRITE_ONLY);
        t_end = std::chrono::high_resolution_clock::now();
        auto t_copy_to_device =
            std::chrono::duration_cast<microseconds>(t_end - t_start).count();

        // Read the program source
        q.setKernel("VectorAdd.cl", "vectorAdd");

        // Execute the function on the device (using 32 threads here)
        cl::NDRange globalSize(N);
        cl::NDRange localSize(BLOCK_SIZE);

        t_start = std::chrono::high_resolution_clock::now();
        cl::Event event = q(globalSize, localSize, 2.0f, N);
        event.wait();
        t_end = std::chrono::high_resolution_clock::now();
        auto t_kernel =
            std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
                .count();

        // Copy the output variable from device to host
        t_start = std::chrono::high_resolution_clock::now();
        q.readBuffer(c, 2);
        t_end = std::chrono::high_resolution_clock::now();
        auto t_copy_to_host =
            std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
                .count();

        // Print the result
        std::cout << "RESULTS: " << std::endl;
        for (int i = 0; i < N; i++)
            std::cout << "  out[" << i << "]: " << c[i] << "\n";

        std::cout << "Time to create data: " << t_create_data << " microseconds\n";
        std::cout << "Time to copy data to device: " << t_copy_to_device
                  << " microseconds\n";
        std::cout << "Time to execute kernel: " << t_kernel << " microseconds\n";
        std::cout << "Time to copy data to host: " << t_copy_to_host
                  << " microseconds\n";
        std::cout << "Time to execute the whole program: "
                  << t_create_data + t_copy_to_device + t_kernel + t_copy_to_host
                  << " microseconds\n";
    }
    catch (cl::Error err)
    {
        std::cerr << "Error (" << err.err() << "): " << err.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
    return 0;
}