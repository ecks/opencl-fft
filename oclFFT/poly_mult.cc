#include "FFT.h"
#include <iostream>
#include <vector>

#include <oclUtils.h>
#include <shrQATest.h>

#define PI 3.14159265
#define EPSILON 0.000001

using namespace std;

vector<double> multiply_polys(vector<double> poly_a, vector<double> poly_b);

const char* cSourceFile = "FFT.cl";

void * cl_poly_a, cl_poly_b;

cl_context cxGPUContext;
cl_command_queue cqCommandQueue;
cl_platform_id cpPlatform;
cl_device_id cdDevice;
cl_program cpProgram;
cl_kernel ckKernel;
cl_mem cmDevPolyMultA;
size_t szGlobalWorkSize;
cl_int ci_Err1;
char* cPathAndName = NULL;
char* cSourceCL = NULL;

int main(int argc, const char **argv)
{
    // 7x^2 + 3x + 9
    vector<double> poly_a;
    poly_a.push_back(9);
    poly_a.push_back(3);
    poly_a.push_back(7);
    // -13x + 5
    vector<double> poly_b;
    poly_b.push_back(5);
    poly_b.push_back(-13);
    // -91x^3 - 4x^2 - 102x + 45
    vector<double> result = multiply_polys(poly_a, poly_b, argc, argv);
    bool success = abs(result[0] - 45) < EPSILON
            && abs(result[1] + 102) < EPSILON
            && abs(result[2] + 4) < EPSILON
            && abs(result[3] + 91) < EPSILON;
    cout << "Multiplying polynomials: " << (success ? "OK" : "FAILED") << endl;
}

vector<double> multiply_polys(vector<double> poly_a, vector<double> poly_b, int argc, const char **argv)
{
    // 1. Make place for resulting polynomial and ensure n is a power of two.
    int n = poly_a.size() + poly_b.size();
    int power_of_2 = 2;
    while (power_of_2 < n)
        power_of_2 <<= 1;
    n = power_of_2;
    opencl_init(n, argc, argv);
    poly_a.resize(n, 0);
    poly_b.resize(n, 0);
    // 2. Compute point-value representation of a and b for values of unity
    // roots using DFT.
    vector<FFT::Complex> poly_a_complex(n);
    vector<FFT::Complex> poly_b_complex(n);
    for (int i = 0; i < n; ++i)
    {
        poly_a_complex[i] = poly_a[i];
        poly_b_complex[i] = poly_b[i];
    }
    FFT dft(n);
    vector<FFT::Complex> poly_a_values = dft.transform(poly_a_complex);
    vector<FFT::Complex> poly_b_values = dft.transform(poly_b_complex);
    for (int i = 0; i < n; ++i)
    {
        poly_a_values[i] *= n;
        poly_b_values[i] *= n;
    }
    // 3. Multiply poly a values by poly b values.
    vector<FFT::Complex> poly_c_values(n);
    for (int i = 0; i < n; ++i)
        poly_c_values[i] = poly_a_values[i] * poly_b_values[i];
    // 4. Compute coefficients representation of c using Inverse DFT.
    FFT idft(n, true);
    vector<FFT::Complex> poly_c_complex = idft.transform(poly_c_values);
    vector<double> poly_c(n);
    for (int i = 0; i < n; ++i)
        poly_c[i] = poly_c_complex[i].real() / n;
    return poly_c;
}

void opencl_init(int n, int argc, const char **argv)
{
    shrQAStart(argc, (char **)argv);
    // set logfile name and start logs
    shrSetLogFileName("oclFFT.txt");
    shrLog("%s Starting...\n\n# of elements per Array \t= %i\n", argv[0], n);
    
    szGlobalWorkSize = n/2;
    shrLog("Initializing data...\n");
    void * cl_poly_a = (void *)malloc(sizeof(float2) * szGlobalWorkSize);
    void * cl_poly_b = (void *)malloc(sizeof(float2) * szGlobalWorkSize);

    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);
    
    shrLog("clGetPlatformID...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, argv, EXIT_FAILURE);
    }   
    
    //Get the devices
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    shrLog("clGetDeviceIDs...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, argv, EXIT_FAILURE);
    }   

    //Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    shrLog("clCreateContext...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, argv, EXIT_FAILURE);
    }   

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
    shrLog("clCreateCommandQueue...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, argv, EXIT_FAILURE);
    }   

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cmDevPolyMultA = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float2) * szGlobalWorkSize, NULL, &ciErr1);

    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Read the OpenCL kernel in from source file
    shrLog("oclLoadProgSource (%s)...\n", cSourceFile);
    cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);
    
    // Create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErr1);
    shrLog("clCreateProgramWithSource...\n");
    if (ciErr1 != CL_SUCCESS)
    {
      shrLog("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, argv, EXIT_FAILURE);
    }
  
    // Build the program with 'mad' Optimization option
    #ifdef MAC
      char* flags = "-cl-fast-relaxed-math -DMAC";
    #else
      char* flags = "-cl-fast-relaxed-math";
    #endif
    ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    shrLog("clBuildProgram...\n");
    if (ciErr1 != CL_SUCCESS)
    {
      shrLog("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, argv, EXIT_FAILURE);
    }
    // Create the kernel
    ckKernel = clCreateKernel(cpProgram, "FFT", &ciErr1);
    shrLog("clCreateKernel (VectorAdd)...\n");
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, argv, EXIT_FAILURE);
    }
    
    // Set the Argument values
    ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevPolyMultA);
    shrLog("clSetKernelArg 0...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, argv, EXIT_FAILURE);
    }
}
