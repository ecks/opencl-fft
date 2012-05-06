#include <oclUtils.h>
#include <shrQATest.h>

#include "oclFFT.h"
#include "FFT.h"
#include <iostream>
#include <vector>

#define PI 3.14159265
#define EPSILON 0.000001
#define EPSILON2 0.001

using namespace std;

vector<double> multiply_polys(vector<double> poly_a, vector<double> poly_b, int argc, const char **argv);
void opencl_init(int n, int argc, const char **argv);
void compareValues(vector<FFT::Complex> cpu_transform_values, void * gpu_transform_values, int n);

const char* cSourceFile = "FFT2.cl";

void * cl_poly_a, * cl_poly_b, * cl_poly_c, * cl_debug;
cl_ulong local_memory_size;
unsigned int points_per_group;
unsigned int points_per_item;

cl_context cxGPUContext;
cl_command_queue cqCommandQueue;
cl_platform_id cpPlatform;
cl_device_id cdDevice;
cl_program cpProgram;
cl_kernel ckKernel;
cl_mem cmDevPolyMultA;
cl_mem cmDevPolyMultB;
cl_mem cmDevPolyMultC;
cl_mem cmPointsPerGroup;
cl_mem cmDevDebug;
cl_mem cmDir;
size_t num_points;
size_t items_per_group;
size_t szGlobalWorkSize;
size_t szLocalWorkSize;
size_t szKernelLength;
size_t log_size;
cl_int ciErr1;
char* cPathAndName = NULL;
char* cSourceCL = NULL;
char* program_log;

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
    opencl_init(n, argc, argv); // init GPU stuff
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

    // Set the Argument values
    ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevPolyMultA);
    ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_float2) * n, NULL);
    ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmPointsPerGroup);
    ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&cmDevDebug);
    ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void*)&cmDir);
    shrLog("clSetKernelArg 0...3\n\n");
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }

    dft.transformGPU(poly_a_complex, cl_poly_a, cl_debug, cmDevPolyMultA, 
                     cmPointsPerGroup, cmDevDebug, cmDir, ckKernel, szGlobalWorkSize, szLocalWorkSize, points_per_group,
                     cqCommandQueue, ciErr1, argc, (const char **)argv);
    compareValues(poly_a_values, cl_poly_a, n);

    vector<FFT::Complex> poly_b_values = dft.transform(poly_b_complex);
    // Set the Argument values
    ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevPolyMultB);
    ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_float2) * n, NULL);
    ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmPointsPerGroup);
    ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&cmDevDebug);
    ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void*)&cmDir);
    shrLog("clSetKernelArg 0...1\n\n");
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }

    dft.transformGPU(poly_b_complex, cl_poly_b, cl_debug, cmDevPolyMultB, 
                     cmPointsPerGroup, cmDevDebug, cmDir, ckKernel, szGlobalWorkSize, szLocalWorkSize, points_per_group,
                     cqCommandQueue, ciErr1, argc, (const char **)argv);
    compareValues(poly_b_values, cl_poly_b, n);

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

    // Set the Argument values
    ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevPolyMultC);
    ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_float2) * n, NULL);
    ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmPointsPerGroup);
    ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&cmDevDebug);
    ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void*)&cmDir);
    shrLog("clSetKernelArg 0...2\n\n");
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }

    idft.transformGPU(poly_c_values, cl_poly_c, cl_debug, cmDevPolyMultC, 
                      cmPointsPerGroup, cmDevDebug, cmDir, ckKernel, szGlobalWorkSize, szLocalWorkSize, points_per_group,
                      cqCommandQueue, ciErr1, argc, (const char **)argv);
    compareValues(poly_c_complex, cl_poly_c, n);

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
    num_points = n;
    shrLog("%s Starting...\n\n# of elements per Array \t= %i\n", argv[0], n);
    
    shrLog("Initializing data...\n");
    cl_poly_a = (void *)malloc(sizeof(cl_float2) * n);
    cl_poly_b = (void *)malloc(sizeof(cl_float2) * n);
    cl_poly_c = (void *)malloc(sizeof(cl_float2) * n);

    cl_debug = (void *)malloc(sizeof(cl_float2) * n);

    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);
    
    shrLog("clGetPlatformID...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }   

    //Get the devices
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    shrLog("clGetDeviceIDs...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }   

    //Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    shrLog("clCreateContext...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }   

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
    shrLog("clCreateCommandQueue...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }   

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cmDevPolyMultA = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * n, NULL, &ciErr1);

    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 

    cmDevPolyMultB = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * n, NULL, &ciErr1);

    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 

    cmDevPolyMultC = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * n, NULL, &ciErr1);

    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 

    cmPointsPerGroup = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &ciErr1);

    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 


    cmDevDebug = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * n, NULL, &ciErr1);

    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 

    cmDir = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &ciErr1);

    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
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
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
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
      clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 
      	0, NULL, &log_size);
      program_log = (char *)malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG,
      		log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      shrLog("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }
    // Create the kernel
    ckKernel = clCreateKernel(cpProgram, "FFT2", &ciErr1);
    shrLog("clCreateKernel FFT2...\n");
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }

    ciErr1 = clGetKernelWorkGroupInfo(ckKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *)&items_per_group, NULL);
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), (void *)&local_memory_size, NULL);
    if(ciErr1 < 0)
    {
      shrLog("Couldn't determine the local memory size");
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }

    points_per_group = local_memory_size/(2*sizeof(float));
    points_per_item = points_per_group/items_per_group;

    szLocalWorkSize = num_points/points_per_item;
    szGlobalWorkSize = num_points/points_per_item;
    points_per_group = num_points;
}

void compareValues(vector<FFT::Complex> cpu_transform_values, void * gpu_transform_values, int n)
{
  cl_float2 * gpu_transform_values_fl = (cl_float2 *)gpu_transform_values;
  int OK = 1;
  for(int i = 0; i < n; i++)
  {
    if((abs(real(cpu_transform_values[i]) - gpu_transform_values_fl[i].x) > EPSILON2) ||
       (abs(imag(cpu_transform_values[i]) - gpu_transform_values_fl[i].y) > EPSILON2))
    {
      OK = 0;
      cout << "Discrepancy at (" << i << ") " << real(cpu_transform_values[i]) << " " << gpu_transform_values_fl[i].x << " "
                                              << imag(cpu_transform_values[i]) << " " << gpu_transform_values_fl[i].y << endl;
    }
  }
  if(OK)
  {
    cout << "OK!" << endl;
  }
}

void Cleanup (int argc, char **argv, int iExitCode)
{
  // Cleanup allocated objects
  shrLog("Starting Cleanup...\n\n");
  if(cPathAndName)free(cPathAndName);
  if(cSourceCL)free(cSourceCL);
  if(ckKernel)clReleaseKernel(ckKernel);
  if(cpProgram)clReleaseProgram(cpProgram);
  if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
  if(cxGPUContext)clReleaseContext(cxGPUContext);
  if(cmDevPolyMultA)clReleaseMemObject(cmDevPolyMultA);
 
  // Free host memory
  free(cl_poly_a);
 
  // finalize logs and leave
  shrQAFinishExit(argc, (const char **)argv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED);
}
