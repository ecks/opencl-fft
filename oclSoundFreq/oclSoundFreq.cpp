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

int samples_per_second = 1024;

void opencl_init(int n, int argc, const char **argv);
void compareValues(vector<FFT::Complex> cpu_transform_values, void * gpu_transform_values, int n);

const char* cSourceFile = "FFT2.cl";

void * cl_complex,* cl_debug;
cl_ulong local_memory_size;
unsigned int points_per_group;
unsigned int points_per_item;

cl_context cxGPUContext;
cl_command_queue cqCommandQueue;
cl_platform_id cpPlatform;
cl_device_id cdDevice;
cl_program cpProgram;
cl_kernel ckKernel;
cl_mem cmDevComplex;
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

int main(int argc, const char * argv[])
{
  samples_per_second = atoi(argv[1]); 
  FILE* f = fopen("pcm.pcm", "rb");
  fseek(f, 0, SEEK_END);
  int n = ftell(f) / 2;
  cout << "Number of samples: " << n << endl;
  opencl_init(n, argc, argv);
  rewind(f);
  short* buf = new short[n];
  fread(buf, n, 2, f);
  fclose(f);
 
  vector<FFT::Complex> buf_complex(n);
  for (int i = 0; i < n; ++i)
    buf_complex[i] = buf[i];
  delete[] buf;
  FFT dft(n);
  vector<FFT::Complex> frequencies = dft.transform(buf_complex);

  ciErr1 = clGetKernelWorkGroupInfo(ckKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *)&items_per_group, NULL);
  ciErr1 |= clGetDeviceInfo(cdDevice, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_memory_size, NULL);

  size_t l_mem_size = (sizeof(cl_float2) * n) > (local_memory_size/2) ? (local_memory_size/2) : (sizeof(cl_float2) * n);

  // Set the Argument values
  ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevComplex);
  ciErr1 |= clSetKernelArg(ckKernel, 1,l_mem_size , NULL);
  ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmPointsPerGroup);
  ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&cmDevDebug);
  ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void*)&cmDir);
//  shrLog("clSetKernelArg 0...4\n\n");
  if (ciErr1 != CL_SUCCESS)
  {
    shrLog("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    Cleanup(argc, (char **)argv, EXIT_FAILURE);
  }
 
  points_per_group = local_memory_size/(2*(2*sizeof(float)));
  points_per_item = (points_per_group/(items_per_group/2));

  szLocalWorkSize = (num_points/points_per_item) > (items_per_group/2) ? (items_per_group/2) : (num_points/points_per_item);
  szGlobalWorkSize = num_points/points_per_item;
//  points_per_group = num_points;

//  szLocalWorkSize = szLocalWorkSize * 2;
//  szGlobalWorkSize = szGlobalWorkSize * 2;

//  cout << "Points per group: " << points_per_group << " local memory size: " << l_mem_size << endl;

  dft.transformGPU(buf_complex, cl_complex, cl_debug, cmDevComplex, 
                   cmPointsPerGroup, cmDevDebug, cmDir, ckKernel, szGlobalWorkSize, szLocalWorkSize, points_per_group,
                   cqCommandQueue, ciErr1, argc, (const char **)argv);
  compareValues(frequencies, cl_complex, n);
 
  for (int k = 0; k < (n >> 1); ++k)
    if (dft.getIntensity(frequencies[k]) > 100)
      cout << (k * samples_per_second / n) << " => "
           << dft.getIntensity(frequencies[k]) << endl;
}

void opencl_init(int n, int argc, const char **argv)
{
    shrQAStart(argc, (char **)argv);
    // set logfile name and start logs
    shrSetLogFileName("oclFFT.txt");
    num_points = n;
//    shrLog("%s Starting...\n\n# of elements per Array \t= %i\n", argv[0], n);
    
//    shrLog("Initializing data...\n");
    cl_complex = (void *)malloc(sizeof(cl_float2) * n);

    cl_debug = (void *)malloc(sizeof(cl_float2) * n);

    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);
    
//    shrLog("clGetPlatformID...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }   

   //Get the devices
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
//    shrLog("clGetDeviceIDs...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }   

    //Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
//    shrLog("clCreateContext...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }   

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
//    shrLog("clCreateCommandQueue...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }   

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cmDevComplex = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * n, NULL, &ciErr1);

//    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 

    cmPointsPerGroup = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &ciErr1);

//    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 


    cmDevDebug = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * n, NULL, &ciErr1);

//    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 

    cmDir = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &ciErr1);

//    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 


     // Read the OpenCL kernel in from source file
//    shrLog("oclLoadProgSource (%s)...\n", cSourceFile);
    cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);
    
    // Create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErr1);
//    shrLog("clCreateProgramWithSource...\n");
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
//    shrLog("clBuildProgram...\n");
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
//    shrLog("clCreateKernel (FFT2)...\n");
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }
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
  if(cmDevComplex)clReleaseMemObject(cmDevComplex);
 
  // Free host memory
  free(cl_complex);
 
  // finalize logs and leave
  shrQAFinishExit(argc, (const char **)argv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED);
}
