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

const int samples_per_second = 8192;

void opencl_init(int n, int argc, const char **argv);
void compareValues(vector<FFT::Complex> cpu_transform_values, void * gpu_transform_values, int n);

const char* cSourceFile = "FFT.cl";

void * cl_complex,* cl_debug;

cl_context cxGPUContext;
cl_command_queue cqCommandQueue;
cl_platform_id cpPlatform;
cl_device_id cdDevice;
cl_program cpProgram;
cl_kernel ckKernel;
cl_mem cmDevComplex;
cl_mem cmDevPolyMultB;
cl_mem cmDevPolyMultC;
cl_mem cmDevDebug;
cl_mem cmInv;
size_t szGlobalWorkSize;
size_t szKernelLength;
cl_int ciErr1;
char* cPathAndName = NULL;
char* cSourceCL = NULL;


int main(int argc, const char **argv)
{ 
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

  // Set the Argument values
  ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevComplex);
  ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmDevDebug);
  ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmInv);
  shrLog("clSetKernelArg 0...2\n\n");
  if (ciErr1 != CL_SUCCESS)
  {
    shrLog("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    Cleanup(argc, (char **)argv, EXIT_FAILURE);
  }
 
  dft.transformGPU(buf_complex, cl_complex, cl_debug, cmDevComplex, cmDevDebug, cmInv, ckKernel, szGlobalWorkSize, cqCommandQueue, ciErr1, argc, (const char **)argv);
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
    szGlobalWorkSize = n/2;
    shrLog("%s Starting...\n\n# of elements per Array \t= %i\n", argv[0], n);
    
    shrLog("Initializing data...\n");
    cl_complex = (void *)malloc(sizeof(cl_double2) * n);

    cl_debug = (void *)malloc(sizeof(cl_double2) * n);

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
    cmDevComplex = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double2) * n, NULL, &ciErr1);

    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 

    cmDevDebug = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double2) * n, NULL, &ciErr1);

    shrLog("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 

    cmInv = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &ciErr1);

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
      shrLog("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }
    // Create the kernel
    ckKernel = clCreateKernel(cpProgram, "FFT", &ciErr1);
    shrLog("clCreateKernel (FFT)...\n");
    if (ciErr1 != CL_SUCCESS)
    {   
      shrLog("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }

    size_t max_work_group_size;
    cl_ulong local_memory_size;
    ciErr1 = clGetKernelWorkGroupInfo(ckKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *)&max_work_group_size, NULL);
    ciErr1 |= clGetDeviceInfo(cdDevice, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), (void *)&local_memory_size, NULL);

    cout << "Max work group size: " << max_work_group_size << " Local memory size : " << local_memory_size << endl;
    
}

void compareValues(vector<FFT::Complex> cpu_transform_values, void * gpu_transform_values, int n)
{
  cl_double2 * gpu_transform_values_fl = (cl_double2 *)gpu_transform_values;
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
