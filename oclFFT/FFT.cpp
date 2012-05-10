#include "FFT.h"
#include "oclFFT.h"
#include <vector>
#include <cassert>
#include <iostream>
#include <ctime>

#define PI 3.14159265

using namespace std;

FFT::FFT(int n, bool inverse)
    : n(n), inverse(inverse), result(vector<Complex>(n))
{
    lgN = 0;
    for (int i = n; i > 1; i >>= 1)
    {
        ++lgN;
        assert((i & 1) == 0);
    }
    omega.resize(lgN);
    int m = 1;
    for (int s = 0; s < lgN; ++s)
    {
        m <<= 1;
        if (inverse)
            omega[s] = exp(Complex(0, 2.0 * PI / m));
        else
        {
            Complex tmp = Complex(0, -2.0 * PI / m);
            omega[s] = exp(tmp);
            cout << "Omega (" << real(omega[s]) << "," << imag(omega[s]) << "), tmp (" << real(tmp) << "," << imag(tmp) << ")" << endl;

        }
    }
}

std::vector<FFT::Complex> FFT::transform(const vector<Complex>& buf)
{
    bitReverseCopy(buf, result);

    for(int i = 0; i < n; i++)
    {
      cout << "Index " << i << ": (before) " << real(result[i]) << " " << imag(result[i]) << endl;
    }
    cout << endl;

    int m = 1;

    start_t = clock();
    for (int s = 0; s < lgN; ++s)
    {
        m <<= 1;
        for (int k = 0; k < n; k += m)
        {
            Complex current_omega = 1;
            for (int j = 0; j < (m >> 1); ++j)
            {
                cout << "Current omega (" << real(current_omega) << "," << imag(current_omega) << ") for j = " << j << endl;
                cout << "Omega (" << real(omega[s]) << "," << imag(omega[s]) << ")" << endl;
                Complex t = current_omega * result[k + j + (m >> 1)];
                Complex u = result[k + j];
                result[k + j] = u + t;
                result[k + j + (m >> 1)] = u - t;
                current_omega *= omega[s];
            }
        }
        for(int i = 0; i < n; i++)
        {
          cout << "Index " << i << ": (after) (s:" << s << ") " << real(result[i]) << " " << imag(result[i]) << endl;
        }
        cout << endl;

    }

    end_t = clock();
    clock_diff = end_t - start_t;
    clock_diff_sec = (double)(clock_diff/1000000.0);
    shrLog("CPU transform diff seconds\t %f \n", clock_diff_sec);

    
    if (inverse == false)
        for (int i = 0; i < n; ++i)
            result[i] /= n;
    for(int i = 0; i < n; i++)
    {
      cout << "Index " << i << ": (after) " << real(result[i]) << " " << imag(result[i]) << endl;
    }
    cout << endl;

    return result;
}

void FFT::transformGPU(const vector<Complex>& buf, void * cl_buf, void * cl_debug_buf, cl_mem cmDev, 
                       cl_mem cmPointsPerGroup, cl_mem cmDebug, cl_mem cmDir, cl_kernel ckKernel, size_t szGlobalWorkSize, size_t szLocalWorkSize, unsigned int points_per_group,
                       cl_command_queue cqCommandQueue, cl_int ciErr, int argc, const char **argv)
{
//  size_t szLocalWorkSize;
  int dir_i = (inverse) ? -1 : 1;
  void * dir = (void *)&dir_i;
  void * pts_per_grp_p = (void *)&points_per_group;
  bitReverseCopy(buf, result);
  cl_float2 * cl_float2_buf = (cl_float2 *)cl_buf;
  cl_float2 * cl_float2_debug_buf = (cl_float2 *)cl_debug_buf;
  for(int i = 0; i < n; i++)
  {
    cl_float2_buf[i].s0 = (float)real(result[i]);
    cl_float2_buf[i].s1 = (float)imag(result[i]);
    cl_float2_debug_buf[i].s0 = -1.0;
    cl_float2_debug_buf[i].s1 = -1.0;
  }

  for(int i = 0; i < n; i++)
  {
    cout << "Index " << i << ": (before) " << cl_float2_buf[i].s0 << " " << cl_float2_buf[i].s1 << endl;
  }
  cout << endl;

  ciErr = clEnqueueWriteBuffer(cqCommandQueue, cmDev, CL_FALSE, 0, sizeof(cl_float2) * n, cl_buf, 0, NULL, NULL);
  if (ciErr != CL_SUCCESS)
  {
    shrLog("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    Cleanup(argc, (char **)argv, EXIT_FAILURE);
  }

  ciErr = clEnqueueWriteBuffer(cqCommandQueue, cmPointsPerGroup, CL_FALSE, 0, sizeof(cl_uint), pts_per_grp_p, 0, NULL, NULL);
  if (ciErr != CL_SUCCESS)
  {
    shrLog("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    Cleanup(argc, (char **)argv, EXIT_FAILURE);
  }

  ciErr = clEnqueueWriteBuffer(cqCommandQueue, cmDebug, CL_FALSE, 0, sizeof(cl_float2) * n, cl_debug_buf, 0, NULL, NULL);
  if (ciErr != CL_SUCCESS)
  {
    shrLog("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    Cleanup(argc, (char **)argv, EXIT_FAILURE);
  }

  ciErr = clEnqueueWriteBuffer(cqCommandQueue, cmDir, CL_FALSE, 0, sizeof(cl_int), dir, 0, NULL, NULL);
  if (ciErr != CL_SUCCESS)
  {
    shrLog("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    Cleanup(argc, (char **)argv, EXIT_FAILURE);
  }



  start_t = clock();
//  for(int s = 0; s < lgN; ++s)
//  {
//    m <<= 1;
//    szLocalWorkSize = m >> 1;

    cout << "Enqueue with Global Work Size " << szGlobalWorkSize << " and Local Work Size " << szLocalWorkSize
         << " and points per group " << points_per_group << endl;
    // Launch kernel
    ciErr = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    if (ciErr != CL_SUCCESS)
    {
      shrLog("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
      Cleanup(argc, (char **)argv, EXIT_FAILURE);
    }

    clFinish(cqCommandQueue);

    ciErr = clEnqueueReadBuffer(cqCommandQueue, cmDebug, CL_TRUE, 0, sizeof(cl_float2) * n, cl_debug_buf, 0, NULL, NULL);
    if (ciErr != CL_SUCCESS)
    {
      shrLog("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
     Cleanup(argc, (char **)argv, EXIT_FAILURE);
    } 

//    ciErr = clEnqueueReadBuffer(cqCommandQueue, cmDev, CL_TRUE, 0, sizeof(cl_float2) * n, cl_buf, 0, NULL, NULL);
//    if (ciErr != CL_SUCCESS)
//    {
//      shrLog("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
//     Cleanup(argc, (char **)argv, EXIT_FAILURE);
//    } 

    for(int i = 0; i < n; i++)
    {
      cout << "Index " << i << " with intermediate = (" << cl_float2_debug_buf[i].s0 << "," << 
                     														       cl_float2_debug_buf[i].s1 << ")" << endl;
      cl_float2_debug_buf[i].s0 = -1;
      cl_float2_debug_buf[i].s1 = -1;
    }
    cout << endl;

//  }

  end_t = clock();
  clock_diff = end_t - start_t;
  clock_diff_sec = (double)(clock_diff/1000000.0);
  shrLog("CPU transform diff seconds\t %f \n", clock_diff_sec);

  ciErr = clEnqueueReadBuffer(cqCommandQueue, cmDev, CL_TRUE, 0, sizeof(cl_float2) * n, cl_buf, 0, NULL, NULL);
  if (ciErr != CL_SUCCESS)
  {
    shrLog("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    Cleanup(argc, (char **)argv, EXIT_FAILURE);
  } 

  for(int i = 0; i < n; i++)
  {
    cout << "Index " << i << ": (after) " << cl_float2_buf[i].s0 << " " << cl_float2_buf[i].s1 << endl;
  }
  cout << endl;

  if(inverse == false)
  {
    for(int i = 0; i < n; ++i)
    {
      cl_float2_buf[i].s0 = cl_float2_buf[i].s0 / n;
      cl_float2_buf[i].s1 = cl_float2_buf[i].s1 / n;
    }
  }

}

double FFT::getIntensity(Complex c)
{
    return abs(c);
}

double FFT::getPhase(Complex c)
{
    return arg(c);
}

void FFT::bitReverseCopy(const vector<Complex>& src, vector<Complex>& dest)
        const
{
    for (int i = 0; i < n; ++i)
    {
        int index = i, rev = 0;
        for (int j = 0; j < lgN; ++j)
        {
            rev = (rev << 1) | (index & 1);
            index >>= 1;
        }
        dest[rev] = src[i];
    }
}
