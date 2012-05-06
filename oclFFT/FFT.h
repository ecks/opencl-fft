#ifndef _FFT_H_
#define _FFT_H_

#include <oclUtils.h>
#include <shrQATest.h>
#include <complex>
#include <vector>
#include <ctime>
class FFT
{
    public:
        typedef std::complex<double> Complex;
        
        /* Initializes FFT. n must be a power of 2. */
        FFT(int n, bool inverse = false);
        /* Computes Discrete Fourier Transform of given buffer. */
        std::vector<Complex> transform(const std::vector<Complex>& buf);
        void transformGPU(const std::vector<Complex>& buf, void * cl_buf, void * cl_debug_buf, cl_mem cmDev, 
                          cl_mem cmPointsPerGroup, cl_mem cmDir, cl_mem cmDebug, cl_kernel ckKernel, size_t szGlobalWorkSize, size_t szLocalWorkSize, unsigned int points_per_group,
                          cl_command_queue cqCommandQueue, cl_int ciErr, int argc, const char **argv);
        static double getIntensity(Complex c);
        static double getPhase(Complex c);
        
    private:
        int n, lgN;
        bool inverse;
        std::vector<Complex> omega;
        std::vector<Complex> result;
        clock_t start_t, end_t, clock_diff;
        double clock_diff_sec;
        
        void bitReverseCopy(const std::vector<Complex>& src,
                std::vector<Complex>& dest) const;
};

#endif
