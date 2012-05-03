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
        void transformGPU(const std::vector<Complex>& buf, void * cl_buf, void * cl_debug_buf, cl_mem cmDev, cl_mem cmInv, cl_mem cmDebug, cl_kernel ckKernel, size_t szGlobalWorkSize,
                          cl_command_queue cqCommandQueue, cl_int ciErr, int argc, const char **argv);
        static double getIntensity(Complex c);
        static double getPhase(Complex c);
        
    private:
        int n, lgN;
        bool inverse;
        std::vector<Complex> omega;
        std::vector<Complex> result;
        double start_t, end_t, clock_diff;
        cl_event start_event, end_event;
        cl_ulong start_time, end_time;
        double total_time;
        
        void bitReverseCopy(const std::vector<Complex>& src,
                std::vector<Complex>& dest) const;
};

#endif
