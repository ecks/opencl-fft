#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__constant  float PI = 3.14159266;

double2 mul_complex(double2 a, double2 b)
{
  double2 c;
  c.x = a.x*b.x-a.y*b.y;
  c.y = a.x*b.y+a.y*b.x;
  return c;
}

double2 pow_complex(double2 a, int p)
{
  if(p == 1)
  {
    return a;
  }

  double2 multiplier = a;
  for(int i = 1; i < p; i++)
  {
    a = mul_complex(a, multiplier);
  }
  return a;
}

double2 exp_complex(double2 a)
{
  double2 b;
  b.x = exp(a.x)*(cos(a.y));
  b.y = exp(a.x)*(sin(a.y));
  return b;
}

__kernel void FFT(__global double2 * a, __global double2 * debug, __global int * inv)
{
  int p = get_group_id(0);
  int j = get_local_id(0);
  
  int l = get_local_size(0);
  int o = get_num_groups(0);
  int g = get_global_id(0);

  int s = log2((float)l);

  int m = 1;
  m <<= s+1;

  int k = p * m;

  double2 omega;
  omega.x = 1.0;
  omega.y = 0.0;

  double2 omega_begin;
  omega_begin.x = 0.0;
  if(*inv == 1)
  {
    omega_begin.y = 2.0 * PI / m;
  }
  else
  {
    omega_begin.y = -2.0 * PI / m;
  }

  double2 current_omega = exp_complex(omega_begin);

  if(j != 0)
  {
    omega = mul_complex(omega, pow_complex(current_omega, j));
  }
  
  debug[g] = omega;

  double2 t = mul_complex(omega,a[k + j + (m >> 1)]);
  double2 u = a[k + j];

  a[k + j] = u + t;
  a[k + j + (m >> 1)] = u - t; 

  barrier(CLK_LOCAL_MEM_FENCE);
} 
