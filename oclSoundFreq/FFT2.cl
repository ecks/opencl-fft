#ifndef M_PI_F
	#define M_PI_F M_PI
#endif

float2 mul_complex(float2 a, float2 b)
{
  return (float2)(a.s0*b.s0 - a.s1*b.s1,a.s0*b.s1 + a.s1*b.s0);
}

float2 exp_complex(float2 a)
{
  return (float2)(exp(a.s0)*cos(a.s1), exp(a.s0)*sin(a.s1));
}

__kernel void FFT2(__global float2 * a, __local float2 * l, __global const uint * points_per_group, __global float2 * debug, __global const int * dir)
{
  int points_per_item = *points_per_group/get_local_size(0);
  int l_addr = get_local_id(0) * points_per_item;
  int a_addr = get_group_id(0) * *points_per_group + l_addr;
  int start_addr;  

  float2 u;
  float2 s;
  float2 v;
  float2 t;

  float2 sumus;
  float2 diffus;
  float2 sumvt;
  float2 diffvt;

  float2 omega;
  float2 cur_omega;
  int angle;

  // perform a 4-point FFT
  for(int i = 0; i < points_per_item; i+=4)
  {
    u = a[a_addr];
    s = a[a_addr+1];
    v = a[a_addr+2];
    t = a[a_addr+3];
  
    sumus = u + s;
    diffus = u - s;
    sumvt = v + t;
    diffvt = (float2)(v.s1 - t.s1, t.s0 - v.s0) * (*dir);
    l[l_addr] = sumus + sumvt;
    l[l_addr+1] = diffus + diffvt;
    l[l_addr+2] = sumus - sumvt;
    l[l_addr+3] = diffus - diffvt;
    l_addr += 4;
    a_addr += 4;
  }

  l_addr = get_local_id(0) * points_per_item;
  a_addr = get_group_id(0) * *points_per_group + l_addr;

//  for(int i = 0; i < points_per_item; i++)
//  {
//    debug[a_addr + i] = l[l_addr + i];
//  }

  // perform all other points necessary. we start at
  // s = 2 since we have already done previos two stages
  int m = 4;
  int lgppi = (int)log2((float)points_per_item);
  int lgppg = (int)log2((float)*points_per_group);

//  for(int s = 2; s < lgppi; ++s)
//  {
//    m <<= 1;
//    l_addr = get_local_id(0) * points_per_item; // reset index
//    omega = exp_complex((float2)(0.0, (*dir) * - 2.0 * M_PI_F / m));
//    for(int k = 0; k < points_per_item; k += m)
//    {
//      cur_omega = (float2)(1.0,0.0);
//      for(int j = 0; j < (m >> 1); ++j)
//      {
//        t = mul_complex(omega, l[l_addr + (m >> 1) + k + j]);
//        u = l[l_addr + k + j];
//        l[l_addr + k + j] = u + t;
//        l[l_addr + (m >> 1) + k + j] = u - t;
//        cur_omega = mul_complex(cur_omega, omega);
//      }
//    }
//  }
  
  barrier(CLK_LOCAL_MEM_FENCE); // synchronize all the threads

  for(int s = lgppi; s < lgppg ; ++s)
  {
    m <<= 1;
    start_addr = (get_local_id(0) + (get_local_id(0) / (m >> 2)) * (m >> 2)) * (points_per_item/2);
    angle = start_addr % m;
    for(int j = start_addr; j < start_addr + points_per_item/2; ++j)
    {
      omega = exp_complex((float2)(0.0, (*dir) * - M_PI_F * angle / (m >> 1)));
      t = mul_complex( omega, l[j + (m >> 1)]);
      u = l[j];
      l[j] = u + t;
      l[j + (m >> 1)] = u - t;
      angle++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

//    l_addr = get_local_id(0) * points_per_item;
//    a_addr = get_group_id(0) * *points_per_group + l_addr;

//    if(s == (lgppg-1))
//    {
//      for(int i = 0; i < points_per_item; i++)
//      {
//        debug[a_addr + i] = l[a_addr + i];
//      }
//    }
  }

  l_addr = get_local_id(0) * points_per_item;
  a_addr = get_group_id(0) * *points_per_group + l_addr;
  for(int i = 0; i < points_per_item; i++)
  {
    a[a_addr + i] = l[l_addr + i];
  }
}

__kernel void FFT2_ALL_POINTS(__global float2 * a, __global const uint * m, __global const uint * points_per_group, __global const int * dir)
{
  int points_per_item = *points_per_group/get_local_size(0);
  int start_addr = (get_global_id(0) + (get_global_id(0) / (*m >> 2)) * (*m >> 2)) * (points_per_item/2);
  int angle = start_addr % *m;
  float2 omega;

  float2 t;
  float2 u;

  for(int j = start_addr; j < start_addr + points_per_item/2; ++j)
  {
    omega = exp_complex((float2)(0.0, (*dir) * - M_PI_F * angle / (*m >> 1)));
    t = mul_complex( omega, a[j + (*m >> 1)]);
    u = a[j];
    a[j] = u + t;
    a[j + (*m >> 1)] = u - t;
    angle++;
  }
}
