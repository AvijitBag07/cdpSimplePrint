/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <helper_cuda.h>
#include <helper_string.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// Variable on the GPU used to generate unique identifiers of blocks.
////////////////////////////////////////////////////////////////////////////////
static dpct::global_memory<int, 0> g_uids(0);

////////////////////////////////////////////////////////////////////////////////
// Print a simple message to signal the block which is currently executing.
////////////////////////////////////////////////////////////////////////////////
void print_info(int depth, int thread, int uid, int parent_uid,
                const sycl::nd_item<3> &item_ct1, const sycl::stream &stream_ct1) {
  if (item_ct1.get_local_id(2) == 0) {
    if (depth == 0)
      /*
      DPCT1015:0: Output needs adjustment.
      */
      stream_ct1 << "BLOCK"<<uid<<"launched by the host\n";
    else {
      char buffer[32];

      for (int i = 0; i < depth; ++i) {
        buffer[3 * i + 0] = '|';
        buffer[3 * i + 1] = ' ';
        buffer[3 * i + 2] = ' ';
      }

      buffer[3 * depth] = '\0';
      /*
      DPCT1015:1: Output needs adjustment.
      */
      stream_ct1 <<buffer <<"BLOCK"<<uid<<"launched by thread "<<thread<<" of block"<<parent_uid<<"\n";
    }
  }

  /*
  DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
}

////////////////////////////////////////////////////////////////////////////////
// The kernel using CUDA dynamic parallelism.
//
// It generates a unique identifier for each block. Prints the information
// about that block. Finally, if the 'max_depth' has not been reached, the
// block launches new blocks directly from the GPU.
////////////////////////////////////////////////////////////////////////////////
/*
DPCT1109:2: Recursive functions cannot be called in SYCL device code. You need
to adjust the code.
*/
void cdp_kernel(int max_depth, int depth, int thread, int parent_uid,
                const sycl::nd_item<3> &item_ct1,
                const sycl::stream &stream_ct1, int &g_uids, int &s_uid) {
  // We create a unique ID per block. Thread 0 does that and shares the value
  // with the other threads.

  if (item_ct1.get_local_id(2) == 0) {
    s_uid = dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &g_uids, 1);
  }

  /*
  DPCT1065:21: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // We print the ID of the block and information about its parent.
  print_info(depth, thread, s_uid, parent_uid, item_ct1, stream_ct1);

  // We launch new blocks if we haven't reached the max_depth yet.
  if (++depth >= max_depth) {
    return;
  }

  /*
  DPCT1130:3: SYCL 2020 standard does not support dynamic parallelism (launching
  kernel in device code). Please rewrite the code.
  */
  /*
  DPCT1109:4: Recursive functions cannot be called in SYCL device code. You need
  to adjust the code.
  */
//  cdp_kernel<<<item_ct1.get_group_range(2), item_ct1.get_local_range(2)>>>(
 //     max_depth, depth, item_ct1.get_local_id(2), s_uid, item_ct1, stream_ct1,
  //    g_uids, s_uid);
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("starting Simple Print (CUDA Dynamic Parallelism)\n");

  // Parse a few command-line arguments.
  int max_depth = 2;

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "h")) {
    printf(
        "Usage: %s depth=<max_depth>\t(where max_depth is a value between 1 "
        "and 8).\n",
        argv[0]);
    exit(EXIT_SUCCESS);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "depth")) {
    max_depth = getCmdLineArgumentInt(argc, (const char **)argv, "depth");

    if (max_depth < 1 || max_depth > 8) {
      printf("depth parameter has to be between 1 and 8\n");
      exit(EXIT_FAILURE);
    }
  }

  // Find/set the device.
  int device = -1;
  dpct::device_info deviceProp;
  device = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_device(device).get_device_info(deviceProp)));

  /*
  DPCT1005:22: The SYCL device version is different from CUDA Compute
  Compatibility. You may need to rewrite this code.
  */
  if (!(deviceProp.get_major_version() > 3 ||
        /*
        DPCT1005:23: The SYCL device version is different from CUDA Compute
        Compatibility. You may need to rewrite this code.
        */
        (deviceProp.get_major_version() == 3 &&
         deviceProp.get_minor_version() >= 5))) {
    //printf("GPU %d - %s  does not support CUDA Dynamic Parallelism\n Exiting.",
      //     device, deviceProp.get_name());
    //exit(EXIT_WAIVED);
  }

  // Print a message describing what the sample does.
  printf(
      "*********************************************************************"
      "******\n");
  printf(
      "The CPU launches 2 blocks of 2 threads each. On the device each thread "
      "will\n");
  printf(
      "launch 2 blocks of 2 threads each. The GPU we will do that "
      "recursively\n");
  printf("until it reaches max_depth=%d\n\n", max_depth);
  int b = 2, t=2;
  printf("In total %d", b);
  int num_blocks = b, sum = t;

  for (int i = 1; i < max_depth; ++i) {
    num_blocks *= b*t;
    printf("+%d", num_blocks);
    sum = b+num_blocks;
  }

  printf("=%d blocks are launched!!! (%d from the GPU)\n", sum, sum - b);
  printf(
      "************************************************************************"
      "***\n\n");

  // Launch the kernel from the CPU.
  printf("Launching cdp_kernel() with CUDA Dynamic Parallelism:\n\n");
  /*
  DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    g_uids.init();

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      sycl::stream stream_ct1(64 * 1024, 80, cgh);

      auto g_uids_ptr_ct1 = g_uids.get_ptr();

      sycl::local_accessor<int, 0> s_uid_acc_ct1(cgh);

      auto ct3 = -1;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, b) * sycl::range<3>(1, 1, t),
                            sycl::range<3>(1, 1, t)),
          [=](sycl::nd_item<3> item_ct1) {
            cdp_kernel(max_depth, 0, 0, ct3, item_ct1, stream_ct1,
                       *g_uids_ptr_ct1, s_uid_acc_ct1);
          });
    });
  }
  /*
  DPCT1010:24: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  checkCudaErrors(0);

  // Finalize.
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));

  exit(EXIT_SUCCESS);
}
