// Copyright (c) Nex-AGI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// MTP-family clone of the 1-batch output projection + TP reduce translation unit.
//
// The MTP round is a separate CUDA-kernel family: even where the draft T=1
// math equals the 1-batch operation, the round launches ITS OWN __global__
// kernels through its own exported launchers and shared library, so the two
// families stay independently inspectable and evolvable. This TU compiles the
// 1-batch source under MTP-family symbol names: the device logic is shared by
// inclusion, while every global entry
// point, launch symbol, and the library identity are MTP-specific.
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#define NEXRT_MTP_RUNTIME_ONLY 1
#define OProjFusedPushKernel MtpOProjFusedPushKernel
#define NexN25ProOProjReduceLaunch NexN25ProMtpOProjReduceLaunch
#include "../full_attn/nex_n2_5_pro_o_proj_reduce_ctypes.cu"
