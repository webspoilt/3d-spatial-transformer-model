#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// CUDA Kernel to voxelize raw LiDAR point clouds extremely fast.
// This completely bypasses Python's Global Interpreter Lock (GIL) and CPU bottlenecks,
// preparing the 3D grid directly on the GPU memory for PyTorch ingestion.

__global__ void VoxelizePointCloudKernel(
    const float* raw_points,  // Input: [N, 3] raw (x,y,z) coordinates
    int* voxel_grid,          // Output: Flattened sparse voxel grid
    int num_points,           // Total number of LiDAR points (e.g., 100,000)
    float voxel_size,         // Size of each 3D cube (e.g., 0.05m)
    int grid_dim_x,           // Voxel grid dimensions
    int grid_dim_y,
    int grid_dim_z,
    float min_x, float min_y, float min_z // Bounding box origin
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx < num_points) {
        // Read point
        float x = raw_points[idx * 3 + 0];
        float y = raw_points[idx * 3 + 1];
        float z = raw_points[idx * 3 + 2];
        
        // Calculate corresponding voxel indices
        int voxel_x = (int)((x - min_x) / voxel_size);
        int voxel_y = (int)((y - min_y) / voxel_size);
        int voxel_z = (int)((z - min_z) / voxel_size);
        
        // Discard out-of-bounds points
        if (voxel_x >= 0 && voxel_x < grid_dim_x &&
            voxel_y >= 0 && voxel_y < grid_dim_y &&
            voxel_z >= 0 && voxel_z < grid_dim_z) {
            
            // Calculate flat index (Row-major 3D->1D mapping)
            int flat_idx = voxel_z * (grid_dim_x * grid_dim_y) + 
                           voxel_y * grid_dim_x + 
                           voxel_x;
            
            // Atomic add to mark voxel as occupied (or count density)
            atomicAdd(&voxel_grid[flat_idx], 1);
        }
    }
}

// C wrapper callable from Python via pybind11 or ctypes
extern "C" void launch_voxelization(
    const float* raw_points_d, 
    int* voxel_grid_d, 
    int num_points, 
    float voxel_size,
    int g_x, int g_y, int g_z,
    float m_x, float m_y, float m_z
) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    
    VoxelizePointCloudKernel<<<blocksPerGrid, threadsPerBlock>>>(
        raw_points_d, voxel_grid_d, num_points, voxel_size, 
        g_x, g_y, g_z, m_x, m_y, m_z
    );
    
    cudaDeviceSynchronize();
}
