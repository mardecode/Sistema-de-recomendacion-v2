#ifndef SPARSE_OPERATIONS_H
#define SPARSE_OPERATIONS_H

#include <fstream>
#include "cud_defs.h"
#include "distances.h"
#include "structures.h"


void distances_one2all(float*& distances, float* d_values, int* d_row_ind, int* d_col_ind, int* d_ind_users, int* d_row_size, int n_users, int max_users, int id_user){
  float block_size = 256;
  dim3 block =  dim3(block_size, 1, 1);
  dim3 grid =  dim3(ceil(n_users / block_size), 1);

  distances = new float[max_users];
  float* d_distances = cuda_array<float>(max_users);






  
}



#endif
