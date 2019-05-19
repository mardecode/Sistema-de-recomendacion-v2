#ifndef SPARSE_OPERATIONS_H
#define SPARSE_OPERATIONS_H

#include <fstream>
#include "cud_defs.h"
#include "distances.h"
#include "structures.h"
#include <set>

bool compare_greater(const pair<float, int>&i, const pair<float, int>&j)
{
  if(i.first == j.first){
    return i.second < j.second;
  }
  return i.first > j.first;
}

template<class T>
void initialize_arr(T* arr, int n, T val){
  for (size_t i = 0; i < n; i++) {
    arr[i] = val;
  }
}

  __global__ void manhatta_x_item(float* d_item_values, int* d_ind_items, int* d_item_col_ind, float* d_distances, bool* d_b_dists, float val, int n_users_x_movie, int id_movie){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n_users_x_movie){
    float* r1 = float_pointer(d_item_values, d_ind_items, id_movie);
    int* c1 = int_pointer(d_item_col_ind, d_ind_items, id_movie);
    // d_distances[c1[i]] += abs(r1[i] - val);
    d_distances[c1[i]] += (r1[i] - val) *(r1[i] - val);
    d_b_dists[c1[i]] = true;
  }
}

void distances_one2all_manhattan(float*& distances, float* values, int* row_ind, int* col_ind, int* ind_users, int* row_size,float* item_values, int* item_row_ind, int* item_col_ind, int* ind_items, int* item_row_size, float* d_item_values, int* d_item_row_ind, int* d_item_col_ind, int* d_ind_items, int* d_item_row_size, int n_users, int max_users, int id_user){
  float block_size = 256;
  dim3 block =  dim3(block_size, 1, 1);

  bool* b_dists = new bool[max_users];
  bool* d_b_dists = cuda_array<bool>(max_users);
  initialize_arr<bool>(b_dists, max_users, false);
  cuda_H2D<bool>(b_dists, d_b_dists, max_users);

  distances = new float[max_users];
  float* d_distances = cuda_array<float>(max_users);
  initialize_arr<float>(distances, max_users, 0);
  cuda_H2D<float>(distances, d_distances, max_users);

  float* ratings_user = float_pointer(values, ind_users, id_user);
  int* ids_movies = int_pointer(col_ind, ind_users, id_user);
  // for (size_t i = 0; i < row_size[id_user]; i++) {
  //   cout<<item_row_size[ids_movies[i]]<<endl;
  // }
  for (size_t i = 0; i < row_size[id_user]; i++) {
    // cout<<"pelicula: "<<ids_movies[i]<<endl;
    int n_users_x_movie = item_row_size[ids_movies[i]];
    // cout<< n_users_x_movie<<endl;

    dim3 grid =  dim3(ceil(n_users_x_movie / block_size), 1);

    manhatta_x_item<<<grid, block>>>(d_item_values, d_ind_items, d_item_col_ind, d_distances, d_b_dists, ratings_user[i], n_users_x_movie, ids_movies[i]);
    CHECK(cudaDeviceSynchronize());
  }
  cuda_D2H(d_distances, distances, max_users);
  cuda_D2H(d_b_dists, b_dists, max_users);

  // set<pair<float, int>, decltype(&compare_greater)> mapa(&compare_greater);
  set<pair<float, int>, less<pair<float, int> > > mapa;
  for (size_t i = 0; i < max_users; i++) {
    if(b_dists[i]){
      mapa.insert(make_pair(sqrt(distances[i]), i));
      // mapa[distances[i]] = dis
      // cout<<"i: "<<i<<" -> "<<distances[i]<<endl;

    }
  }


  auto it = mapa.begin();
  int i = 0;
  for (; it != mapa.end() && (i < 10); it++) {
    i++;
    cout<<it->second<<" "<<it->first<<endl;
  }

}



#endif
