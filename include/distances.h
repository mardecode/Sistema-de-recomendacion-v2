#ifndef SPARSE_OPERATIONS_H
#define SPARSE_OPERATIONS_H

#include <fstream>
#include "cud_defs.h"

#include "structures.h"
#include <set>


template<class T>
void initialize_arr(T* arr, int n, T val){
  for (size_t i = 0; i < n; i++) {
    arr[i] = val;
  }
}




__global__ void desviacion_x_user(float* d_values, int* d_ind_users, int* d_col_ind, float* d_desviaciones, int* d_cardinalidad, bool* d_b_dists, float val, int n_movies_x_user, int id_user){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n_movies_x_user){
    float* r1 = float_pointer(d_values, d_ind_users, id_user);
    int* c1 = int_pointer(d_col_ind, d_ind_users, id_user);
    d_desviaciones[c1[i]] += (val - r1[i]);
    d_cardinalidad[c1[i]]++;
    d_b_dists[c1[i]] = true;
  }
}

__global__ void coseno_x_user(float* d_values, int* d_ind_users, int* d_col_ind, float* d_coseno_num, float* d_coseno_den1, float* d_coseno_den2, float* d_averages, bool* d_b_dists, float val, int n_movies_x_user, int id_user){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n_movies_x_user){
    float* r1 = float_pointer(d_values, d_ind_users, id_user);
    int* c1 = int_pointer(d_col_ind, d_ind_users, id_user);
    d_coseno_num[c1[i]] += (val - d_averages[id_user]) *  (r1[i] - d_averages[id_user]);
    d_coseno_den1[c1[i]] += (val - d_averages[id_user]) *  (val - d_averages[id_user]);
    d_coseno_den2[c1[i]] += (r1[i] - d_averages[id_user]) *  (r1[i] - d_averages[id_user]);
    // d_cardinalidad[c1[i]]++;
    // d_cosenos[c1[i]] =
    d_b_dists[c1[i]] = true;
  }
}


void desviaciones_one2all(float*& desviaciones, int*& cardinalidad,  bool*& b_dists, float* item_values, int* item_row_ind, int* item_col_ind, int* ind_items, int* item_row_size,float* values, int* row_ind, int* col_ind, int* ind_users, int* row_size, float* d_values, int* d_row_ind, int* d_col_ind, int* d_ind_users, int* d_row_size, int n_movies, int max_movies, int id_movie){
  float block_size = 256;
  dim3 block =  dim3(block_size, 1, 1);

  b_dists = new bool[max_movies];
  bool* d_b_dists = cuda_array<bool>(max_movies);
  initialize_arr<bool>(b_dists, max_movies, false);
  cuda_H2D<bool>(b_dists, d_b_dists, max_movies);

  desviaciones = new float[max_movies];
  float* d_desviaciones = cuda_array<float>(max_movies);
  initialize_arr<float>(desviaciones, max_movies, 0);
  cuda_H2D<float>(desviaciones, d_desviaciones, max_movies);

  cardinalidad = new int[max_movies];
  int* d_cardinalidad = cuda_array<int>(max_movies);
  initialize_arr<int>(cardinalidad, max_movies, 0);
  cuda_H2D<int>(cardinalidad, d_cardinalidad, max_movies);


  float* ratings_movie = float_pointer(item_values, ind_items, id_movie);
  int* ids_users = int_pointer(item_col_ind, ind_items, id_movie);
  // for (size_t i = 0; i < row_size[id_user]; i++) {
  //   cout<<item_row_size[ids_movies[i]]<<endl;
  // }
  for (size_t i = 0; i < item_row_size[id_movie]; i++) {
    // cout<<"pelicula: "<<ids_movies[i]<<endl;
    int n_movies_x_user = row_size[ids_users[i]];
    // cout<< n_users_x_movie<<endl;

    dim3 grid =  dim3(ceil(n_movies_x_user / block_size), 1);

    desviacion_x_user<<<grid, block>>>(d_values, d_ind_users, d_col_ind, d_desviaciones, d_cardinalidad, d_b_dists, ratings_movie[i], n_movies_x_user, ids_users[i]);
    CHECK(cudaDeviceSynchronize());
  }

  cout<<"aas"<<endl;
  cuda_D2H(d_desviaciones, desviaciones, max_movies);
  cuda_D2H(d_b_dists, b_dists, max_movies);
  cuda_D2H(d_cardinalidad, cardinalidad, max_movies);

  for (size_t i = 0; i < 10; i++) {
    if(b_dists[i])
      cout<<"i: "<< i<<" "<<desviaciones[i]<<" "<<cardinalidad[i]<<"  ->  "<<(desviaciones[i] / cardinalidad[i])<<endl;
      // distances[i] = sqrt(distances[i]);
  }

}


void coseno_ajustado_one2all(float*& cosenos,  bool*& b_dists, float* item_values, int* item_row_ind, int* item_col_ind, int* ind_items, int* item_row_size,float* values, int* row_ind, int* col_ind, int* ind_users, int* row_size, float* d_values, int* d_row_ind, int* d_col_ind, int* d_ind_users, int* d_row_size, int n_movies, int max_movies, int id_movie, float* maxs, float* mins, float* averages){
  float block_size = 256;
  dim3 block =  dim3(block_size, 1, 1);

  b_dists = new bool[max_movies];
  bool* d_b_dists = cuda_array<bool>(max_movies);
  initialize_arr<bool>(b_dists, max_movies, false);
  cuda_H2D<bool>(b_dists, d_b_dists, max_movies);

  cosenos = new float[max_movies];
  float* d_cosenos = cuda_array<float>(max_movies);
  initialize_arr<float>(cosenos, max_movies, 0);
  cuda_H2D<float>(cosenos, d_cosenos, max_movies);

  float* coseno_num = new float[max_movies];
  float* d_coseno_num = cuda_array<float>(max_movies);
  initialize_arr<float>(coseno_num, max_movies, 0);
  cuda_H2D<float>(coseno_num, d_coseno_num, max_movies);

  float* coseno_den1 = new float[max_movies];
  float* d_coseno_den1 = cuda_array<float>(max_movies);
  initialize_arr<float>(coseno_den1, max_movies, 0);
  cuda_H2D<float>(coseno_den1, d_coseno_den1, max_movies);

  float* coseno_den2 = new float[max_movies];
  float* d_coseno_den2 = cuda_array<float>(max_movies);
  initialize_arr<float>(coseno_den2, max_movies, 0);
  cuda_H2D<float>(coseno_den2, d_coseno_den2, max_movies);


  float* d_averages = cuda_array<float>(max_movies);
  cuda_H2D<float>(averages, d_averages, max_movies);



  float* ratings_movie = float_pointer(item_values, ind_items, id_movie);
  int* ids_users = int_pointer(item_col_ind, ind_items, id_movie);
  // for (size_t i = 0; i < row_size[id_user]; i++) {
  //   cout<<item_row_size[ids_movies[i]]<<endl;
  // }
  for (size_t i = 0; i < item_row_size[id_movie]; i++) {
    // cout<<"pelicula: "<<ids_movies[i]<<endl;
    int n_movies_x_user = row_size[ids_users[i]];
    // cout<< n_users_x_movie<<endl;

    dim3 grid =  dim3(ceil(n_movies_x_user / block_size), 1);

    coseno_x_user<<<grid, block>>>(d_values, d_ind_users, d_col_ind, d_coseno_num, d_coseno_den1, d_coseno_den2, d_averages, d_b_dists, ratings_movie[i], n_movies_x_user, ids_users[i]);
    CHECK(cudaDeviceSynchronize());
  }

  cout<<"aas"<<endl;
  cuda_D2H(d_coseno_num, coseno_num, max_movies);
  cuda_D2H(d_coseno_den1, coseno_den1, max_movies);
  cuda_D2H(d_coseno_den2, coseno_den2, max_movies);
  cuda_D2H(d_b_dists, b_dists, max_movies);

  for (size_t i = 0; i < 10; i++) {
    if(b_dists[i]){
      cout<<"i: "<< i<<" "<<coseno_num[i] / (sqrt(coseno_den1[i]) * sqrt(coseno_den2[i]))<<" "<<"  ->  "<<endl;
      // distances[i] = sqrt(distances[i]);

    }
  }
  cout<<"fin"<<endl;

}








#endif
