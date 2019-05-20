#include "scripts.h"
#include "structures.h"
#include "cud_defs.h"
#include "distances.h"
#include "knn.h"

int main(int argc, char const *argv[]) {
  int n_ratings, n_users, n_movies;
  int n_ratings_20, n_users_20, n_ratings_27, n_users_27, n_ratings_l, n_users_l, n_movies_27;

  n_ratings_27 = 27753444;
  n_users_27 = 283228;
  // n_movies_27 = 53889;

  n_ratings_20 = 20000263;
  n_users_20 = 138493;

  n_ratings_l = 49;
  n_users_l = 8;

  // n_ratings = n_ratings_20;
  // n_users = n_users_20;

  n_ratings = n_ratings_27;
  n_users = n_users_27;
  // n_movies = n_movies_27;

  // n_ratings = n_ratings_l;
  // n_users = n_users_l;

  int max_users = 300000;
  int max_movies = 200000;


  // n_ratings
  // n_of_users("../databases/libro/ratings.csv", n_ratings, n_users, true);
  // cout<<n_ratings<<" "<<n_users<<endl;

  float* values;
  int *row_ind, * col_ind;
  int * ind_users, *row_size;

  float* d_values;
  int *d_row_ind, * d_col_ind;
  int * d_ind_users, * d_row_size;

  float* item_values;
  int *item_row_ind, * item_col_ind;
  int * ind_items, *item_row_size;

  float* d_item_values;
  int *d_item_row_ind, * d_item_col_ind;
  int * d_ind_items, *d_item_row_size;

  d_values = cuda_array<float>(n_ratings);
  d_row_ind = cuda_array<int>(n_ratings);
  d_col_ind = cuda_array<int>(n_ratings);
  d_ind_users = cuda_array<int>(max_users);
  d_row_size = cuda_array<int>(max_users);

  d_item_values = cuda_array<float>(n_ratings);
  d_item_row_ind = cuda_array<int>(n_ratings);
  d_item_col_ind = cuda_array<int>(n_ratings);
  d_ind_items = cuda_array<int>(max_movies);
  d_item_row_size = cuda_array<int>(max_movies);



  map<int, string> movies_names;

  // read_ML_movies("../databases/ml-20m/movies.csv", movies_names, true);
  // read_ML_ratings("../databases/ml-20m/ratings.csv", n_ratings, n_users, true, values, row_ind, col_ind, ind_users, row_size, "27");

  // read_ML_movies("../../collaborative_filtering/databases/ml-latest/movies.csv", movies_names, true);
  // read_ML_ratings("../collaborative_filtering/databases/ml-latest/ratings.csv", n_ratings, n_users, true, values, row_ind, col_ind, ind_users, row_size, "27");

  read_ML_ratings("../collaborative_filtering/databases/ml-latest/ratings.csv", n_ratings, n_users, true, values, row_ind, col_ind, ind_users, row_size, "27");
  read_ML_ratings_items("../collaborative_filtering/databases/ml-latest/ratings.csv", n_ratings, n_users, max_movies, true,  item_values,  item_row_ind,  item_col_ind,  ind_items, item_row_size, "27");

  // read_ML_ratings("../collaborative_filtering/databases/libro/ratings.csv", n_ratings, n_users, true, values, row_ind, col_ind, ind_users, row_size, "l");
  // read_ML_ratings_items("../collaborative_filtering/databases/libro/ratings.csv", n_ratings, n_users, max_movies, true,  item_values,  item_row_ind,  item_col_ind,  ind_items, item_row_size, "l");


  // for (size_t i = 0; i < 100; i++) {
  //   cout<<item_row_size[i]<<endl;
  // }
  cuda_H2D<float>(values, d_values, n_ratings);
  cuda_H2D<int>(row_ind, d_row_ind, n_ratings);
  cuda_H2D<int>(col_ind, d_col_ind, n_ratings);
  cuda_H2D<int>(ind_users, d_ind_users, max_users);
  cuda_H2D<int>(row_size, d_row_size, max_users);

  cuda_H2D<float>(item_values, d_item_values, n_ratings);
  cuda_H2D<int>(item_row_ind, d_item_row_ind, n_ratings);
  cuda_H2D<int>(item_col_ind, d_item_col_ind, n_ratings);
  cuda_H2D<int>(ind_items, d_ind_items, max_movies);
  cuda_H2D<int>(item_row_size, d_item_row_size, max_movies);


  float *distances;
  bool * b_dists;
  vector<pair<int, float> > knns;
  int id_user = 16006;
  int k = 10;
  reloj r;
  r.start();
  distances_one2all(distances, b_dists, values, row_ind, col_ind, ind_users, row_size, item_values, item_row_ind, item_col_ind, ind_items, item_row_size, d_item_values, d_item_row_ind, d_item_col_ind, d_ind_items, d_item_row_size, n_users, max_users, id_user, PEARSON);
  knns = knn_greater_cuda(distances, b_dists, max_users, id_user, k);
  r.stop();
  cout<<"Tiempo total: "<<r.time()<<"ms"<<endl;
  for (size_t i = 0; i < k; i++) {
    cout<<knns[i].first<<" "<<knns[i].second<<endl;
  }
  // float* r1 = float_pointer(values, ind_users, 8);
  // int* c1 = int_pointer(col_ind, ind_users, 8);
  // for (size_t i = 0; i < row_size[8]; i++) {
  //   cout<<c1[i]<<" "<<r1[i]<<endl;
  // }

  return 0;
}
