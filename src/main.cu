#include "scripts.h"
#include "structures.h"
#include "cud_defs.h"

int main(int argc, char const *argv[]) {
  int n_ratings, n_users;
  int n_ratings_20, n_users_20, n_ratings_27, n_users_27, n_ratings_l, n_users_l;

  n_ratings_27 = 27753444;
  n_users_27 = 283228;

  n_ratings_20 = 20000263;
  n_users_20 = 138493;

  n_ratings_l = 49;
  n_users_l = 8;

  // n_ratings = n_ratings_20;
  // n_users = n_users_20;

  n_ratings = n_ratings_27;
  n_users = n_users_27;

  int max_users = 300000;


  // n_ratings
  // n_of_users("../databases/libro/ratings.csv", n_ratings, n_users, true);
  // cout<<n_ratings<<" "<<n_users<<endl;

  float* values;
  int *row_ind, * col_ind;
  int * ind_users, *row_size;

  float* d_values;
  int *d_row_ind, * d_col_ind;
  int * d_ind_users, * d_row_size;

  d_values = cuda_array<float>(n_ratings);
  d_row_ind = cuda_array<int>(n_ratings);
  d_col_ind = cuda_array<int>(n_ratings);
  d_ind_users = cuda_array<int>(max_users);
  d_row_size = cuda_array<int>(max_users);

  map<int, string> movies_names;

  // read_ML_movies("../databases/ml-20m/movies.csv", movies_names, true);
  // read_ML_ratings("../databases/ml-20m/ratings.csv", n_ratings, n_users, true, values, row_ind, col_ind, ind_users, row_size, "27");

  // read_ML_movies("../../collaborative_filtering/databases/ml-latest/movies.csv", movies_names, true);
  read_ML_ratings("../collaborative_filtering/databases/ml-latest/ratings.csv", n_ratings, n_users, true, values, row_ind, col_ind, ind_users, row_size, "27");

  cuda_H2D<float>(values, d_values, n_ratings);
  cuda_H2D<int>(row_ind, d_row_ind, n_ratings);
  cuda_H2D<int>(col_ind, d_col_ind, n_ratings);
  cuda_H2D<int>(ind_users, d_ind_users, max_users);
  cuda_H2D<int>(row_size, d_row_size, max_users);
  return 0;
}