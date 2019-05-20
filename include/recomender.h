#ifndef RECOMENDER_H
#define RECOMENDER_H

#include "knn.h"
#include <tuple>
#include <queue>

void k_recomendaciones_propuesta(vector<int>& contador,vector<int>& ids_movies, vector<float>& movies_ratings, float* values, int* row_ind, int* col_ind, int* ind_users, int* row_size,float* item_values, int* item_row_ind, int* item_col_ind, int* ind_items, int* item_row_size, float* d_item_values, int* d_item_row_ind, int* d_item_col_ind, int* d_ind_items, int* d_item_row_size, int n_users, int max_users, int id_user, int measure, int k){
  cout<<endl;
  cout<<"K recomendaciones ordenadas: "<<endl;
  cout<<"--------------------------------"<<endl;
  float* distances;
  bool* b_dists;
  float* dists_users;
  int* pos_users;
  reloj r, r2;
  vector<pair<int, float> > knns;

  r.start();
  distances_one2all(distances, b_dists, values, row_ind, col_ind, ind_users, row_size, item_values, item_row_ind, item_col_ind, ind_items, item_row_size, d_item_values, d_item_row_ind, d_item_col_ind, d_ind_items, d_item_row_size, n_users, max_users, id_user, measure);
  r.stop();
  cout<<"Calculo de distancias: "<<r.time()<<"ms"<<endl;

  r2.start();
  if((measure == PEARSON) || (measure == COSINE))
    knns = knn_greater_cuda(distances, b_dists, max_users, id_user, k);
  else if(measure == EUCLIDEAN || measure == MANHATTAN)
    knns = knn_less_cuda(distances, b_dists, max_users, id_user, k);
  r2.stop();
  cout<<"Calculo de knns: "<<r2.time()<<"ms"<<endl;


  // cout<<"Ids vecinos cercanos - distancias"<<endl;
  // for (size_t i = 0; i < k; i++) {
  //
  //   cout<<pos_users[i] + 1<<"  "<<dists_users[i]<<endl;
  // }
  //

  float* vals_user = float_pointer(values, ind_users, id_user);
  int* ids_movies_user = int_pointer(col_ind, ind_users, id_user);

  // float* vals_user1 = float_pointer(values, ind_users, 279674);
  // int* ids_movies_user1 = int_pointer(col_ind, ind_users, 279674);
  //
  // for (size_t i = 0; i < row_size[279674]; i++) {
  //   cout<<"otro "<<ids_movies_user1[i]<<" -> "<<vals_user1[i]<<endl;
  // }

  map<int, float> map_user;
  // cout<<"vistos: "<<endl;
  for (size_t i = 0; i < row_size[id_user]; i++) {
    map_user[ids_movies_user[i]] = vals_user[i];
    // cout<<"user: "<< ids_movies_user[i]<<" -> "<<vals_user[i]<<endl;
  }



  map<string, pair<float, int>> ord_map;
  priority_queue<tuple<int, float, int>, vector<tuple<int, float, int> >, less<tuple<int, float, int>> > pq;
  map<int, pair<float, int> > movies;
  // float* t_ratings = new float[k]; int* t_ids = new int[k];
  for (size_t i = 0; i < k; i++) {
    float* vals = float_pointer(values, ind_users, knns[i].first);
    int* ids_movies = int_pointer(col_ind, ind_users, knns[i].first);
    for (size_t j = 0; j < row_size[knns[i].first]; j++) {
      auto it = map_user.find(ids_movies[j]);
      if(it == map_user.end()){
        auto pelicula_it = movies.find(ids_movies[j]);
        if(pelicula_it == movies.end()){
          movies[ids_movies[j]] = make_pair(vals[j], 1);

        }
        else{
          pelicula_it->second.first += vals[j];
          pelicula_it->second.second += 1;
        }
      }
    }
  }

  for (auto it = movies.begin(); it != movies.end(); it++) {
    pq.push(make_tuple(it->second.second, it->second.first / it->second.second, it->first));
  }
  int counter = 0;
  while(counter < k && !pq.empty()) {
    tuple<int, float, int> pelicula = pq.top();
    if(get<1>(pelicula) >=3.5){
      ids_movies.push_back(get<2>(pelicula));
      movies_ratings.push_back(get<1>(pelicula));
      contador.push_back(get<0>(pelicula));
      counter++;
    }
    pq.pop();
    // cout<<"id: "<<it->first<<" -> "<<it->second.first / it->second.second<<endl;
  }
  cout<<"--------------------------------"<<endl;
  cout<<endl<<endl;

}


#endif





