#include "scripts.h"
#include "structures.h"
#include "cud_defs.h"
#include "distances.h"
#include "knn.h"
#include "writer.h"
#include "recomender.h"
#include <pthread.h>


int map_n_ratings, map_n_users, map_n_movies;
int n_ratings, n_users, n_movies;
int n_ratings_20, n_users_20, n_ratings_27, n_users_27, n_ratings_l, n_users_l, n_movies_27;
int max_users = 300000;
int max_movies = 200000;

map<int, map<int, float>* > map_users;
map<int, map<int, float>* > map_items;

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


// para coseno ajustado
float *maxs, *mins, *averages;
float * d_averages;




void *print_message_function( void *ptr );

void *get_map_users(void* ptr);
void *get_map_items(void* ptr);

void generate_user_arrays(){
  delete(values); delete(row_ind); delete(col_ind); delete(ind_users); delete(row_size);
  values = new float[n_ratings]; //ratings
  row_ind = new int[n_ratings]; //id_users
  col_ind = new int[n_ratings]; //id_items
  ind_users = new int[max_users]; // indice users
  row_size = new int[max_users]; //indice items

  int i = 0;
  auto it = map_users.begin();
  while (it != map_users.end()) {
    auto ite = it->second->begin();
    ind_users[it->first] = i;
    row_size[it->first] = it->second->size();
    while (ite != it->second->end()) {
      values[i] = ite->second;
      row_ind[i] = it->first;
      col_ind[i] = ite->first;
      i++;
      ite++;
    }
    it++;
  }

}

void generate_item_arrays(){
  delete(item_values); delete(item_row_ind); delete(item_col_ind); delete(ind_items); delete(item_row_size);
  item_values = new float[n_ratings]; //ratings
  item_row_ind = new int[n_ratings]; //id_users
  item_col_ind = new int[n_ratings]; //id_items
  ind_items = new int[max_movies]; // indice users
  item_row_size = new int[max_movies]; //indice items

  int i = 0;
  auto it = map_items.begin();
  while (it != map_items.end()) {
    auto ite = it->second->begin();
    ind_items[it->first] = i;
    item_row_size[it->first] = it->second->size();
    while (ite != it->second->end()) {
      item_values[i] = ite->second;
      item_row_ind[i] = it->first;
      item_col_ind[i] = ite->first;
      i++;
      ite++;
    }
    it++;
  }

}



int main(int argc, char const *argv[]) {
  // float* as = new float[4];
  // for (size_t i = 0; i < 4; i++) {
  //   if(as[i] == 0)
  //     cout<<as[i]<<endl;
  // }

  pthread_t thread1, thread2;
  const char *message1 = "Thread 1";
  int  iret1, iret2;





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




  // n_ratings
  // n_of_users("../databases/libro/ratings.csv", n_ratings, n_users, true);
  // cout<<n_ratings<<" "<<n_users<<endl;


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

  d_averages = cuda_array<float>(max_users);





  // read_ML_movies("../databases/ml-20m/movies.csv", movies_names, true);
  // read_ML_ratings("../databases/ml-20m/ratings.csv", n_ratings, n_users, true, values, row_ind, col_ind, ind_users, row_size, "27");

  // read_ML_movies("../../collaborative_filtering/databases/ml-latest/movies.csv", movies_names, true);
  // read_ML_ratings("../collaborative_filtering/databases/ml-latest/ratings.csv", n_ratings, n_users, true, values, row_ind, col_ind, ind_users, row_size, "27");

  reloj a;
  a.start();
  read_ML_ratings("dataset/ratings.csv", n_ratings, n_users, true, values, row_ind, col_ind, ind_users, row_size, "27");
  read_ML_ratings_items("dataset/ratings.csv", n_ratings, n_users, max_movies, true,  item_values,  item_row_ind,  item_col_ind,  ind_items, item_row_size, "27");
  a.stop();
  cout<<"Tiempo de carga de bd: "<<a.time()<<"ms"<<endl;
  // read_ML_ratings("../collaborative_filtering/databases/libro/ratings.csv", n_ratings, n_users, true, values, row_ind, col_ind, ind_users, row_size, "l");
  // read_ML_ratings_items("../collaborative_filtering/databases/libro/ratings.csv", n_ratings, n_users, max_movies, true,  item_values,  item_row_ind,  item_col_ind,  ind_items, item_row_size, "l");


  average_per_user(values, ind_users, row_size, maxs, mins, averages, max_users);
  cuda_H2D<float>(averages, d_averages, max_users);






  iret1 = pthread_create( &thread1, NULL, get_map_users, (void*) message1);
  if(iret1){
    fprintf(stderr,"Error - pthread_create() return code: %d\n",iret1);
    exit(EXIT_FAILURE);
  }
  printf("pthread_create() for thread 1 returns: %d\n",iret1);

  iret2 = pthread_create( &thread2, NULL, get_map_items, (void*) message1);
  if(iret2){
    fprintf(stderr,"Error - pthread_create() return code: %d\n",iret2);
    exit(EXIT_FAILURE);
  }
  printf("pthread_create() for thread 1 returns: %d\n",iret2);









  // pthread_join( thread1, NULL);



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


  // float *distances;
  // bool * b_dists;
  // vector<pair<int, float> > knns;
  // int id_user = 16006;
  // // int id_user = 30503;
  // // int id_user = 283228;
  // int measure = MANHATTAN;
  // int k = 10;
  // reloj r;
  // r.start();
  // cout<<"asda"<<endl;
  // distances_one2all(distances, b_dists, values, row_ind, col_ind, ind_users, row_size, item_values, item_row_ind, item_col_ind, ind_items, item_row_size, d_item_values, d_item_row_ind, d_item_col_ind, d_ind_items, d_item_row_size, n_users, max_users, id_user, measure);
  // if(measure == PEARSON){
  //   // knns = knn_greater_map(distances, b_dists, max_users, id_user, k);
  //   knns = knn_greater_cuda(distances, b_dists, max_users, id_user, k);
  // }
  // else{
  //   knns = knn_less_cuda(distances, b_dists, max_users, id_user, k);
  //   // knns = knn_less_map(distances, b_dists, max_users, id_user, k);
  // }
  // r.stop();
  // cout<<"Tiempo total: "<<r.time()<<"ms"<<endl;
  // for (size_t i = 0; i < k; i++) {
  //   cout<<knns[i].first<<" "<<knns[i].second<<endl;
  // }
  // vector<int> contador;
  // vector<int> ids_movies;
  // vector<float> movies_ratings;
  //
  // k_recomendaciones_propuesta(contador, ids_movies, movies_ratings, values, row_ind, col_ind, ind_users, row_size, item_values, item_row_ind, item_col_ind, ind_items, item_row_size, d_item_values, d_item_row_ind, d_item_col_ind, d_ind_items, d_item_row_size, n_users, max_users, id_user, measure, k);
  //
  // for (size_t i = 0; i < k; i++) {
  //   cout<<ids_movies[i]<<" "<<movies_ratings[i]<<" "<<contador[i]<<endl;
  // }


  
  cout << "Comienza Escritura" << endl;
  create_matrix_desviaciones(n_ratings,max_movies,d_item_values,d_item_row_ind,d_item_col_ind,d_ind_items,d_item_row_size,row_size);



  // cout<<endl<<endl<<endl;

  // // float* r1 = float_pointer(values, ind_users, 1);
  // // int* c1 = int_pointer(col_ind, ind_users, 1);
  // // for (size_t i = 0; i < row_size[1]; i++) {
  // //   cout<<c1[i]<<" -> "<<r1[i]<<endl;
  // // }

  // float* r1 = float_pointer(item_values, ind_items, 1);
  // int* c1 = int_pointer(item_col_ind, ind_items, 1);
  // for (size_t i = 0; i < 10; i++) {
  //   cout<<c1[i]<<" -> "<<r1[i]<<endl;
  // }


  // cout<<"------------------"<<endl;

  // pthread_join( thread1, NULL);
  // pthread_join( thread2, NULL);
  // cout<<"Size map: "<<map_users.size()<<" "<<map_items.size()<<endl;

  // // auto it = map_users.find(1);
  // // if(it == map_users.end())
  // //   cout <<"not found!!!"<<endl;
  // // else
  // // for (auto ite = it->second->begin(); ite != it->second->end(); ite++) {
  // //   cout<<ite->first<<" -> "<<ite->second<<endl;
  // // }

  // auto it = map_items.find(1);
  // int k1 = 0;
  // if(it == map_items.end())
  //   cout <<"not found!!!"<<endl;
  // else
  // for (auto ite = it->second->begin(); (k1 < 10) && ite != it->second->end(); ite++) {
  //   cout<<ite->first<<" -> "<<ite->second<<endl;
  //   k1++;
  // }

  // cout<<"------------------"<<endl;


  // reloj p;
  // p.start();
  // generate_user_arrays();
  // generate_item_arrays();
  // p.stop();
  // cout<<"Tiempo de generacion de user arrays"<<p.time()<<"ms"<<endl;



  // r1 = float_pointer(item_values, ind_items, 1);
  // c1 = int_pointer(item_col_ind, ind_items, 1);
  // for (size_t i = 0; i < 10; i++) {
  //   cout<<c1[i]<<" -> "<<r1[i]<<endl;
  // }




  return 0;
}



/// FIN MAIN // 
void *get_map_users(void* ptr){
  cout<<"Creando map users"<<endl;
  map_n_ratings = n_ratings;
  map_n_users = n_users;
  map_n_movies = n_movies;
  for (size_t i = 0; i < max_users; i++) {
    if(i % 10000 == 0){
      // cout<<i<<endl;
    }
    if(row_size[i] != 0){
      if (i == 1)
        cout<<"si esta 1"<<endl;
      float* r1 = float_pointer(values, ind_users, i);
      int* c1 = int_pointer(col_ind, ind_users, i);
      for (size_t j = 0; j < row_size[i]; j++) {
        auto it = map_users.find(i);
        if(it == map_users.end()){
          map<int, float> * mapa = new map<int, float>();
          map_users[i] = mapa;
        }
        (*(map_users[i]))[c1[j]] = r1[j];
      }
    }
  }
  cout<<"Fin map users"<<endl;
}

void *get_map_items(void* ptr){
  map_n_ratings = n_ratings;
  map_n_users = n_users;
  map_n_movies = n_movies;
  cout<<"Creando map items"<<endl;
  for (size_t i = 0; i < max_users; i++) {
    if(i % 10000 == 0){
      // cout<<i<<endl;
    }
    if(row_size[i] != 0){
      float* r1 = float_pointer(values, ind_users, i);
      int* c1 = int_pointer(col_ind, ind_users, i);
      for (size_t j = 0; j < row_size[i]; j++) {
        auto it = map_items.find(c1[j]);
        if(it == map_items.end()){
          map<int, float> * mapa = new map<int, float>();
          map_items[c1[j]] = mapa;
        }
        (*(map_items[c1[j]]))[i] = r1[j];
      }
    }
  }
  cout<<"Fin map items"<<endl;
}
