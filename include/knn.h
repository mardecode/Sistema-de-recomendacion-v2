#ifndef KNN_H
#define KNN_H

#include <set>
#include "distances.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>



bool compare_greater(const pair<float, int>&i, const pair<float, int>&j)
{
  if(i.first == j.first){
    return i.second < j.second;
  }
  return i.first > j.first;
}



vector<pair<int, float> > knn_less_map(float* distances, bool* b_dists, int max_users, int user_id, int k){
  vector<pair<int, float> > knns(k);

  // set<pair<float, int>, decltype(&compare_greater)> mapa(&compare_greater);
  set<pair<float, int>, less<pair<float, int> > > mapa;
  for (size_t i = 0; i < max_users; i++) {
    if(b_dists[i] && i != user_id){
      mapa.insert(make_pair(distances[i], i));
      // mapa[distances[i]] = dis
      // cout<<"i: "<<i<<" -> "<<distances[i]<<endl;

    }
  }


  auto it = mapa.begin();
  int i = 0;
  for (; it != mapa.end() && (i < k); it++) {
    knns[i] = make_pair(it->second, it->first);
    i++;
    cout<<it->second<<" "<<it->first<<endl;
  }
  return knns;
}

vector<pair<int, float> > knn_greater_map(float* distances, bool* b_dists, int max_users, int user_id, int k){
  vector<pair<int, float> > knns(k);

  set<pair<float, int>, decltype(&compare_greater)> mapa(&compare_greater);
  // set<pair<float, int>, less<pair<float, int> > > mapa;
  for (size_t i = 0; i < max_users; i++) {
    if(b_dists[i] && i != user_id){
      mapa.insert(make_pair(distances[i], i));
      // mapa[distances[i]] = dis
      // cout<<"i: "<<i<<" -> "<<distances[i]<<endl;
    }
  }


  auto it = mapa.begin();
  int i = 0;
  for (; it != mapa.end() && (i < k); it++) {
    knns[i] = make_pair(it->second, it->first);
    i++;
    // cout<<it->second<<" "<<it->first<<endl;
  }
  return knns;
}

vector<pair<int, float> > knn_greater_cuda(float* distances, bool* b_dists, int max_users, int user_id, int k){
  // float* t_distances; int* t_pos_users;
  // t_distances = new float[n_users]; t_pos_users = new int[n_users];
  // ids_users = new int[k]; dists_users = new float[k];
  vector<pair<int, float> > knns(k);
  int n = 0;
  for (size_t i = 0; i < max_users; i++) {
    if(b_dists[i])
      n++;
  }


  thrust::host_vector<float> t_distances(n);
  thrust::host_vector<int> t_ids_users(n);
  thrust::device_vector<float> d_t_distances;
  thrust::device_vector<int> d_t_ids_users;

  int c = 0;
  for (size_t i = 0; i < max_users; i++) {
    if (b_dists[i] && (i != user_id)) {
      t_distances[c] = distances[i];
      t_ids_users[c] = i;
      c++;
    }
  }

  d_t_distances = t_distances;
  d_t_ids_users = t_ids_users;


  reloj r;
  r.start();
  thrust::sort_by_key(d_t_distances.begin(), d_t_distances.begin() + n, d_t_ids_users.begin());

  thrust::copy(d_t_distances.begin(), d_t_distances.end(), t_distances.begin());
  thrust::copy(d_t_ids_users.begin(), d_t_ids_users.end(), t_ids_users.begin());
  r.stop();
  cout<<"Cuda knns: "<<r.time()<<"ms"<<endl;

  for (size_t i = 0; i < k; i++) {
    knns[i] = make_pair(t_ids_users[n - i - 1], t_distances[n - i - 1]);
    // dists_users[i] = t_distances[n - i - 1];
    // ids_users[i] = t_ids_users[n - i - 1];
  }
  return knns;
}



#endif
