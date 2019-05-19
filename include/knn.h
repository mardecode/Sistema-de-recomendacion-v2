#ifndef KNN_H
#define KNN_H

void knn_less_std2(float* distances, int*& pos_users, float*& dists_users, int n_users, int k, int user_pos){
  pos_users = new int[k]; dists_users = new float[k];
  vector< pair<float, int> >v;

  for (size_t i = 0; i < n_users; i++) {
    if(i != user_pos)
      v.push_back(make_pair(distances[i], i));
  }
  sort(v.begin(),v.end());
  for (size_t i = 0; i < k; i++) {
    pos_users[i] = v[i].second;
    dists_users[i] = v[i].first;
  }
}

void knn_greater_std2(float* distances, int*& pos_users, float*& dists_users, int n_users, int k, int user_pos){
  pos_users = new int[k]; dists_users = new float[k];
  vector< pair<float, int> >v;

  for (size_t i = 0; i < n_users; i++) {
    if(i != user_pos)
      v.push_back(make_pair(distances[i], i));
  }
  sort(v.begin(),v.end(), compare_greater);
  for (size_t i = 0; i < k; i++) {
    pos_users[i] = v[i].second;
    dists_users[i] = v[i].first;
  }
}


#endif
