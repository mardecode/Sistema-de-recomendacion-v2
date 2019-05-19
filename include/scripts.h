#ifndef SCRIPTS_H
#define SCRIPTS_H
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <map>
#include <fstream>

#include "structures.h"

using namespace std;

vector<string> split(const string& s, char delimiter){
   vector<string> tokens;
   string token;
   istringstream tokenStream(s);
   while (getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}


class reloj{
public:
  void start(){c_start = clock();}
  void stop(){c_stop = clock();}
  double time(){return (double)(c_stop - c_start) * 1000.0 / CLOCKS_PER_SEC;}
  clock_t c_start, c_stop;
  double elapsed;
  reloj(){

  }
  ~reloj(){

  }

};

bool fexists(const std::string& filename) {
  std::ifstream ifile(filename.c_str());
  return (bool)ifile;
}

template<class T>
void write_array(T*& arr, int size, string nombre){
  ofstream out(nombre, ios::out | ios::binary);
  if(!out) {
    cout << "Cannot open file.";
    return;
  }
  cout<<"Writing: "<<sizeof(T) * size<<" bytes"<<endl;
  out.write(&(((char*)arr)[0]), sizeof(T) * size);
  out.close();
}

template<class T>
void read_array(T*& arr, int size, string nombre){
  ifstream in(nombre, ios::in | ios::binary);
  in.read(&(((char*)arr)[0]), sizeof(T) * size);
  in.close();
}

void n_of_users(string path, int& n_ratings, int& n_users, bool header){
  ifstream infile(path);
  string line;
  if(header) getline(infile, line);
  vector<string> tokens;

  int id_user, curr_id_user, users_counter, ratings_counter;

  ratings_counter = 0;  users_counter = 0;  id_user = -1;

  while (getline(infile, line)) {
    if(ratings_counter % 1000000 == 0){
      cout<<ratings_counter<<endl;
    }
    tokens = split(line, ',');
    curr_id_user = atoi(tokens[0].c_str());
    if(id_user < curr_id_user){
      users_counter++;
      id_user = curr_id_user;
    }
    ratings_counter++;
  }
  n_ratings = ratings_counter;
  n_users = users_counter;
  cout<<n_ratings<<" "<<n_users<<endl;
}

void read_ML_ratings(string path, int n_ratings, int n_users, bool header, float*& values, int*& row_ind, int*& col_ind, int*& ind_users, int*& row_size, string version){
  string path_b = "binarios/";
  int max_users = 300000;
  values = new float[n_ratings];
  row_ind = new int[n_ratings];
  col_ind = new int[n_ratings];
  ind_users = new int[max_users];
  row_size = new int[max_users];

  if(fexists(path_b + "values_" + version) && fexists(path_b + "row_ind_" + version) && fexists(path_b + "col_ind_" + version) && fexists(path_b + "ind_users_" + version) && fexists(path_b + "row_size_" + version)){
    cout<<"Reading values"<<endl;
    read_array<float>(values, n_ratings, path_b + "values_" + version);
    cout<<"Reading row_ind"<<endl;
    read_array<int>(row_ind, n_ratings, path_b + "row_ind_" + version);
    cout<<"Reading col_ind"<<endl;
    read_array<int>(col_ind, n_ratings, path_b + "col_ind_" + version);
    cout<<"Reading ind_users"<<endl;
    read_array<int>(ind_users, max_users, path_b + "ind_users_" + version);
    cout<<"Reading row_size"<<endl;
    read_array<int>(row_size, max_users, path_b + "row_size_" + version);
  }
  else{
    ifstream infile(path);
    string line;
    if(header) getline(infile, line);
    vector<string> tokens;

    int id_user, curr_id_user, curr_id_item, users_counter, ratings_counter, n_r;
    float curr_rating;
    ratings_counter = 0;  users_counter = 0;  id_user = -1; n_r = 0;

    while (getline(infile, line)) {
      if(ratings_counter % 1000000 == 0)
      cout<<ratings_counter<<endl;
      tokens = split(line, ',');
      curr_id_user = atoi(tokens[0].c_str());
      curr_id_item = atoi(tokens[1].c_str());
      curr_rating = atof(tokens[2].c_str());
      if(id_user < curr_id_user){
        if(id_user != -1)
          row_size[id_user] = n_r;
        n_r = 0;
        ind_users[curr_id_user] = ratings_counter;
        id_user = curr_id_user;
        users_counter++;
      }

      values[ratings_counter] = curr_rating;
      row_ind[ratings_counter] = curr_id_user;
      col_ind[ratings_counter] = curr_id_item;
      n_r ++;
      ratings_counter++;
    }
    cout<<ratings_counter<<" - "<<users_counter<<endl;
    // row_size[n_users - 1] = n_r;
    row_size[curr_id_user] = n_r;

    cout<<"Writing values"<<endl;
    write_array<float>(values, n_ratings, path_b + "values_" + version);
    cout<<"Writing row_ind"<<endl;
    write_array<int>(row_ind, n_ratings, path_b + "row_ind_" + version);
    cout<<"Writing col_ind"<<endl;
    write_array<int>(col_ind, n_ratings, path_b + "col_ind_" + version);
    cout<<"Writing ind_users"<<endl;
    write_array<int>(ind_users, max_users, path_b + "ind_users_" + version);
    cout<<"Writing row_size"<<endl;
    write_array<int>(row_size, max_users, path_b + "row_size_" + version);

  }

}


void read_ML_ratings_items(string path, int n_ratings, int n_users, int n_movies, int n_ids_movies, bool header, float*& item_values, int*& item_row_ind, int*& item_col_ind, int*& ind_items, int*& item_row_size, string version, int*& pos_movies){
  string path_b = "binarios/";

  item_values = new float[n_ratings];
  item_row_ind = new int[n_ratings];
  item_col_ind = new int[n_ratings];
  ind_items = new int[n_movies];
  item_row_size = new int[n_movies];
  pos_movies = new int[n_ids_movies];

  if(fexists(path_b + "item_values_" + version) && fexists(path_b + "item_row_ind_" + version) && fexists(path_b + "item_col_ind_" + version) && fexists(path_b + "ind_item_" + version) && fexists(path_b + "item_row_size_" + version) && fexists(path_b + "pos_movies_" + version)){
    cout<<"Reading item_values"<<endl;
    read_array<float>(item_values, n_ratings, path_b + "item_values_" + version);
    cout<<"Reading item_row_ind"<<endl;
    read_array<int>(item_row_ind, n_ratings, path_b + "item_row_ind_" + version);
    cout<<"Reading item_col_ind"<<endl;
    read_array<int>(item_col_ind, n_ratings, path_b + "item_col_ind_" + version);
    cout<<"Reading ind_items"<<endl;
    read_array<int>(ind_items, n_movies, path_b + "ind_item_" + version);
    cout<<"Reading item_row_size"<<endl;
    read_array<int>(item_row_size, n_movies, path_b + "item_row_size_" + version);
    cout<<"Reading pos_movies"<<endl;
    read_array<int>(pos_movies, n_ids_movies, path_b + "pos_movies_" + version);
  }else{
    ifstream infile(path);
    string line;
    int n_movies;
    if(header) getline(infile, line);
    vector<string> tokens;

    map<int, map<int, float>* > map_movies_items;
    int cont = 0;
    while (getline(infile, line)) {
      if(cont % 1000000 == 0)
        cout<<cont<<endl;
      tokens = split(line, ',');
      auto it = map_movies_items.find(atoi(tokens[1].c_str()));
      if(it == map_movies_items.end()){
        map<int, float> * mapa = new map<int, float>();
        map_movies_items[atoi(tokens[1].c_str())] = mapa;
      }
      (*(map_movies_items[atoi(tokens[1].c_str())]))[atoi(tokens[0].c_str())] = atof(tokens[2].c_str());
      cont++;
    }

    n_movies = map_movies_items.size();
    cout<<"Numero de peliculas: "<<n_movies<<endl;

    int i = 0;
    int j = 0;
    auto it = map_movies_items.begin();
    while (it != map_movies_items.end()) {
      pos_movies[it->first] = j;
      auto ite = it->second->begin();
      ind_items[j] = i;
      item_row_size[j] = it->second->size();
      while (ite != it->second->end()) {
        item_values[i] = ite->second;
        item_row_ind[i] = it->first;
        item_col_ind[i] = ite->first;
        i++;
        ite++;
      }
      j++;
      it++;
    }
    cout<<"Writing item_values"<<endl;
    write_array<float>(item_values, n_ratings, path_b + "item_values_" + version);
    cout<<"Writing item_row_ind"<<endl;
    write_array<int>(item_row_ind, n_ratings, path_b + "item_row_ind_" + version);
    cout<<"Writing item_col_ind"<<endl;
    write_array<int>(item_col_ind, n_ratings, path_b + "item_col_ind_" + version);
    cout<<"Writing ind_items"<<endl;
    write_array<int>(ind_items, n_movies, path_b + "ind_item_" + version);
    cout<<"Writing item_row_size"<<endl;
    write_array<int>(item_row_size, n_movies, path_b + "item_row_size_" + version);
    cout<<"Writing pos_movies"<<endl;
    write_array<int>(pos_movies, n_ids_movies, path_b + "pos_movies_" + version);


  }

}




void read_ML_movies(string path, map<int, string>& movie_names, bool header){
  ifstream infile(path);
  string line;
  if(header) getline(infile, line);
  vector<string> tokens;
  int movies_counter = 0;
  int id;
  while (getline(infile, line) ) {
    // if(movies_counter % 10000 == 0)
      // cout<<movies_counter<<"  "<<line<<endl;
    tokens = split(line, ',');
    id = atoi(tokens[0].c_str());
    movie_names[id] = tokens[1];
    // cout<<tokens[0]<<endl;
    movies_counter++;
  }
  cout<<"Numero de peliculas: "<<movies_counter<<endl;
}

#endif
