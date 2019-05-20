#include "scripts.h"
#include "structures.h"
#include "cud_defs.h"
#include "distances.h"
#include "knn.h"

// PISTACHE HEADERS
#include <pistache/endpoint.h>
//sudo apt-get install libssl-dev
//https://github.com/oktal/pistache
using namespace Pistache;

//RAPIDJSON HEADERS
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
using namespace rapidjson;
//http://rapidjson.org/index.html

#include <string>
using namespace std;
#include <iostream>

//int n_ratings, n_users, n_movies;
//int n_ratings_20, n_users_20, n_ratings_27, n_users_27, n_ratings_l, n_users_l, n_movies_27;

int n_ratings_27 = 27753444;
int n_users_27 = 283228;
// n_movies_27 = 53889;

int n_ratings_20 = 20000263;
int n_users_20 = 138493;

int n_ratings_l = 49;
int n_users_l = 8;

// n_ratings = n_ratings_20;
// n_users = n_users_20;

int n_ratings = n_ratings_27;
int n_users = n_users_27;
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

struct HelloHandler : public Http::Handler {

  HTTP_PROTOTYPE(HelloHandler)
  void onRequest(const Http::Request& request , Http::ResponseWriter writer) override{

    if (request.resource() == "/" && request.method() == Http::Method::Get) {
        Http::serveFile(writer, "static/index.html");
    }
    else{

        if (request.resource() == "/knn" && request.method() == Http::Method::Post){
                
                //Copia Contenido a json
                int n = request.body().length();
                char json[n + 1]; 
                strcpy(json, request.body().c_str()); 
                            
                cout << "->" <<json << endl;
                Document d;
                d.Parse(json);

                assert(d.IsObject());
                // cout << "Es Documento" << endl;

                // int iduser = 
                // int distancia = 

                vector<pair<int, float> > knns;
                float * distances; 
                bool * b_dists;

                int id_user = d["iduser"].GetInt();
                int measure = d["distancia"].GetInt();
                int k = d["k"].GetInt();
                // float umbral = d["umbral"].GetFloat();

                reloj r;
                r.start();
                distances_one2all(distances, b_dists, values, row_ind, col_ind, ind_users, row_size, item_values, item_row_ind, item_col_ind, ind_items, item_row_size, d_item_values, d_item_row_ind, d_item_col_ind, d_ind_items, d_item_row_size, n_users, max_users, id_user, measure);
                if(measure == PEARSON){
                  knns = knn_greater_cuda(distances, b_dists, max_users, id_user, k);
                }
                else{
                  knns = knn_less_cuda(distances, b_dists, max_users, id_user, k);
                }
                r.stop();
                cout<<"Tiempo total: "<<r.time()<<"ms"<<endl;
                string salida = "{";
                salida += " \"user\": [";
                for (size_t i = 0; i < k; i++) {
                  cout<<knns[i].first<<" "<<knns[i].second<<endl;
                  salida += "{ \"user\": " + to_string(knns[i].first) + ",";
                  salida += "\"distancia\": " + to_string(knns[i].second) + "}" ; 
                  if (i!=k-1) salida += ",";
                }
                salida+= "]";
                salida += ",\"time\":  " +  to_string(r.time()) ; 
                salida += "}";

                
      
                // //KNN PROCEDURE
                // auto start = chrono::steady_clock::now();
                // auto u = g.findUser(iduser); //prueba
                // k_vec k_vecinos_cercanos; //k vecinos
                // // list<pair<int,float> > recomendacion;//
                // map<NodoItem*,pair<float,int>> recomendacion;
                
                // string salida = "{";
                // if(u) {
                //     salida += " \"user\": [";
                //     u->knn(k,distancia,k_vecinos_cercanos);
                    
                //     int c_vecinos = 0 ; 
                //     for(auto & vecino : k_vecinos_cercanos){
                //         salida += "{ \"user\": " + to_string(vecino.second->id) + ",";
                //         salida += "\"distancia\": " + to_string(vecino.first) + "}" ; 
                //         if (c_vecinos < k_vecinos_cercanos.size()-1 ) salida += ",";
                //         c_vecinos ++;
                //     }

                //     c_vecinos = 0;
                //     salida += "],\"recomendacion\": ["; 
                //     u->recomendacion(k_vecinos_cercanos,recomendacion,umbral);
                //     for(auto & rec : recomendacion){
                //         salida += "{\"idItem\":" + to_string(rec.first->id ) + ",";
                //         salida += " \"rating\": " + to_string(rec.second.first/rec.second.second) + ",";
                //         salida += " \"nombre\": \"" + rec.first->nombre + " \" } ";
                //         if (c_vecinos < recomendacion.size() -1 ) salida += ",";
                //         c_vecinos ++;
                //     }
                //     salida+= "]";
                //     //salida += to_string(recomendacion)

                // }
                // else {
                //     cout << "no user" << endl;
                //     salida+=" \"error\": \"No se encontro el usuario\" ";
                // }

                // auto fin = chrono::steady_clock::now();
                // //cout <<"KNN: " <<chrono::duration_cast<chrono::milliseconds>(fin-start).count()<<endl;
                // salida += ",\"time\":  " +  to_string(chrono::duration_cast<chrono::milliseconds>(fin-start).count()) ; 
                // salida += "}";
                // //END KNN procedure 

                cout << "Salida: "<< salida << endl;        
                // string salida("{}");
                writer.send(Http::Code::Ok, salida, MIME(Application, Json));
        }
        if (request.resource() == "/item" && request.method() == Http::Method::Post){
            //Copia Contenido a json
            int n = request.body().length();
            char json[n + 1]; 
            strcpy(json, request.body().c_str()); 
                        
            cout << "->" <<json << endl;
            Document d;
            d.Parse(json);

            assert(d.IsObject());
            cout << "Es Documento" << endl;

            // int iduser = d["iduser"].GetInt();
            // int distancia = d["distancia"].GetInt();
            // int k = d["k"].GetInt();
            // int item_b = d["item"].GetInt();

            // auto start = chrono::steady_clock::now();
            // auto u = g.findUser(iduser); // Encuentra User
            // auto i = g.findItem(item_b); // Encuentra User
            // k_vec k_vecinos_cercanos; //k vecinos

            // string salida = "{";
            //     if(u && i) {
            //         u->knn_restricto(k,distancia,i,k_vecinos_cercanos);
            //         k_vec_rest kvecinosrest;
            //         float rating = u->get_influencias(k_vecinos_cercanos,i,kvecinosrest);
            //         if(rating != -1){

            //             // list < tuple < user,  distancia , influencia , rating,  rating*influencia > 
            //             salida += "\"rating\":" + to_string(rating) + ",";
            //             salida += " \"user\": [";
            //             int c_vecinos = 0 ; 
            //             for(auto & vecino : kvecinosrest){
            //                 salida += "{ \"user\": " + to_string(get<0>(vecino)->id) + ",";
            //                 salida += "\"distancia\": " + to_string(get<1>(vecino)) + "," ; 
            //                 salida += "\"influencia\": " + to_string(get<2>(vecino)) + "," ; 
            //                 salida += "\"rating\": " + to_string(get<3>(vecino)) + "," ; 
            //                 salida += "\"ratingxinfluencia\": " + to_string(get<4>(vecino)) + "}" ; 
            //                 if (c_vecinos < k_vecinos_cercanos.size()-1 ) salida += ",";
            //                 c_vecinos ++;
            //             }

            //             c_vecinos = 0;
            //             salida += "]";
            //         }
            //         else{
            //             salida+=" \"error\": \"El usuario ya vio la pelicula\" ";
            //         } 

            //     }
            //     else {
            //         cout << "no user o Item" << endl;
            //         salida+=" \"error\": \"No se encontro el usuario o Item\" ";
            //     }

            //     auto fin = chrono::steady_clock::now();
            //     //cout <<"KNN: " <<chrono::duration_cast<chrono::milliseconds>(fin-start).count()<<endl;
            //     salida += ",\"time\":  " +  to_string(chrono::duration_cast<chrono::milliseconds>(fin-start).count()) ; 
            //     salida += "}";
            //     //END KNN procedure 

            //     cout << "Salida: "<< salida << endl;     
                string salida = "{}";
                writer.send(Http::Code::Ok, salida, MIME(Application, Json));

        }
    }

  }
};

int main(int argc, char const *argv[]) {
  

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

  read_ML_ratings("dataset/ratings27.csv", n_ratings, n_users, true, values, row_ind, col_ind, ind_users, row_size, "27");
  read_ML_ratings_items("dataset/ratings27.csv", n_ratings, n_users, max_movies, true,  item_values,  item_row_ind,  item_col_ind,  ind_items, item_row_size, "27");

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


  
  
  // float* r1 = float_pointer(values, ind_users, 8);
  // int* c1 = int_pointer(col_ind, ind_users, 8);
  // for (size_t i = 0; i < row_size[8]; i++) {
  //   cout<<c1[i]<<" "<<r1[i]<<endl;
  // }
  
  Pistache::Address addr(Pistache::Ipv4::any(), Pistache::Port(9081));
  auto opts = Pistache::Http::Endpoint::options()
      .threads(1);

  Http::Endpoint server(addr);
  server.init(opts);
  server.setHandler(Http::make_handler<HelloHandler>());
  server.serve();

  server.shutdown();

  return 0;
}
