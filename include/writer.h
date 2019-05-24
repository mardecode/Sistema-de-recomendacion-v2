#pragma once

#include <fstream>
#include <iostream>
#include "distances.h"
#include "scripts.h"

using namespace std;

void create_matrix_desviaciones(int n_ratings,int max_movies,float * d_item_values,int * d_item_row_ind,int * d_item_col_ind, int * d_ind_items,int * d_item_row_size,int * row_size){
    float* desviaciones;
    int* cardinalidad;
    bool* b_dists;
    int id_movie = 1;
    
    float* cosenos;

    reloj j2;
    j2.start();

    // desviaciones_one2all( desviaciones, cardinalidad, b_dists, item_values, item_row_ind, item_col_ind, ind_items, item_row_size, values, row_ind, col_ind, ind_users, row_size, d_values, d_row_ind, d_col_ind, d_ind_users, d_row_size, n_movies, max_movies, id_movie);
    // coseno_ajustado_one2all(cosenos, b_dists, item_values, item_row_ind, item_col_ind, ind_items, item_row_size, values, row_ind, col_ind, ind_users, row_size, d_values, d_row_ind, d_col_ind, d_ind_users, d_row_size, n_movies, max_movies, id_movie, maxs, mins, averages);
    // adjusted_cosine_one2all(cosenos, id_movie, n_ratings,max_movies, d_item_values, d_item_row_ind, d_item_col_ind, d_ind_items, d_item_row_size, d_averages);
    // for (size_t i = 0; i < 10; i++) {
    //   if(row_size[i] != 0)
    //     cout<<cosenos[i]<<endl;
    // }
    // (float*& distances, int*& cardinalidad, int id_movie, int n_ratings, int max_movies,float*& d_item_values, int*& d_item_row_ind, int*& d_item_col_ind, int*& d_ind_items, int*& d_item_row_size)
    
    // max_movies

    int filas = 4; 
    auto myfile = std::fstream("file.binary", std::ios::out | std::ios::binary);
    for (int i = 0; i <= 1; i++)
    {
        float * fila;
        desviacion_one2all(desviaciones, cardinalidad, id_movie, n_ratings, max_movies, d_item_values, d_item_row_ind, d_item_col_ind, d_ind_items, d_item_row_size);
        // cout << fila << endl;
        myfile.write((char*) fila , sizeof(float)*(i+1) );
    }
    myfile.close();

    for (size_t i = 0; i < 10; i++) {
    if(row_size[i] != 0)
        cout<<desviaciones[i]<<"  ->  "<<cardinalidad[i]<<endl;
    }

    j2.stop();
    cout<<"tiempo de uno a todos items: "<<j2.time()<<"ms"<<endl;

}

