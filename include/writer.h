#pragma once

#include <fstream>
#include <iostream>
#include "distances.h"
#include "scripts.h"

using namespace std;

void create_matrix_desviaciones(int n_ratings,int max_movies,float * d_item_values,int * d_item_row_ind,int * d_item_col_ind, int * d_ind_items,int * d_item_row_size,int * item_row_size){
    
    // desviaciones_one2all( desviaciones, cardinalidad, b_dists, item_values, item_row_ind, item_col_ind, ind_items, item_row_size, values, row_ind, col_ind, ind_users, row_size, d_values, d_row_ind, d_col_ind, d_ind_users, d_row_size, n_movies, max_movies, id_movie);
    // coseno_ajustado_one2all(cosenos, b_dists, item_values, item_row_ind, item_col_ind, ind_items, item_row_size, values, row_ind, col_ind, ind_users, row_size, d_values, d_row_ind, d_col_ind, d_ind_users, d_row_size, n_movies, max_movies, id_movie, maxs, mins, averages);
    // adjusted_cosine_one2all(cosenos, id_movie, n_ratings,max_movies, d_item_values, d_item_row_ind, d_item_col_ind, d_ind_items, d_item_row_size, d_averages);
    // for (size_t i = 0; i < 10; i++) {
    //   if(row_size[i] != 0)
    //     cout<<cosenos[i]<<endl;
    // }
    // (float*& distances, int*& cardinalidad, int id_movie, int n_ratings, int max_movies,float*& d_item_values, int*& d_item_row_ind, int*& d_item_col_ind, int*& d_ind_items, int*& d_item_row_size)
        
    auto myfile = std::fstream("file.binary", std::ios::out | std::ios::binary);
    int j = 1;
    
    reloj r;
    r.start();
    float * filas_archivo = new float[max_movies];
    int contar = 0;
    for (size_t i = 0; i < max_movies; i++)
    {
        filas_archivo[i] = -1;
        // cout << i<<" " << item_row_size[i] << endl;
        // if ( item_row_size[i] != 0 ) contar ++;
    }
    // cout << " Contados " << contar << endl;
    int contador_filas= 0;

    
    
    // return;
    for (int i = 0; i <= max_movies ; i++)
    // for (int i = max_movies-1; i>=0   ; i--)
    {
        if ( i % 10 == 0) {
            r.stop();
            cout<< i/10 <<" Mil filas escritas en: "<<r.time()<<"ms"<<endl;
            r.start();
            
        }

        float * desviaciones;
        int * cardinalidad;
        int id_movie = i;
        if (item_row_size[i] != 0  ){
            desviacion_one2all(desviaciones, cardinalidad, id_movie, n_ratings, max_movies, d_item_values, d_item_row_ind, d_item_col_ind, d_ind_items, d_item_row_size);
            
            float * desv_temp = new float[max_movies];
            int countk = 0;
            for (size_t k = 0; k < max_movies; k++){
                if(item_row_size[k] != 0 ) { desv_temp[countk] = desviaciones[k] ; countk++ ; }
            }
            // cout << countk << endl;
            // cout << "empieza a escribir" << endl;
            myfile.write((char*) desv_temp , sizeof(float)*j );
            filas_archivo[i] = j-1;
            
            j++;
            // cout << "aqui" << endl;
            delete(desviaciones);
            delete(cardinalidad);
            delete(desv_temp);
        }
        
    }
    myfile.close();

    auto myfile2 = std::fstream("file2.binary", std::ios::out | std::ios::binary);
    myfile2.write((char*) filas_archivo , sizeof(float)*max_movies );
    myfile2.close();



    // for (size_t i = 0; i < 10; i++) {
    // if(row_size[i] != 0)
    //     cout<<desviaciones[i]<<"  ->  "<<cardinalidad[i]<<endl;
    // }

    

}