#include <iostream>
#include <fstream>
using namespace std;

float arreglo[4][4] =  { 
{ 1,2,3,4  }, 
{ 5,6,7,8 },
{ 9,10,11,12},
{ 13,14,15,16}} ;

float * get_array(int i){
    return arreglo[i];
}

void write_row(){
    int filas = 4; 
    auto myfile = std::fstream("file.binary", std::ios::out | std::ios::binary);
    for (int i = 0; i <= 1; i++)
    {
        float * fila = get_array(i);
        // cout << fila << endl;
        myfile.write((char*) fila , sizeof(float)*(i+1) );
    }
    myfile.close();
}

void read(int i, int j){
    char datos[sizeof(float)] ;
    if (i<j) swap ( i, j);
    ifstream myFile ("file.binary", ios::in | ios::binary);
    int acceso = ( i*(i+1)/2 + j )*sizeof(float);
    //cout << "acc" << acceso << endl;
    cout << sizeof(float) << endl;
    myFile.seekg (acceso, myFile.beg);
    
    myFile.read(datos,sizeof(float));

    float * f = (float * )datos;
    cout << float( f[0] ) << endl;
}

int main(){

    write_row();
    read(0,2);

    // for (size_t i = 0; i < 4; i++)
    // {
    //     float * fila = get_array(i);
    //     for (size_t j = 0; j < 4; j++)
    //     {
    //         cout << fila[j] << ",";
    //     }   
        
    // }
    


}