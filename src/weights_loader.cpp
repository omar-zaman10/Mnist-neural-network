#include <iostream>
#include <fstream>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include <rapidjson/istreamwrapper.h>
#include <cstdio>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include <unordered_map>
#include "xtensor-blas/xblas.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor-blas/xlapack.hpp"


using namespace std;
using namespace rapidjson;

unordered_map<int, xt::xarray<double> > biases_loader(){
    ifstream ifs { "../data/biases.json" };
    if ( ifs.is_open() )
    {
        cout <<"Opened json biases data file for reading" << endl;
    
    }
    else{
        
        cerr << "Could not open file for reading!\n";
    }

    IStreamWrapper isw { ifs };

    Document doc {};
    doc.ParseStream( isw );
    
    cout << "document prepared: "<< endl;


    unordered_map<int, xt::xarray<double> > hashmap;


    xt::xarray<double> biases; 


    for (Value::ConstMemberIterator itr = doc.MemberBegin();
        itr != doc.MemberEnd(); ++itr)
    {
        biases = xt::zeros<double>({itr->value.Size()});
        
        
        for (int i = 0; i < itr->value.Size(); i++){ 
        biases[i] = itr->value[i].GetDouble() ;}        
        
        hashmap[stoi(itr->name.GetString())] = biases;

    }

    
    return hashmap;

}



unordered_map<int, xt::xarray<double>>  weights_loader(){
    ifstream ifs { "../data/weights.json" };
    if ( ifs.is_open() )
    {
        cout <<"Opened json weights data file for reading" << endl;
    
    }
    else{
        
        cerr << "Could not open file for reading!\n";
    }

    IStreamWrapper isw { ifs };

    Document doc {};
    doc.ParseStream( isw );
    
    cout << "document prepared: "<< endl;


    unordered_map<int, xt::xarray<double> > hashmap;

    xt::xarray<double> weights; 

    for (Value::ConstMemberIterator itr = doc.MemberBegin();
        itr != doc.MemberEnd(); ++itr)
    {
        weights = xt::zeros<double>({itr->value.Size(),itr->value[0].Size()});

        

        for (int i = 0; i < itr->value.Size(); i++){ 
            for(int j = 0; j < itr->value[0].Size(); j++){


                weights[i*itr->value[0].Size() +j] = itr->value[i][j].GetDouble();

            }

        }        
        
        hashmap[stoi(itr->name.GetString())] = weights;
        
    }

    return hashmap;

}

