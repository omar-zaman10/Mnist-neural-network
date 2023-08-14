#include <iostream>
#include <fstream>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include <rapidjson/istreamwrapper.h>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtl/xsequence.hpp"
#include <cstdio>
#include "xtensor/xview.hpp"
#include <tuple>



using namespace std;
using namespace rapidjson;


tuple<xt::xarray<double>,xt::xarray<double>> JsonLoader(){
    
    ifstream ifs { "../data/training_data.json" };
    if ( ifs.is_open() )
    {
        cout <<"Opened json training_data file for reading" << endl;
    
    }
    else{
        
        cerr << "Could not open file for reading!\n";
    }

    IStreamWrapper isw { ifs };

    Document doc {};
    doc.ParseStream( isw );
    
    cout << "document prepared: "<< endl;

    

    const Value& input_data = doc["input_data"];

    const Value& labels = doc["labels"];

    xt::xarray<double> x = xt::zeros<double>({input_data.Size(),input_data[0].Size()});
    xt::xarray<double> y = xt::zeros<double>({labels.Size()});

    for (int i = 0; i < labels.Size(); i++){
        y[i] =  labels[i].GetDouble();
    }

    

    for(int i=0; i < input_data.Size(); i++){

        for(int j=0; j< input_data[0].Size(); j++){

            x[i*input_data[0].Size() + j] = input_data[i][j].GetDouble();

        }

    }


    return {x, y};
        

}
