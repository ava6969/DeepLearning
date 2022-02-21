//
// Created by dewe on 9/18/21.
//

#pragma once
#include "exception"
#include "sstream"

namespace sam_dn{
    struct SamException : std::exception{

        std::string _msg;
        std::stringstream ss;
        virtual std::string msg() = 0;
    };

    struct InvalidWeightInitializer : SamException{
        std::string wrongType;

        explicit InvalidWeightInitializer(std::string _type): wrongType(std::move(_type)){}

        std::string msg() override{
            ss << "InvalidWeightInitializer: " << wrongType <<
               ", use either ortho, xavier_uniform or xavier_normal";
            return ss.str();
        }
    };

    struct InvalidDimSize : SamException{
        long size;
        long boundary;

        InvalidDimSize(long _size, long boundary):
                size(_size),
                boundary(boundary){

        }

        std::string msg() override{
            ss << "InvalidDimSize: " << size <<
               " < " << boundary ;
            return ss.str();
        }
    };
}