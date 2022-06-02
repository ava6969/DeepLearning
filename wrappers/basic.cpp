//
// Created by dewe on 5/27/22.
//

#include "basic.h"

namespace sam_dn{

    SplitAndStackImpl::SplitAndStackImpl(Option opt): ModuleWithSizeInfoImpl(opt),
    splitAxis(opt.split_axis),
    stackAxis(opt.stack_axis),
    splitSize(opt.split_size){
        LOG(WARNING) << "SplitAndStackImpl output size cannot be correctly calculated\n";
    }

}