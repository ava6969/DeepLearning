namespace sam_dn {
    RecurrentNetTemplate
    GTRXL_IMPL_T::GTRXLImpl(GTRXLOption opt):
        BaseModuleImpl<GTRXLOption, MemoryType, StateType, batchFirst>(opt),
        device(opt.device)
    {
        // TODO: Fill this in
    }

    GTRXLTemplate
    void GTRXL_IMPL_T::initialStateWithBatchSizeAndDevice(int batch_size, torch::Device _device) noexcept {
        // TODO: Fill this in
    }

    GTRXLTemplate
    void GTRXL_IMPL_T::initialStateWithBatchSize(int batch_size) noexcept {
        // TODO: Fill this in
    }

    GTRXLTemplate
    void GTRXL_IMPL_T::to(torch::Device _device, bool non_blocking) {
        // TODO: Fill this in
    }

    // GTRXLTemplate
    // torch::Tensor GTRXL_IMPL_T::pass(torch::Tensor const& _mask, torch::Tensor const& x, TensorDict const& hxs){
    //     StateType maskedState;
    //     torch::Tensor out;

    //     if constexpr(type == 'l'){

    //         auto&& hx = hxs.at(hidden_state_id.first).view({num_layers, -1, hidden_size});
    //         auto&& cx = hxs.at(hidden_state_id.second).view({num_layers, -1, hidden_size});
    //         maskedState = reset_states ? std::make_tuple( hx * _mask, cx * _mask) : std::make_tuple( hx, cx);
    //     } else{
    //         auto&& hx = hxs.at(hidden_state_id).view({num_layers, -1, hidden_size});
    //         maskedState = reset_states ? hx * _mask : hx;
    //     }
    //     std::tie(out, this->m_States) = this->m_BaseModel(x, maskedState);
    //     return out;
    // }

    GTRXLTemplate
    TensorDict* GTRXL_IMPL_T::forwardDict(TensorDict *x) noexcept {
        auto const &input = x->at(this->m_Input);
        return x;
    }
}