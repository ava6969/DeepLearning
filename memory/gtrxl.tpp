namespace sam_dn
{
    GTRXLTemplate
    GTRXL_IMPL_T::GTRXLImpl(GTRXLOption opt) : BaseModuleImpl<GTRXLOption>(opt)
    {
        this->m_BaseModel = {};
        this->m_BaseModel->push_back(StableTransformerXL(
            opt.embedding_size,
            opt.num_layers,
            opt.num_heads,
            opt.d_inner_head,
            opt.d_inner_ff,
            opt.dropout_o,
            opt.dropout_a,
            opt.bg
        ));
        this->register_module("transformer", this->m_BaseModel);
    }

    GTRXLTemplate
    torch::Tensor GTRXL_IMPL_T::forward(torch::Tensor input) {
        auto& [result, _memory] = ((StableTransformerXL*)&*this->m_BaseModel->children()[0])->pass(input, this->memory);
        this->memory = _memory;
        return result;
    }
}