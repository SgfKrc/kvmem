#pragma once

#include "llama-memory-hybrid.h"
#include "llama-memory-kvmem.h"

#include <memory>

// llama_memory_hybrid whose attention half is a KVMem slot-pool and whose
// recurrent half is stock llama_memory_recurrent.
//
// Graph still static_cast's to llama_memory_hybrid_context. GDN sees every
// token. Query-replay holes must not roll back GDN. MTP verify reject of a
// short suffix (≤ n_rs_seq) uses GPU snapshot planes.
class llama_memory_kvmem_hybrid : public llama_memory_hybrid {
public:
    llama_memory_kvmem_hybrid(
            const llama_model & model,
            const llama_memory_params & params,
            const llama_cparams & cparams);

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;

    // The inherited implementations read mem_attn, which is the *native* kv cache
    // this class borrows, not the KVMem slot-pool wrapping it. Under KVMem the
    // native rows are only the resident working set, so the base class would
    // report an empty sequence for a fully staged-out history. Ask the KVMem half
    // instead and combine with the stock recurrent half exactly the way
    // llama_memory_hybrid does (min = max of the two mins, max = min of the two
    // maxes): the recurrent state lives at a single position and cannot resume
    // anywhere else, so it still bounds the range.
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    llama_memory_kvmem * attn_kvmem() { return attn_kvmem_.get(); }

private:
    std::unique_ptr<llama_memory_kvmem> attn_kvmem_;
};
