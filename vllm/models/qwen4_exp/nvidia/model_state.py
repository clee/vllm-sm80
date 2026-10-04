# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import os as _os

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = (
            bool(config.ple_layer_ids) and get_pp_group().is_first_rank
        )
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        # [local patch] PLE lives on the first PP rank only (see gating
        # above); the PP=1 requirement applies only to ranks that run PLE.

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        # src302/tonyd2wild staged gather: VLLM_PLE_STAGED=1 gathers PLE rows
        # from the preadv table on the host in prepare_inputs, into a fixed
        # GPU staging buffer the forward reads by views only.
        self._staged_ple: list = []
        self._ple_table = None
        if _os.environ.get("VLLM_PLE_STAGED", "0") == "1":
            from .ngram_embedding import Qwen4ExpNGramEmbedding
            from .ple_preadv import PLEPreadvTable

            model_dir = _os.environ.get(
                "QWEN4EXP_PLE_MODEL_DIR",
                _os.environ.get("VLLM_PLE_MMAP_DIR", "/models").rsplit(
                    "/.ple-mmap", 1
                )[0],
            )
            # ple_layer_ids are absolute ids (layer_idx + 1); PLE shards are
            # named for the decoder layer index below.
            ple_layer_idxs = sorted(int(i) - 1 for i in config.ple_layer_ids)
            for m in model.modules():
                if isinstance(m, Qwen4ExpNGramEmbedding):
                    emb = m.ngram_embedding
                    if self._ple_table is None:
                        self._ple_table = PLEPreadvTable(
                            model_dir, ple_layer_idxs[0]
                        )
                        if int(emb.org_vocab_size) != self._ple_table.rows_total:
                            raise ValueError(
                                f"staged PLE table rows {self._ple_table.rows_total} "
                                f"!= vocab {emb.org_vocab_size}"
                            )
                        logger.info("staged PLE table: %s", self._ple_table.describe())
                    m._ple_init_staging(self.max_num_tokens, self.device)
                    self._staged_ple.append(m)

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc)
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[-1])
        ngram_context = self._prepare_ngram_context(input_batch, req_states)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        if self._staged_ple:
            num_tokens = int(input_batch.num_tokens)
            padded = int(input_batch.num_tokens_after_padding)
            num_reqs = int(input_batch.num_reqs)
            for m in self._staged_ple:
                m._ple_stage_rows(
                    input_batch.input_ids,
                    query_start_loc[: num_reqs + 1],
                    ngram_context[:num_reqs],
                    num_tokens,
                    padded,
                    self._ple_table,
                )
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)

        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        for m in self._staged_ple:
            m._ple_stage_zero(num_tokens)  # capture-time: no host reads here
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
