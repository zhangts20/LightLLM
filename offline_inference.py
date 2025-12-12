import argparse
import torch
from transformers import AutoTokenizer, PretrainedConfig
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.distributed import dist_group_manager
from lightllm.models import get_model
from lightllm.server.api_cli import make_argument_parser
from lightllm.utils.dist_utils import init_distributed_env
from lightllm.utils.envs_utils import set_env_start_args, is_npu


class LightLLMInfer:

    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        self.dtype = getattr(torch, args.data_type)
        self.device = "npu" if is_npu() else "cuda"
        # 初始化模型
        model_cfg, _ = PretrainedConfig.get_config_dict(args.model_dir)
        model_kvargs = {
            "args": args,
            "nccl_host": args.nccl_host,
            "data_type": args.data_type,
            "nccl_port": args.nccl_port,
            "rank_id": 0,
            "world_size": args.tp,
            "dp_size": 1,
            "weight_dir": args.model_dir,
            "quant_type": args.quant_type,
            "load_way": "HF",
            "max_total_token_num": args.max_total_token_num,
            "graph_max_len_in_batch": args.max_req_total_len,
            "graph_max_batch_size": args.graph_max_batch_size,
            "mem_fraction": args.mem_fraction,
            "max_req_num": 2048,
            "batch_max_tokens": 1024,
            "run_mode": "normal",
            "max_seq_length": args.max_req_total_len,
            "disable_cudagraph": args.disable_cudagraph,
            "mode": args.mode,
        }

        init_distributed_env(model_kvargs)
        dist_group_manager.create_groups(group_size=1)

        self.model_part, _ = get_model(model_cfg, model_kvargs=model_kvargs)
        self.model_part: TpPartBaseModel
        # 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    def prepare_model_input(self, prompt: str) -> dict:
        messages = [{"role": "user", "content": prompt}]
        applied_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(applied_text, return_tensors="pt")
        input_ids: torch.Tensor = inputs.input_ids.to(self.device)

        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[1]
        b_req_idx = torch.tensor(
            [self.model_part.req_manager.alloc() for _ in range(batch_size)],
            dtype=torch.int32,
            device="cpu")
        b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
        for i in range(batch_size):
            b_seq_len[i] = input_length
        b_ready_cache_len = torch.zeros(batch_size,
                                        dtype=torch.int32,
                                        device="cpu")

        total_token_num = batch_size * input_length
        mem_indexes = self.model_part.req_manager.mem_manager.alloc(
            input_ids.numel())
        b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device="cpu")

        return {
            "batch_size": batch_size,
            "max_len_in_batch": input_length,
            "input_ids": input_ids.squeeze(0),
            "mem_indexes": mem_indexes,
            "b_req_idx": b_req_idx,
            "b_mtp_index": b_mtp_index,
            "b_seq_len": b_seq_len,
            "total_token_num": total_token_num,
            "b_ready_cache_len": b_ready_cache_len,
        }

    def _infer_impl(
        self,
        *,
        batch_size: int,
        max_len_in_batch: int,
        input_ids: torch.Tensor,
        mem_indexes: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_mtp_index: torch.Tensor,
        b_seq_len: torch.Tensor,
        total_token_num: int,
        b_ready_cache_len: torch.Tensor = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        if is_prefill:
            max_q_seq_len = max_len_in_batch
            max_kv_seq_len = max_len_in_batch
            b_mtp_index = torch.zeros(batch_size,
                                      dtype=torch.int32,
                                      device="cpu")
            b_prefill_start_loc = b_seq_len.cumsum(
                dim=0, dtype=torch.int32) - b_seq_len
            prefix_total_token_num = 0
        else:
            max_q_seq_len = 1
            max_kv_seq_len = max_len_in_batch
            b_prefill_start_loc = None
            prefix_total_token_num = None
            b_ready_cache_len = None

        model_input = ModelInput(
            batch_size=batch_size,
            total_token_num=total_token_num,
            max_len_in_batch=max_len_in_batch,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
            input_ids=input_ids,
            b_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_mtp_index=b_mtp_index,
            mem_indexes_cpu=mem_indexes,
            is_prefill=is_prefill,
            b_ready_cache_len=b_ready_cache_len,
            b_prefill_start_loc=b_prefill_start_loc,
            prefix_total_token_num=prefix_total_token_num,
        )
        output = self.model_part.forward(model_input)

        return output.logits

    def run_infer(self, prompt: str, max_tokens: int):
        all_output_ids = []
        inputs = self.prepare_model_input(prompt)
        for i in range(max_tokens):
            inputs.update({"is_prefill": i == 0})
            logits = self._infer_impl(**inputs)
            prob_out = torch.softmax(logits, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            all_output_ids.append(predict_ids.item())

            inputs["input_ids"] = predict_ids.squeeze(0)
            inputs["total_token_num"] += inputs["batch_size"]
            inputs["b_seq_len"] += 1
            inputs[
                "mem_indexes"] = self.model_part.req_manager.mem_manager.alloc(
                    predict_ids.shape[0])
            inputs["max_len_in_batch"] = inputs["max_len_in_batch"] + i + 1
        # 释放
        self.model_part.mem_manager.free_all()
        self.model_part.req_manager.free_all()
        torch.distributed.destroy_process_group()

        return self.tokenizer.decode(all_output_ids)


if __name__ == "__main__":
    parser = make_argument_parser()
    parser.add_argument("--prompt", default="What is AI?")
    parser.add_argument("--max_tokens", type=int, default=17)

    args = parser.parse_args()
    set_env_start_args(args)

    # torch.multiprocessing.set_start_method("spawn")

    lightllm_infer = LightLLMInfer(args)
    generated_text = lightllm_infer.run_infer(prompt=args.prompt,
                                              max_tokens=args.max_tokens)
    print(f"{generated_text!r}")
