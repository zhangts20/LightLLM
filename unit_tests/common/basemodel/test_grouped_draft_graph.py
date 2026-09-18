import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]

def methods():
    path = ROOT / "lightllm/common/basemodel/basemodel.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TpPartBaseModel")
    names = {"_get_decode_graph", "_decode_padding_block_width", "_create_padded_decode_model_input"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    cls.bases = []
    class Graph:
        def __init__(self, **kw):
            self.config = kw
            self.graph = {}
    ns = {"DecodeGraph": Graph, "ModelInput": object, "torch": torch,
          "copy": copy, "F": F, "get_page_size": lambda: 16}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
    m = ns["TpPartBaseModel"]()
    m.graph = Graph()
    m.graph_max_batch_size = 18
    m.graph_max_len_in_batch = 1024
    m.tp_world_size_ = 4
    m.platform_backend = SimpleNamespace(name="maca")
    m.args = SimpleNamespace(mtp_mode="eagle_with_att", graph_split_batch_size=4, graph_grow_step_size=2)
    return m


def test_same_rows_different_query_layouts_do_not_share_graphs():
    m = methods()
    single = m._get_decode_graph(1)
    grouped = m._get_decode_graph(4)
    single.graph[4] = "four independent requests"
    grouped.graph[4] = "one request, four queries"
    assert m._get_decode_graph(1).graph[4] != m._get_decode_graph(4).graph[4]
    assert grouped.config["max_batch_size"] == 16
    assert grouped.config["batch_step_size_before_split"] == 4
    assert m._get_decode_graph(4) is grouped
    m.graph = None
    assert m._get_decode_graph(4) is None


def test_grouped_padding_owns_distinct_hold_page_slots():
    m = methods()
    m.req_manager = SimpleNamespace(HOLD_REQUEST_ID=0, req_to_token_indexs=torch.zeros((3, 32), dtype=torch.int32))
    m.mem_manager = SimpleNamespace(HOLD_TOKEN_MEMINDEX=64)
    ids = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
    source = SimpleNamespace(batch_size=4, total_token_num=74, max_kv_seq_len=20,
        decode_query_group_size=4, input_ids=torch.arange(4), b_req_idx=ids,
        b_mtp_index=torch.arange(4, dtype=torch.int32), b_seq_len=torch.arange(17, 21, dtype=torch.int32),
        b_position_delta=torch.zeros(4, dtype=torch.int32), mem_indexes=torch.arange(16, 20, dtype=torch.int32),
        multimodal_params=[{} for _ in range(4)], b_shared_seq_len=torch.zeros(4, dtype=torch.int32),
        b_shared_radix_node_id=torch.full((4,), -1, dtype=torch.int64), mtp_draft_input_hiddens=None, check_input=lambda: None)
    padded = m._create_padded_decode_model_input(source, 8)
    assert source.batch_size == 4
    assert padded.b_seq_len.tolist() == [17,18,19,20,1,2,3,4]
    assert padded.mem_indexes.tolist() == [16,17,18,19,64,65,66,67]
    assert padded.b_mtp_index.tolist() == [0,1,2,3,0,1,2,3]
    assert m.req_manager.req_to_token_indexs[0,:4].tolist() == [64,65,66,67]
    assert padded.decode_query_group_size == 4
