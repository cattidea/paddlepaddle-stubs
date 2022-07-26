from __future__ import annotations

from typing import Any, Optional

class FleetUtil:
    op_role_key: Any = ...
    def __init__(self, mode: str = ...) -> None: ...
    def rank0_print(self, s: Any) -> None: ...
    def rank0_info(self, s: Any) -> None: ...
    def rank0_error(self, s: Any) -> None: ...
    def set_zero(self, var_name: Any, scope: Any = ..., place: Any = ..., param_type: str = ...) -> None: ...
    def print_global_auc(
        self, scope: Any = ..., stat_pos: str = ..., stat_neg: str = ..., print_prefix: str = ...
    ) -> None: ...
    def get_global_auc(self, scope: Any = ..., stat_pos: str = ..., stat_neg: str = ...): ...
    def load_fleet_model_one_table(self, table_id: Any, path: Any) -> None: ...
    def load_fleet_model(self, path: Any, mode: int = ...) -> None: ...
    def save_fleet_model(self, path: Any, mode: int = ...) -> None: ...
    def write_model_donefile(
        self,
        output_path: Any,
        day: Any,
        pass_id: Any,
        xbox_base_key: Any,
        hadoop_fs_name: Any,
        hadoop_fs_ugi: Any,
        hadoop_home: str = ...,
        donefile_name: str = ...,
    ) -> None: ...
    def write_xbox_donefile(
        self,
        output_path: Any,
        day: Any,
        pass_id: Any,
        xbox_base_key: Any,
        data_path: Any,
        hadoop_fs_name: Any,
        hadoop_fs_ugi: Any,
        monitor_data: Any = ...,
        hadoop_home: str = ...,
        donefile_name: Optional[Any] = ...,
    ) -> None: ...
    def write_cache_donefile(
        self,
        output_path: Any,
        day: Any,
        pass_id: Any,
        key_num: Any,
        hadoop_fs_name: Any,
        hadoop_fs_ugi: Any,
        hadoop_home: str = ...,
        donefile_name: str = ...,
        **kwargs: Any,
    ) -> None: ...
    def load_model(self, output_path: Any, day: Any, pass_id: Any) -> None: ...
    def save_model(self, output_path: Any, day: Any, pass_id: Any) -> None: ...
    def save_batch_model(self, output_path: Any, day: Any) -> None: ...
    def save_delta_model(self, output_path: Any, day: Any, pass_id: Any) -> None: ...
    def save_xbox_base_model(self, output_path: Any, day: Any) -> None: ...
    def save_cache_model(self, output_path: Any, day: Any, pass_id: Any, mode: int = ..., **kwargs: Any): ...
    def save_cache_base_model(self, output_path: Any, day: Any, **kwargs: Any): ...
    def pull_all_dense_params(self, scope: Any, program: Any) -> None: ...
    def save_paddle_inference_model(
        self,
        executor: Any,
        scope: Any,
        program: Any,
        feeded_vars: Any,
        target_vars: Any,
        output_path: Any,
        day: Any,
        pass_id: Any,
        hadoop_fs_name: Any,
        hadoop_fs_ugi: Any,
        hadoop_home: str = ...,
        save_combine: bool = ...,
    ) -> None: ...
    def save_paddle_params(
        self,
        executor: Any,
        scope: Any,
        program: Any,
        model_name: Any,
        output_path: Any,
        day: Any,
        pass_id: Any,
        hadoop_fs_name: Any,
        hadoop_fs_ugi: Any,
        hadoop_home: str = ...,
        var_names: Optional[Any] = ...,
        save_combine: bool = ...,
    ) -> None: ...
    def get_last_save_xbox_base(
        self, output_path: Any, hadoop_fs_name: Any, hadoop_fs_ugi: Any, hadoop_home: str = ...
    ): ...
    def get_last_save_xbox(self, output_path: Any, hadoop_fs_name: Any, hadoop_fs_ugi: Any, hadoop_home: str = ...): ...
    def get_last_save_model(
        self, output_path: Any, hadoop_fs_name: Any, hadoop_fs_ugi: Any, hadoop_home: str = ...
    ): ...
    def get_online_pass_interval(
        self, days: Any, hours: Any, split_interval: Any, split_per_pass: Any, is_data_hourly_placed: Any
    ): ...
    def get_global_metrics(
        self,
        scope: Any = ...,
        stat_pos_name: str = ...,
        stat_neg_name: str = ...,
        sqrerr_name: str = ...,
        abserr_name: str = ...,
        prob_name: str = ...,
        q_name: str = ...,
        pos_ins_num_name: str = ...,
        total_ins_num_name: str = ...,
    ): ...
    def print_global_metrics(
        self,
        scope: Any = ...,
        stat_pos_name: str = ...,
        stat_neg_name: str = ...,
        sqrerr_name: str = ...,
        abserr_name: str = ...,
        prob_name: str = ...,
        q_name: str = ...,
        pos_ins_num_name: str = ...,
        total_ins_num_name: str = ...,
        print_prefix: str = ...,
    ) -> None: ...
    def program_type_trans(self, prog_dir: Any, prog_fn: Any, is_text: Any): ...
    def draw_from_program_file(
        self, model_filename: Any, is_text: Any, output_dir: Any, output_filename: Any
    ) -> None: ...
    def draw_from_program(self, program: Any, output_dir: Any, output_name: Any) -> None: ...
    def check_two_programs(self, config: Any): ...
    def check_vars_and_dump(self, config: Any): ...
    def parse_program_proto(self, prog_path: Any, is_text: Any, output_dir: Any) -> None: ...
    def split_program_by_device(self, program: Any): ...
