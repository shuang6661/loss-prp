import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD系统级电热仿真-完全体", layout="wide")

# =============================================================================
# 工程守则（后续迭代请保留）
# 1. 三模态工况必须一直存在：Motoring / Regeneration / Stall。
# 2. 所有原始器件数据必须先归一化成单芯模型，再做系统级扩容与总损耗还原。
# 3. SiC 主开关必须保留“V0 = 0”的纯阻性锁，不允许误改成 IGBT 的 PN 结模型。
# 4. 死区补偿、温漂锁、防双重放大锁、外推监视都必须显式展示，不能藏在黑盒里。
# 5. STAR-CCM+ 默认主输出按“总热源 Total Heat Source (W)”组织，因为这是当前主要使用方式。
# =============================================================================

DEFAULT_MAIN_VI = pd.DataFrame(
    {
        "Temp (℃)": [25, 150, 25, 150],
        "Current (A)": [100.0, 100.0, 600.0, 600.0],
        "V_drop (V)": [1.10, 1.05, 2.20, 2.50],
    }
)
DEFAULT_MAIN_EI = pd.DataFrame(
    {
        "Temp (℃)": [25, 150, 25, 150],
        "Current (A)": [100.0, 100.0, 600.0, 600.0],
        "Eon (mJ)": [5.9, 8.5, 70.0, 95.0],
        "Eoff (mJ)": [4.9, 7.2, 45.0, 60.0],
    }
)
DEFAULT_DIODE_VI = pd.DataFrame(
    {
        "Temp (℃)": [25, 150, 25, 150],
        "Current (A)": [100.0, 100.0, 600.0, 600.0],
        "Vf (V)": [1.20, 1.10, 2.00, 2.20],
    }
)
DEFAULT_DIODE_EI = pd.DataFrame(
    {
        "Temp (℃)": [25, 150, 25, 150],
        "Current (A)": [100.0, 100.0, 600.0, 600.0],
        "Erec (mJ)": [1.9, 3.5, 15.0, 25.0],
    }
)

DEFAULT_MAIN_LINEAR = pd.DataFrame(
    {
        "Temp (鈩?": [25, 150],
        "Current (A)": [600.0, 600.0],
        "V0 (V)": [0.85, 1.05],
        "R_dynamic (Ω)": [0.0023, 0.0027],
    }
)
DEFAULT_DIODE_LINEAR = pd.DataFrame(
    {
        "Temp (鈩?": [25, 150],
        "Current (A)": [600.0, 600.0],
        "V0 (V)": [0.95, 1.10],
        "R_dynamic (Ω)": [0.0018, 0.0021],
    }
)

TEMP_COL = "Temp (degC)"
CURRENT_COL = "Current (A)"


def clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def canonicalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        col_str = str(col)
        if "Temp" in col_str or "temp" in col_str:
            rename_map[col] = TEMP_COL
        elif "Current" in col_str or col_str.strip().lower() in {"ic (a)", "if (a)", "current"}:
            rename_map[col] = CURRENT_COL
    return df.rename(columns=rename_map)


def validate_numeric_table(df: pd.DataFrame, table_name: str, required_cols: list[str]):
    """
    对 data_editor 输入的矩阵做工程化清洗。

    这里必须做严谨处理，因为实际复制规格书时会出现：
    - 空单元格
    - object 类型
    - 重复温度/电流点
    - 非数字字符
    """
    cleaned = canonicalize_df_columns(df.copy()).replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [col for col in required_cols if col not in cleaned.columns]
    if missing_cols:
        return cleaned, [f"{table_name} 缺少必要列：{', '.join(missing_cols)}"], warnings

    raw_required = cleaned[required_cols].copy()
    numeric_required = raw_required.apply(pd.to_numeric, errors="coerce")

    for col in required_cols:
        bad_mask = raw_required[col].notna() & numeric_required[col].isna()
        if bad_mask.any():
            sample_values = raw_required.loc[bad_mask, col].astype(str).unique()[:3]
            errors.append(f"{table_name} 的列 {col} 存在非数字内容，例如：{', '.join(sample_values)}")

    cleaned[required_cols] = numeric_required
    cleaned = cleaned.dropna(subset=required_cols)
    if cleaned.empty:
        errors.append(f"{table_name} 没有可参与计算的有效行。")
        return cleaned, errors, warnings

    if (cleaned[CURRENT_COL] < 0).any():
        errors.append(f"{table_name} 的 {CURRENT_COL} 不能为负数。")
    if (cleaned[TEMP_COL] < -273.15).any():
        errors.append(f"{table_name} 的 {TEMP_COL} 低于绝对零度。")

    value_cols = [col for col in required_cols if col not in (TEMP_COL, CURRENT_COL)]
    for value_col in value_cols:
        if (cleaned[value_col] < 0).any():
            errors.append(f"{table_name} 的 {value_col} 不能为负数。")

    duplicate_count = cleaned.duplicated(subset=[TEMP_COL, CURRENT_COL]).sum()
    if duplicate_count:
        cleaned = cleaned.groupby([TEMP_COL, CURRENT_COL], as_index=False)[value_cols].mean()
        warnings.append(f"{table_name} 存在 {duplicate_count} 组重复温度-电流点，程序已自动取平均值。")

    cleaned = cleaned.sort_values([TEMP_COL, CURRENT_COL]).reset_index(drop=True)
    temp_count = cleaned[TEMP_COL].nunique()
    point_counts = cleaned.groupby(TEMP_COL)[CURRENT_COL].nunique()
    sparse_temps = [str(temp) for temp, count in point_counts.items() if count < 2]
    if sparse_temps:
        warnings.append(
            f"{table_name} 在温度 {', '.join(sparse_temps)} ℃ 下只有 1 个电流点，对该温度只能退化为单点外推。"
        )
    if temp_count == 1:
        warnings.append(f"{table_name} 当前仅有 1 个温度维度，程序将无法做完整二维温度插值。")

    return cleaned, errors, warnings


def validate_scalar_inputs(inputs: dict):
    errors: list[str] = []
    warnings: list[str] = []

    if inputs["n_sim"] <= 0:
        errors.append("目标仿真芯片数 N_sim 必须大于 0。")
    if inputs["n_arm_system"] <= 0:
        errors.append("系统桥臂数 N_arm_sys 必须大于 0。")
    if inputs["v_ref"] <= 0:
        errors.append("规格书基准 V_nom 必须大于 0。")
    if inputs["vdc_act"] < 0 or inputs["iout_rms"] < 0 or inputs["fsw"] < 0 or inputs["fout"] < 0:
        errors.append("V_dc / I_out / f_sw / f_out 都不能为负数。")
    if not 0.0 <= inputs["m_index"] <= 1.15:
        errors.append("调制系数 M 建议位于 0.0 到 1.15 之间。")
    if not 0.0 <= inputs["cosphi"] <= 1.0:
        errors.append("功率因数幅值 cos_phi 建议位于 0.0 到 1.0 之间。")
    if min(inputs["rg_on_ref"], inputs["rg_off_ref"], inputs["rg_on_act"], inputs["rg_off_act"]) < 0:
        errors.append("门极电阻参数不能为负数。")
    if inputs["r_pkg_mohm"] < 0 or inputs["r_arm_mohm"] < 0:
        errors.append("R_pkg,chip 与 R_arm 都不能为负数。")
    if inputs["dead_time_us"] < 0:
        errors.append("死区时间 t_dead 不能为负数。")
    if "闭环" in inputs["sim_mode"]:
        if inputs["rth_jc_main"] < 0 or inputs["rth_jc_diode"] < 0:
            errors.append("闭环模式下，RthJC_main 与 RthJC_diode 都不能为负数。")

    if "sim_mode" in inputs and "diode_coupling_factor" in inputs and "闭环" in inputs["sim_mode"]:
        if not 0.0 <= inputs["diode_coupling_factor"] <= 1.5:
            errors.append("Diode coupling factor must be between 0.0 and 1.5.")
        if inputs["diode_self_heating_factor"] < 0.0:
            errors.append("Diode self-heating factor must be non-negative.")

    if inputs["r_pkg_mohm"] == 0 and inputs["r_arm_mohm"] == 0:
        warnings.append("当前默认未额外计入封装内阻和外部电路电阻，等价假设这些寄生压降已包含在原始 V-I 数据中。")
    if inputs["fsw"] == 0:
        warnings.append("当前 f_sw = 0 Hz，开关损耗将为 0。")
    if inputs["iout_rms"] == 0:
        warnings.append("当前 I_out = 0 A，结果会接近空载。")
    if inputs["dead_time_us"] > 0 and inputs["fsw"] > 0:
        dead_ratio = 2.0 * inputs["dead_time_us"] * 1e-6 * inputs["fsw"]
        if dead_ratio > 0.15:
            warnings.append("死区时间相对于开关周期偏大，死区补偿会显著改变有效占空比。")

    return errors, warnings


def calc_coupled_junction_temperatures(inputs: dict, p_main_chip: float, p_diode_chip: float) -> tuple[float, float, dict]:
    thermal_model = inputs.get("thermal_model", "main_rth_coupled")
    if thermal_model == "dual_rth_independent":
        tj_main_new = inputs["t_case_main"] + p_main_chip * inputs["rth_jc_main"]
        tj_diode_new = inputs["t_case_diode"] + p_diode_chip * inputs["rth_jc_diode"]
        thermal_meta = {
            "thermal_model": thermal_model,
            "shared_case_temp": np.nan,
            "main_rise": tj_main_new - inputs["t_case_main"],
            "diode_coupled_rise": 0.0,
            "diode_self_rise": tj_diode_new - inputs["t_case_diode"],
            "driving_power_w": p_main_chip,
        }
        return tj_main_new, tj_diode_new, thermal_meta

    if thermal_model == "half_bridge_main_reference":
        shared_case_temp = inputs["t_case_main"]
        driving_power = p_main_chip + p_diode_chip if "SiC" in inputs["device_type"] else p_main_chip
        arm_total_rise = driving_power * inputs["rth_jc_main"]
        tj_main_new = shared_case_temp + arm_total_rise
        tj_diode_new = tj_main_new if "SiC" in inputs["device_type"] else shared_case_temp + arm_total_rise
        thermal_meta = {
            "thermal_model": thermal_model,
            "shared_case_temp": shared_case_temp,
            "main_rise": arm_total_rise,
            "diode_coupled_rise": arm_total_rise,
            "diode_self_rise": 0.0,
            "driving_power_w": driving_power,
        }
        return tj_main_new, tj_diode_new, thermal_meta

    shared_case_temp = inputs["t_case_main"]
    main_rise = p_main_chip * inputs["rth_jc_main"]
    diode_coupled_rise = main_rise * inputs["diode_coupling_factor"]
    diode_self_rise = p_diode_chip * inputs["rth_jc_main"] * inputs["diode_self_heating_factor"]
    tj_main_new = shared_case_temp + main_rise
    tj_diode_new = shared_case_temp + diode_coupled_rise + diode_self_rise
    thermal_meta = {
        "thermal_model": thermal_model,
        "shared_case_temp": shared_case_temp,
        "main_rise": main_rise,
        "diode_coupled_rise": diode_coupled_rise,
        "diode_self_rise": diode_self_rise,
        "driving_power_w": p_main_chip,
    }
    return tj_main_new, tj_diode_new, thermal_meta


def describe_temperature_strategy(df: pd.DataFrame, temp_coeff: float) -> dict:
    """
    防双重放大锁：
    - 如果表格本身已经有多个温度维度，就直接用二维矩阵插值，经验温漂系数锁死为 0；
    - 如果只有一个温度维度，才允许启用经验温漂外推。
    """
    unique_temps = sorted(float(t) for t in df[TEMP_COL].dropna().unique()) if TEMP_COL in df.columns else []
    multi_temp = len(unique_temps) >= 2
    effective_temp_coeff = float(temp_coeff)
    return {
        "unique_temps": unique_temps,
        "multi_temp": multi_temp,
        "effective_temp_coeff": effective_temp_coeff,
        "strategy_label": "二维矩阵曲面插值 + 显式温漂修正" if multi_temp else "单温度查表 + 经验温漂外推",
    }


def assess_interp_usage(df: pd.DataFrame, table_name: str, target_i: float, target_t: float) -> dict:
    current_min = float(df[CURRENT_COL].min()) if CURRENT_COL in df.columns and not df.empty else np.nan
    current_max = float(df[CURRENT_COL].max()) if CURRENT_COL in df.columns and not df.empty else np.nan
    temp_min = float(df[TEMP_COL].min()) if TEMP_COL in df.columns and not df.empty else np.nan
    temp_max = float(df[TEMP_COL].max()) if TEMP_COL in df.columns and not df.empty else np.nan

    current_extrapolated = bool(np.isfinite(current_min) and np.isfinite(current_max) and (target_i < current_min or target_i > current_max))
    temp_extrapolated = bool(np.isfinite(temp_min) and np.isfinite(temp_max) and (target_t < temp_min or target_t > temp_max))

    return {
        "table_name": table_name,
        "target_i": target_i,
        "target_t": target_t,
        "current_min": current_min,
        "current_max": current_max,
        "temp_min": temp_min,
        "temp_max": temp_max,
        "current_extrapolated": current_extrapolated,
        "temp_extrapolated": temp_extrapolated,
        "any_extrapolated": current_extrapolated or temp_extrapolated,
    }


def build_matrix_health_df(inputs: dict, tables: dict) -> pd.DataFrame:
    rows = []
    cond_source_label = "手动 V0/R 表" if inputs.get("cond_param_input_mode") == "manual_linearized" else inputs["cond_data_type"]
    cond_strategy_label = "手动线性化参数插值" if inputs.get("cond_param_input_mode") == "manual_linearized" else "纯二维 V-I 曲面插值"
    table_specs = [
        ("主开关导通表", tables["ev_main"], cond_source_label, inputs["n_src_cond"] if "Module" in inputs["cond_data_type"] else 1, cond_strategy_label, None),
        ("主开关开关能量表", tables["ee_main"], inputs["sw_data_type"], inputs["n_src_sw"] if "Module" in inputs["sw_data_type"] else 1, None, inputs["t_coeff_igbt"]),
        ("二极管导通表", tables["ev_diode"], cond_source_label, inputs["n_src_cond"] if "Module" in inputs["cond_data_type"] else 1, cond_strategy_label, None),
        ("二极管恢复能量表", tables["ee_diode"], inputs["sw_data_type"], inputs["n_src_sw"] if "Module" in inputs["sw_data_type"] else 1, None, inputs["t_coeff_frd"]),
    ]

    for table_name, df, source_type, src_count, fixed_strategy, temp_coeff in table_specs:
        temp_count = int(df[TEMP_COL].nunique()) if TEMP_COL in df.columns and not df.empty else 0
        current_count = int(df[CURRENT_COL].nunique()) if CURRENT_COL in df.columns and not df.empty else 0
        current_min = float(df[CURRENT_COL].min()) if CURRENT_COL in df.columns and not df.empty else np.nan
        current_max = float(df[CURRENT_COL].max()) if CURRENT_COL in df.columns and not df.empty else np.nan

        if fixed_strategy is None:
            strategy_meta = describe_temperature_strategy(df, temp_coeff if temp_coeff is not None else 0.0)
            strategy_label = strategy_meta["strategy_label"]
            effective_temp_coeff = strategy_meta["effective_temp_coeff"]
        else:
            strategy_label = fixed_strategy
            effective_temp_coeff = 0.0

        rows.append(
            {
                "数据表": table_name,
                "原始来源": source_type,
                "归一化并联数 N_src": src_count,
                "温度维数": temp_count,
                "电流维数": current_count,
                "电流范围 (A)": f"{current_min:.2f} ~ {current_max:.2f}" if np.isfinite(current_min) and np.isfinite(current_max) else "N/A",
                "插值 / 温漂策略": strategy_label,
                "生效温漂系数": effective_temp_coeff,
            }
        )
    return pd.DataFrame(rows)


def normalize_vi_df(df: pd.DataFrame, n_src: int) -> pd.DataFrame:
    """
    导通表归一化：
    - 模块导通数据的电流轴代表整模块电流；
    - 构建单芯模型时必须除以并联芯片数；
    - 电压轴不除，因为并联芯片两端电压相同。
    """
    res_df = df.copy()
    if n_src > 1:
        res_df[CURRENT_COL] = res_df[CURRENT_COL] / float(n_src)
    return res_df


def normalize_ei_df(df: pd.DataFrame, n_src: int, e_cols: list[str]) -> pd.DataFrame:
    """
    开关能量表归一化：
    - 模块级双脉冲通常给的是整臂能量和；
    - 构建单芯模型时，电流轴与能量轴都要除以并联芯片数。
    """
    res_df = df.copy()
    if n_src > 1:
        res_df[CURRENT_COL] = res_df[CURRENT_COL] / float(n_src)
        for col in e_cols:
            if col in res_df.columns:
                res_df[col] = res_df[col] / float(n_src)
    return res_df


def normalize_linearized_param_df(df: pd.DataFrame, n_src: int) -> pd.DataFrame:
    """
    手动线性化参数表归一化：
    - 电流轴按并联芯片数折算到单芯；
    - V0 不变，因为并联芯片两端电压一致；
    - 等效电阻需还原到单芯，因此乘回并联芯片数。
    """
    res_df = df.copy()
    if n_src > 1:
        res_df[CURRENT_COL] = res_df[CURRENT_COL] / float(n_src)
        res_df["R_dynamic (Ω)"] = res_df["R_dynamic (Ω)"] * float(n_src)
    return res_df


def safe_interp(df: pd.DataFrame, target_i: float, target_t: float, item_name: str) -> float:
    """
    二维矩阵插值：
    1. 每个温度切片内按电流做一维插值；
    2. 再在温度维上插值；
    3. 数据边界不够时允许外推，但会被外推监视器显式记录。
    """
    clean_df = canonicalize_df_columns(df.dropna())
    if clean_df.empty or item_name not in clean_df.columns:
        return 0.0

    temp_list: list[float] = []
    val_list: list[float] = []
    for temp, group in clean_df.groupby(TEMP_COL):
        sorted_group = group.sort_values(CURRENT_COL)
        if len(sorted_group) >= 2:
            func = interp1d(sorted_group[CURRENT_COL], sorted_group[item_name], kind="linear", fill_value="extrapolate")
            val_list.append(max(0.0, float(func(target_i))))
            temp_list.append(float(temp))
        elif len(sorted_group) == 1:
            val_list.append(max(0.0, float(sorted_group[item_name].iloc[0])))
            temp_list.append(float(temp))

    if len(temp_list) >= 2:
        temp_interp = interp1d(temp_list, val_list, kind="linear", fill_value="extrapolate")
        return max(0.0, float(temp_interp(target_t)))
    if len(temp_list) == 1:
        return max(0.0, float(val_list[0]))
    return 0.0


def build_linearized_device_model(
    df: pd.DataFrame,
    target_i: float,
    target_t: float,
    item_name: str,
    force_zero_intercept: bool = False,
    input_mode: str = "lookup_vi",
):
    """
    用 Ipk 与 Ipk/2 两点把导通曲线拆成 V0 + R·I。

    这是你原来一直坚持保留的工程表达方式：
    - IGBT / 二极管：保留压降截距 V0
    - SiC 主开关：强制 V0 = 0，只保留纯阻性项
    """
    target_i = max(float(target_i), 1e-6)
    half_i = max(target_i / 2.0, 1e-6)

    if input_mode == "manual_linearized":
        r_eq = max(0.0, safe_interp(df, target_i, target_t, "R_dynamic (Ω)"))
        v0_manual = max(0.0, safe_interp(df, target_i, target_t, "V0 (V)"))
        v0 = 0.0 if force_zero_intercept else v0_manual
        v_pk = max(0.0, v0 + r_eq * target_i)
        v_half = max(0.0, v0 + r_eq * half_i)
    else:
        v_pk = safe_interp(df, target_i, target_t, item_name)
        v_half = safe_interp(df, half_i, target_t, item_name)
        denom = target_i - half_i
        r_eq = max(0.0, (v_pk - v_half) / denom) if denom > 1e-12 else 0.0
        v0 = 0.0 if force_zero_intercept else max(0.0, v_pk - r_eq * target_i)

    return {
        "v_pk": max(0.0, float(v_pk)),
        "v_half": max(0.0, float(v_half)),
        "r_eq": max(0.0, float(r_eq)),
        "v0": max(0.0, float(v0)),
    }


def calc_switching_energy(
    df: pd.DataFrame,
    i_pk: float,
    tj: float,
    algo_type: str,
    i_nom_ref: float,
    item_name: str,
    vdc: float,
    vref: float,
    kv: float,
    ract: float,
    rref: float,
    kr: float,
    temp_coeff: float,
    tref: float,
    ki_frd: float = 1.0,
    is_diode: bool = False,
) -> dict:
    """
    开关能量修正链：
    基础能量提取 -> 温漂锁判断 -> 门极电阻修正 -> 电压指数修正

    这里新增两种企业对标模式：
    1. CAE精确二维插值：直接按当前 Ipk 与 Tj 查矩阵；
    2. 标称点直线比例法：先查 I_nom_ref 的标称能量，再按电流线性比例放大。
    """
    strategy_meta = describe_temperature_strategy(df, temp_coeff)
    if "直线比例法" in algo_type:
        nominal_current = max(float(i_nom_ref), 1e-12)
        e_nom = safe_interp(df, nominal_current, tj, item_name)
        current_factor = (
            math.pow(max(float(i_pk), 0.0) / nominal_current, ki_frd)
            if is_diode
            else (max(float(i_pk), 0.0) / nominal_current)
        )
        e_base = e_nom * current_factor
        extraction_label = f"标称点直线比例法 (I_nom={nominal_current:.3f} A)"
        e_nom_mj = max(0.0, float(e_nom))
    else:
        e_base = safe_interp(df, i_pk, tj, item_name)
        extraction_label = "CAE精确二维插值"
        e_nom_mj = np.nan

    if e_base <= 0.0:
        return {
            "energy_mj": 0.0,
            "e_base_mj": 0.0,
            "e_nom_mj": e_nom_mj,
            "temp_correction": 1.0,
            "rg_correction": 1.0,
            "voltage_correction": 1.0,
            "effective_temp_coeff": strategy_meta["effective_temp_coeff"],
            "strategy_label": strategy_meta["strategy_label"],
            "temp_lock_active": strategy_meta["multi_temp"],
            "extraction_label": extraction_label,
        }

    temp_correction = max(0.0, 1.0 + strategy_meta["effective_temp_coeff"] * (tj - tref))
    rg_correction = math.pow(max(ract, 1e-12) / max(rref, 1e-12), kr) if rref > 0 else 1.0
    voltage_correction = math.pow(max(vdc, 1e-12) / max(vref, 1e-12), kv) if vref > 0 else 1.0
    energy_mj = max(0.0, e_base * temp_correction * rg_correction * voltage_correction)

    return {
        "energy_mj": energy_mj,
        "e_base_mj": max(0.0, float(e_base)),
        "e_nom_mj": e_nom_mj,
        "temp_correction": temp_correction,
        "rg_correction": rg_correction,
        "voltage_correction": voltage_correction,
        "effective_temp_coeff": strategy_meta["effective_temp_coeff"],
        "strategy_label": strategy_meta["strategy_label"],
        "temp_lock_active": strategy_meta["multi_temp"],
        "extraction_label": extraction_label,
    }


def calc_dead_time_compensation(mode: str, fsw: float, dead_time_us: float, m_index: float, current_sign: float, vdc: float):
    """
    死区补偿显式保留：
    D_dead = 2 * t_dead * f_sw
    M_eff  = M - sgn(i) * K_mode * D_dead
    """
    if fsw <= 0.0 or dead_time_us <= 0.0:
        return {
            "dead_ratio": 0.0,
            "m_eff": m_index,
            "current_sign": current_sign,
            "phase_voltage_error_v": 0.0,
            "modulation_gain": 0.0,
        }

    dead_time_s = dead_time_us * 1e-6
    dead_ratio = clamp(2.0 * dead_time_s * fsw, 0.0, 0.20)
    modulation_gain = 4.0 / math.pi if mode == "SVPWM" else 2.0 / math.pi
    m_eff = clamp(m_index - current_sign * modulation_gain * dead_ratio, 0.0, 1.15)
    phase_voltage_error_v = 0.5 * vdc * dead_ratio * current_sign
    return {
        "dead_ratio": dead_ratio,
        "m_eff": m_eff,
        "current_sign": current_sign,
        "phase_voltage_error_v": phase_voltage_error_v,
        "modulation_gain": modulation_gain,
    }


def calc_pwm_conduction_losses(
    mode: str,
    m_eff: float,
    active_cosphi: float,
    theta: float,
    i_pk_chip: float,
    main_model: dict,
    diode_model: dict,
    r_pkg_chip: float,
    r_arm_chip: float,
    dead_meta: dict,
    is_sic: bool = False,
):
    """
    PWM 导通损耗：
    - 保留 SPWM / SVPWM 两套解析式
    - R_pkg 与 R_arm 显式进入 R 项
    - 死区引起的导通路径重分配显式加入
    """
    r_main_total = main_model["r_eq"] + r_pkg_chip + r_arm_chip
    r_diode_total = diode_model["r_eq"] + r_pkg_chip + r_arm_chip

    inst_main = main_model["v0"] * i_pk_chip + r_main_total * i_pk_chip**2
    inst_diode = diode_model["v0"] * i_pk_chip + r_diode_total * i_pk_chip**2

    if mode == "SVPWM":
        kv0_m = (m_eff * active_cosphi) / 4.0
        kr_m = m_eff * (24.0 * active_cosphi - 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) - 3.0 * math.sqrt(3.0)) / 24.0
        kv0_d = (4.0 - m_eff * math.pi * active_cosphi) / 4.0
        kr_d = m_eff * (
            6.0 * math.pi
            - 24.0 * active_cosphi
            + 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta)
            + 3.0 * math.sqrt(3.0)
        ) / 24.0

        p_cond_main = (kv0_m * main_model["v0"] * i_pk_chip) + (kr_m * r_main_total * i_pk_chip**2) / math.pi
        p_cond_diode = (kv0_d * diode_model["v0"] * i_pk_chip) / math.pi + (kr_d * r_diode_total * i_pk_chip**2) / math.pi
    else:
        if is_sic:
            p_cond_main = r_main_total * i_pk_chip**2 * (1.0 / 8.0 + m_eff * active_cosphi / (3.0 * math.pi))
        else:
            p_cond_main = (
                main_model["v0"] * i_pk_chip * (1.0 / (2.0 * math.pi) + m_eff * active_cosphi / 8.0)
                + r_main_total * i_pk_chip**2 * (1.0 / 8.0 + m_eff * active_cosphi / (3.0 * math.pi))
            )
        p_cond_diode = (
            diode_model["v0"] * i_pk_chip * (1.0 / (2.0 * math.pi) - m_eff * active_cosphi / 8.0)
            + r_diode_total * i_pk_chip**2 * (1.0 / 8.0 - m_eff * active_cosphi / (3.0 * math.pi))
        )

    p_cond_main = max(0.0, float(p_cond_main))
    p_cond_diode = max(0.0, float(p_cond_diode))

    dead_ratio = dead_meta["dead_ratio"]
    if dead_ratio > 0.0:
        avg_inst_main = (main_model["v0"] * i_pk_chip * 2.0 / math.pi) + (r_main_total * i_pk_chip**2 * 0.5)
        avg_inst_diode = (diode_model["v0"] * i_pk_chip * 2.0 / math.pi) + (r_diode_total * i_pk_chip**2 * 0.5)
        if dead_meta["current_sign"] >= 0.0:
            p_cond_main = max(0.0, p_cond_main - dead_ratio * avg_inst_main)
            p_cond_diode = p_cond_diode + dead_ratio * avg_inst_diode
        else:
            p_cond_diode = max(0.0, p_cond_diode - dead_ratio * avg_inst_diode)
            p_cond_main = p_cond_main + dead_ratio * avg_inst_main

    return {
        "p_cond_main": p_cond_main,
        "p_cond_diode": p_cond_diode,
        "r_main_total": r_main_total,
        "r_diode_total": r_diode_total,
        "avg_inst_main": avg_inst_main if dead_ratio > 0.0 else 0.0,
        "avg_inst_diode": avg_inst_diode if dead_ratio > 0.0 else 0.0,
    }


def calc_stall_losses(
    m_eff: float,
    i_pk_chip: float,
    main_model: dict,
    diode_model: dict,
    r_pkg_chip: float,
    r_arm_chip: float,
    dead_meta: dict,
):
    """
    堵转极限工况：
    舍弃 1/pi 正弦平均因子，按最大占空比做最保守的直流重载评估。
    """
    r_main_total = main_model["r_eq"] + r_pkg_chip + r_arm_chip
    r_diode_total = diode_model["r_eq"] + r_pkg_chip + r_arm_chip

    inst_main = main_model["v0"] * i_pk_chip + r_main_total * i_pk_chip**2
    inst_diode = diode_model["v0"] * i_pk_chip + r_diode_total * i_pk_chip**2
    d_max = clamp(0.5 * (1.0 + m_eff), 0.0, 1.0)
    p_cond_main = d_max * inst_main
    p_cond_diode = (1.0 - d_max) * inst_diode

    dead_ratio = dead_meta["dead_ratio"]
    if dead_ratio > 0.0:
        p_cond_main = max(0.0, p_cond_main - dead_ratio * inst_main)
        p_cond_diode = p_cond_diode + dead_ratio * inst_diode

    return {
        "p_cond_main": p_cond_main,
        "p_cond_diode": p_cond_diode,
        "r_main_total": r_main_total,
        "r_diode_total": r_diode_total,
        "d_max": d_max,
    }


def build_star_ccm_total_heat_table(n_sim: int, p_main_chip: float, p_diode_chip: float) -> pd.DataFrame:
    rows = []
    for idx in range(1, n_sim + 1):
        rows.append({"Region": f"MainChip_{idx}", "Category": "Main Switch", "Total Heat Source (W)": round(p_main_chip, 6)})
        rows.append({"Region": f"DiodeChip_{idx}", "Category": "Freewheel Diode", "Total Heat Source (W)": round(p_diode_chip, 6)})
    return pd.DataFrame(rows)


def build_icepak_heat_table(n_sim: int, p_main_chip: float, p_diode_chip: float) -> pd.DataFrame:
    rows = []
    for idx in range(1, n_sim + 1):
        rows.append({"Region": f"MainChip_{idx}", "Category": "Main Switch", "Heat Generation Rate (W)": round(p_main_chip, 6)})
        rows.append({"Region": f"DiodeChip_{idx}", "Category": "Freewheel Diode", "Heat Generation Rate (W)": round(p_diode_chip, 6)})
    return pd.DataFrame(rows)


def build_excel_bytes(sheet_map: dict[str, pd.DataFrame]):
    """
    Excel 导出是可选能力：
    - 装了 openpyxl 就正常导出
    - 没装也不影响主计算与 CSV 导出
    """
    try:
        import openpyxl  # noqa: F401
    except ModuleNotFoundError:
        return None, "当前运行环境未安装 openpyxl，已自动跳过 Excel 导出；CSV 导出仍可正常使用。"

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheet_map.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output.getvalue(), None


def build_formula_audit_df(inputs: dict, dead_meta: dict, eon_meta: dict, eoff_meta: dict, erec_meta: dict) -> pd.DataFrame:
    sic_lock_active = "SiC" in inputs["device_type"]
    mode_label = "SVPWM 马鞍波解析式" if inputs["mode"] == "SVPWM" else "SPWM 正弦解析式"
    thermal_label = "闭环双节点热迭代" if "闭环" in inputs["sim_mode"] else "开环固定结温"
    return pd.DataFrame(
        [
            {"审计项": "运行工况", "当前实现": inputs["op_mode"], "说明": "Motoring / Regeneration / Stall 三模态独立计算"},
            {"审计项": "调制解析式", "当前实现": mode_label, "说明": "保留原始工程积分框架，死区修正体现在 M_eff"},
            {"审计项": "主开关导通模型", "当前实现": "SiC 纯阻性锁 ON" if sic_lock_active else "IGBT 的 V0 + R 线性化", "说明": "SiC 主开关强制 V0 = 0"},
            {"审计项": "续流路径模型", "当前实现": "Vf0 + R 线性化", "说明": "续流二极管保留压降项"},
            {"审计项": "单芯/半桥换算路径", "当前实现": "半桥臂后参优先，先算半桥臂再折算单芯", "说明": "仅在 Bare Die 数据路径下直接按单芯坐标计算"},
            {"审计项": "死区补偿", "当前实现": f"启用，M_eff = {dead_meta['m_eff']:.6f}" if dead_meta["dead_ratio"] > 0 else "关闭", "说明": "同时修正有效调制系数和导通路径分配"},
            {"审计项": "Eon 温度策略", "当前实现": eon_meta["strategy_label"], "说明": "多温度矩阵时经验温漂自动锁死为 0"},
            {"审计项": "Eoff 温度策略", "当前实现": eoff_meta["strategy_label"], "说明": "防双重放大锁已启用"},
            {"审计项": "Erec 温度策略", "当前实现": erec_meta["strategy_label"], "说明": "防双重放大锁已启用"},
            {"审计项": "开关能量提取算法", "当前实现": eon_meta["extraction_label"], "说明": "导通域与开关域电流独立评估；FRD 比例法保留 K_i_frd 非线性缩放"},
            {"审计项": "热学模式", "当前实现": thermal_label, "说明": "IGBT 默认仅用主开关损耗映射半桥臂热阻；SiC 默认用主开关与体二极管组合损耗映射；保留兼容双热阻模式"},
            {"审计项": "STAR-CCM+ 输出偏好", "当前实现": "总热源 Total Heat Source (W)", "说明": "当前以单颗总热源表为主输出"},
        ]
    )


def simulate_system(inputs: dict, tables: dict):
    cond_src_count = inputs["n_src_cond"] if "Module" in inputs["cond_data_type"] else 1
    sw_src_count = inputs["n_src_sw"] if "Module" in inputs["sw_data_type"] else 1
    use_half_bridge_cond = "Module" in inputs["cond_data_type"]
    use_half_bridge_sw = "Module" in inputs["sw_data_type"]
    bridge_chip_count = max(inputs["n_sim"], 1)

    if use_half_bridge_cond:
        if inputs.get("cond_param_input_mode") == "manual_linearized":
            norm_ev_m = canonicalize_df_columns(tables["ev_main"].copy())
            norm_ev_d = canonicalize_df_columns(tables["ev_diode"].copy())
        else:
            norm_ev_m = canonicalize_df_columns(tables["ev_main"].copy())
            norm_ev_d = canonicalize_df_columns(tables["ev_diode"].copy())
        i_pk_cond_domain = math.sqrt(2.0) * inputs["iout_rms"]
        r_arm_eval = inputs["r_arm_mohm"] / 1000.0
        r_pkg_eval = inputs["r_pkg_mohm"] / 1000.0
    else:
        if inputs.get("cond_param_input_mode") == "manual_linearized":
            norm_ev_m = normalize_linearized_param_df(tables["ev_main"], cond_src_count)
            norm_ev_d = normalize_linearized_param_df(tables["ev_diode"], cond_src_count)
        else:
            norm_ev_m = normalize_vi_df(tables["ev_main"], cond_src_count)
            norm_ev_d = normalize_vi_df(tables["ev_diode"], cond_src_count)
        i_pk_cond_domain = math.sqrt(2.0) * (inputs["iout_rms"] / bridge_chip_count) if bridge_chip_count > 0 else 0.0
        r_arm_eval = (inputs["r_arm_mohm"] / 1000.0) * bridge_chip_count
        r_pkg_eval = inputs["r_pkg_mohm"] / 1000.0

    if use_half_bridge_sw:
        norm_ee_m = canonicalize_df_columns(tables["ee_main"].copy())
        norm_ee_d = canonicalize_df_columns(tables["ee_diode"].copy())
        i_pk_sw_domain = math.sqrt(2.0) * inputs["iout_rms"]
        i_nom_eval = float(inputs["i_nom_ref"])
    else:
        norm_ee_m = normalize_ei_df(tables["ee_main"], sw_src_count, ["Eon (mJ)", "Eoff (mJ)"])
        norm_ee_d = normalize_ei_df(tables["ee_diode"], sw_src_count, ["Erec (mJ)"])
        i_pk_sw_domain = math.sqrt(2.0) * (inputs["iout_rms"] / bridge_chip_count) if bridge_chip_count > 0 else 0.0
        i_nom_eval = float(inputs["i_nom_ref"]) / bridge_chip_count if bridge_chip_count > 0 else float(inputs["i_nom_ref"])

    i_pk_chip = math.sqrt(2.0) * (inputs["iout_rms"] / bridge_chip_count) if bridge_chip_count > 0 else 0.0

    active_cosphi = -abs(inputs["cosphi"]) if "Regeneration" in inputs["op_mode"] else abs(inputs["cosphi"])
    active_cosphi = clamp(active_cosphi, -1.0, 1.0)
    theta = math.acos(active_cosphi) if abs(active_cosphi) <= 1.0 else 0.0

    current_sign = -1.0 if "Regeneration" in inputs["op_mode"] else 1.0
    dead_meta = calc_dead_time_compensation(inputs["mode"], inputs["fsw"], inputs["dead_time_us"], inputs["m_index"], current_sign, inputs["vdc_act"])

    max_iter = 30
    min_iter_before_break = 15 if "闭环" in inputs["sim_mode"] else 1
    tolerance = 0.05
    if "开环" in inputs["sim_mode"]:
        tj_main_current = inputs["fixed_tj"]
        tj_diode_current = inputs["fixed_tj"]
        loop_count = 1
    else:
        tj_main_current = inputs["t_case_main"] + 5.0
        tj_diode_current = inputs["t_case_diode"] + 5.0
        loop_count = max_iter

    iteration_rows = []
    extrapolation_log: dict[str, dict] = {}
    thermal_meta = {
        "thermal_model": inputs.get("thermal_model", ""),
        "shared_case_temp": np.nan,
        "main_rise": np.nan,
        "diode_coupled_rise": np.nan,
        "diode_self_rise": np.nan,
    }

    for loop_idx in range(loop_count):
        interp_checks = [
            assess_interp_usage(norm_ev_m, "主开关导通表", i_pk_cond_domain, tj_main_current),
            assess_interp_usage(norm_ev_d, "二极管导通表", i_pk_cond_domain, tj_diode_current),
            assess_interp_usage(norm_ee_m, "主开关开关能量表", i_pk_sw_domain, tj_main_current),
            assess_interp_usage(norm_ee_d, "二极管恢复能量表", i_pk_sw_domain, tj_diode_current),
        ]
        for check in interp_checks:
            if check["table_name"] not in extrapolation_log:
                extrapolation_log[check["table_name"]] = check.copy()
            else:
                extrapolation_log[check["table_name"]]["current_extrapolated"] |= check["current_extrapolated"]
                extrapolation_log[check["table_name"]]["temp_extrapolated"] |= check["temp_extrapolated"]
                extrapolation_log[check["table_name"]]["any_extrapolated"] = (
                    extrapolation_log[check["table_name"]]["current_extrapolated"] or extrapolation_log[check["table_name"]]["temp_extrapolated"]
                )
                extrapolation_log[check["table_name"]]["target_i"] = max(extrapolation_log[check["table_name"]]["target_i"], check["target_i"])
                extrapolation_log[check["table_name"]]["target_t"] = check["target_t"]

        main_model = build_linearized_device_model(
            norm_ev_m,
            i_pk_cond_domain,
            tj_main_current,
            "V_drop (V)",
            force_zero_intercept="SiC" in inputs["device_type"],
            input_mode=inputs.get("cond_param_input_mode", "lookup_vi"),
        )
        diode_model = build_linearized_device_model(
            norm_ev_d,
            i_pk_cond_domain,
            tj_diode_current,
            "Vf (V)",
            force_zero_intercept=False,
            input_mode=inputs.get("cond_param_input_mode", "lookup_vi"),
        )

        if "Stall" in inputs["op_mode"]:
            cond_result = calc_stall_losses(dead_meta["m_eff"], i_pk_cond_domain, main_model, diode_model, r_pkg_eval, r_arm_eval, dead_meta)
        else:
            cond_result = calc_pwm_conduction_losses(
                inputs["mode"],
                dead_meta["m_eff"],
                active_cosphi,
                theta,
                i_pk_cond_domain,
                main_model,
                diode_model,
                r_pkg_eval,
                r_arm_eval,
                dead_meta,
                is_sic="SiC" in inputs["device_type"],
            )

        eon_meta = calc_switching_energy(
            norm_ee_m,
            i_pk_sw_domain,
            tj_main_current,
            inputs["algo_type"],
            i_nom_eval,
            "Eon (mJ)",
            inputs["vdc_act"],
            inputs["v_ref"],
            inputs["kv_on"],
            inputs["rg_on_act"],
            inputs["rg_on_ref"],
            inputs["kron"],
            inputs["t_coeff_igbt"],
            inputs["t_ref_dp"],
        )
        eoff_meta = calc_switching_energy(
            norm_ee_m,
            i_pk_sw_domain,
            tj_main_current,
            inputs["algo_type"],
            i_nom_eval,
            "Eoff (mJ)",
            inputs["vdc_act"],
            inputs["v_ref"],
            inputs["kv_off"],
            inputs["rg_off_act"],
            inputs["rg_off_ref"],
            inputs["kroff"],
            inputs["t_coeff_igbt"],
            inputs["t_ref_dp"],
        )
        erec_meta = calc_switching_energy(
            norm_ee_d,
            i_pk_sw_domain,
            tj_diode_current,
            inputs["algo_type"],
            i_nom_eval,
            "Erec (mJ)",
            inputs["vdc_act"],
            inputs["v_ref"],
            inputs["kv_frd"],
            1.0,
            1.0,
            0.0,
            inputs["t_coeff_frd"],
            inputs["t_ref_dp"],
            ki_frd=inputs["ki_frd"],
            is_diode=True,
        )

        eon_adj = eon_meta["energy_mj"]
        eoff_adj = eoff_meta["energy_mj"]
        erec_adj = erec_meta["energy_mj"]

        if "Stall" in inputs["op_mode"]:
            p_sw_main_chip = inputs["fsw"] * ((eon_adj + eoff_adj) / 1000.0)
            p_sw_diode_chip = inputs["fsw"] * (erec_adj / 1000.0)
        else:
            p_sw_main_chip = (inputs["fsw"] / math.pi) * ((eon_adj + eoff_adj) / 1000.0)
            p_sw_diode_chip = (inputs["fsw"] / math.pi) * (erec_adj / 1000.0)

        if use_half_bridge_cond or use_half_bridge_sw:
            p_main_arm = cond_result["p_cond_main"] + p_sw_main_chip
            p_diode_arm = cond_result["p_cond_diode"] + p_sw_diode_chip
            p_main_chip = p_main_arm / bridge_chip_count
            p_diode_chip = p_diode_arm / bridge_chip_count
        else:
            p_main_chip = cond_result["p_cond_main"] + p_sw_main_chip
            p_diode_chip = cond_result["p_cond_diode"] + p_sw_diode_chip
            p_main_arm = p_main_chip * bridge_chip_count
            p_diode_arm = p_diode_chip * bridge_chip_count
        p_total_arm = p_main_arm + p_diode_arm

        if "闭环" in inputs["sim_mode"]:
            tj_main_new, tj_diode_new, thermal_meta = calc_coupled_junction_temperatures(inputs, p_main_chip, p_diode_chip)
        else:
            tj_main_new = inputs["fixed_tj"]
            tj_diode_new = inputs["fixed_tj"]

        iteration_rows.append(
            {
                "Iter": loop_idx + 1,
                "Tj_main_used (℃)": round(tj_main_current, 6),
                "Tj_diode_used (℃)": round(tj_diode_current, 6),
                "P_main_chip (W)": round(p_main_chip, 6),
                "P_diode_chip (W)": round(p_diode_chip, 6),
                "P_main_arm (W)": round(p_main_arm, 6),
                "P_diode_arm (W)": round(p_diode_arm, 6),
                "Tj_main_new (℃)": round(tj_main_new, 6),
                "Tj_diode_new (℃)": round(tj_diode_new, 6),
                "Main_rise (K)": round(float(thermal_meta["main_rise"]), 6) if np.isfinite(thermal_meta["main_rise"]) else np.nan,
                "Diode_coupled_rise (K)": round(float(thermal_meta["diode_coupled_rise"]), 6) if np.isfinite(thermal_meta["diode_coupled_rise"]) else np.nan,
                "Diode_self_rise (K)": round(float(thermal_meta["diode_self_rise"]), 6) if np.isfinite(thermal_meta["diode_self_rise"]) else np.nan,
            }
        )

        tj_main_current = tj_main_new
        tj_diode_current = tj_diode_new

        if "闭环" in inputs["sim_mode"]:
            if (loop_idx + 1) >= min_iter_before_break and max(
                abs(tj_main_new - iteration_rows[-1]["Tj_main_used (℃)"]),
                abs(tj_diode_new - iteration_rows[-1]["Tj_diode_used (℃)"]),
            ) < tolerance:
                break

    dominant_tj = max(tj_main_current, tj_diode_current)
    p_total_system = p_total_arm * inputs["n_arm_system"]
    p_main_total_system = p_main_arm * inputs["n_arm_system"]
    p_diode_total_system = p_diode_arm * inputs["n_arm_system"]
    p_main_switch_position = p_main_arm
    p_diode_switch_position = p_diode_arm

    star_ccm_df = build_star_ccm_total_heat_table(bridge_chip_count, p_main_chip, p_diode_chip)
    icepak_df = build_icepak_heat_table(bridge_chip_count, p_main_chip, p_diode_chip)
    matrix_health_df = build_matrix_health_df(inputs, tables)

    extrapolation_df = pd.DataFrame(
        [
            {
                "数据表": item["table_name"],
                "目标电流 (A)": item["target_i"],
                "目标结温 (℃)": item["target_t"],
                "电流覆盖范围 (A)": f"{item['current_min']:.2f} ~ {item['current_max']:.2f}",
                "温度覆盖范围 (℃)": f"{item['temp_min']:.2f} ~ {item['temp_max']:.2f}",
                "电流越界": "是" if item["current_extrapolated"] else "否",
                "温度越界": "是" if item["temp_extrapolated"] else "否",
            }
            for item in extrapolation_log.values()
        ]
    )
    extrapolation_messages = []
    for item in extrapolation_log.values():
        if item["any_extrapolated"]:
            reasons = []
            if item["current_extrapolated"]:
                reasons.append("电流轴外推")
            if item["temp_extrapolated"]:
                reasons.append("温度轴外推")
            extrapolation_messages.append(f"{item['table_name']} 发生 {' + '.join(reasons)}，请优先补齐矩阵边界。")

    loss_breakdown_df = pd.DataFrame(
        [
            {"对象": "主开关芯片", "项目": "导通损耗", "单颗 (W)": cond_result["p_cond_main"], "单臂总计 (W)": cond_result["p_cond_main"] * inputs["n_sim"]},
            {"对象": "主开关芯片", "项目": "开通+关断损耗", "单颗 (W)": p_sw_main_chip, "单臂总计 (W)": p_sw_main_chip * inputs["n_sim"]},
            {"对象": "续流二极管", "项目": "导通损耗", "单颗 (W)": cond_result["p_cond_diode"], "单臂总计 (W)": cond_result["p_cond_diode"] * inputs["n_sim"]},
            {"对象": "续流二极管", "项目": "恢复损耗", "单颗 (W)": p_sw_diode_chip, "单臂总计 (W)": p_sw_diode_chip * inputs["n_sim"]},
            {"对象": "系统级缩放", "项目": "主芯片系统总损耗", "单颗 (W)": np.nan, "单臂总计 (W)": p_main_total_system},
            {"对象": "系统级缩放", "项目": "二极管系统总损耗", "单颗 (W)": np.nan, "单臂总计 (W)": p_diode_total_system},
            {"对象": "系统级缩放", "项目": "系统总功耗", "单颗 (W)": np.nan, "单臂总计 (W)": p_total_system},
        ]
    )

    linearized_df = pd.DataFrame(
        [
            {
                "对象": "主开关芯片",
                "Tj_used (℃)": tj_main_current,
                "V@Ipk (V)": main_model["v_pk"],
                "V@Ipk/2 (V)": main_model["v_half"],
                "V0 (V)": main_model["v0"],
                "R_dynamic (Ω)": main_model["r_eq"],
                "R_total (Ω)": cond_result["r_main_total"],
                "Pure Resistive Lock": "ON" if "SiC" in inputs["device_type"] else "OFF",
            },
            {
                "对象": "续流二极管",
                "Tj_used (℃)": tj_diode_current,
                "V@Ipk (V)": diode_model["v_pk"],
                "V@Ipk/2 (V)": diode_model["v_half"],
                "V0 (V)": diode_model["v0"],
                "R_dynamic (Ω)": diode_model["r_eq"],
                "R_total (Ω)": cond_result["r_diode_total"],
                "Pure Resistive Lock": "N/A",
            },
        ]
    )

    diode_temp_label = "体二极管结温" if "SiC" in inputs["device_type"] else "续流路径参考温度"

    summary_df = pd.DataFrame(
        [
            {"指标": "运行场景", "数值": inputs["op_mode"], "单位": ""},
            {"指标": "芯片技术类型", "数值": inputs["device_type"], "单位": ""},
            {"指标": "调制模式", "数值": inputs["mode"], "单位": ""},
            {"指标": "目标并联芯片数 N_sim", "数值": inputs["n_sim"], "单位": ""},
            {"指标": "系统桥臂数 N_arm_sys", "数值": inputs["n_arm_system"], "单位": ""},
            {"指标": "单颗峰值电流 I_pk", "数值": i_pk_chip, "单位": "A"},
            {"指标": "导通域评估电流 I_pk_cond", "数值": i_pk_cond_domain, "单位": "A"},
            {"指标": "开关域评估电流 I_pk_sw", "数值": i_pk_sw_domain, "单位": "A"},
            {"指标": "主开关当前动态电阻", "数值": main_model["r_eq"], "单位": "Ω"},
            {"指标": "主开关当前压降", "数值": main_model["v_pk"], "单位": "V"},
            {"指标": "二极管当前动态电阻", "数值": diode_model["r_eq"], "单位": "Ω"},
            {"指标": "二极管当前压降", "数值": diode_model["v_pk"], "单位": "V"},
            {"指标": "主芯片结温", "数值": tj_main_current, "单位": "℃"},
            {"指标": diode_temp_label, "数值": tj_diode_current, "单位": "℃"},
            {"指标": "控制结温（取最大）", "数值": dominant_tj, "单位": "℃"},
            {"指标": "热学映射口径", "数值": inputs["thermal_model"], "单位": ""},
            {"指标": "半桥臂单颗总损耗", "数值": p_main_chip + p_diode_chip, "单位": "W"},
            {"指标": "结温映射驱动损耗", "数值": thermal_meta.get("driving_power_w", np.nan), "单位": "W"},
            {"指标": "主芯片温升", "数值": thermal_meta["main_rise"], "单位": "K"},
            {"指标": "续流路径映射温升", "数值": thermal_meta["diode_coupled_rise"], "单位": "K"},
            {"指标": "续流路径独立附加温升", "数值": thermal_meta["diode_self_rise"], "单位": "K"},
            {"指标": "主芯片单颗发热率", "数值": p_main_chip, "单位": "W"},
            {"指标": "二极管单颗发热率", "数值": p_diode_chip, "单位": "W"},
            {"指标": "主开关并联位总损耗", "数值": p_main_switch_position, "单位": "W"},
            {"指标": "续流并联位总损耗", "数值": p_diode_switch_position, "单位": "W"},
            {"指标": "单臂总功耗", "数值": p_total_arm, "单位": "W"},
            {"指标": "系统级总功耗", "数值": p_total_system, "单位": "W"},
            {"指标": "死区等效占空比损失", "数值": dead_meta["dead_ratio"], "单位": "p.u."},
            {"指标": "死区修正后的调制系数", "数值": dead_meta["m_eff"], "单位": ""},
            {"指标": "死区等效相电压误差", "数值": dead_meta["phase_voltage_error_v"], "单位": "V"},
            {"指标": "闭环最少迭代门槛", "数值": min_iter_before_break, "单位": "次"},
            {"指标": "实际迭代次数", "数值": len(iteration_rows), "单位": "次"},
            {"指标": "Eon 温度策略", "数值": eon_meta["strategy_label"], "单位": ""},
            {"指标": "Eoff 温度策略", "数值": eoff_meta["strategy_label"], "单位": ""},
            {"指标": "Erec 温度策略", "数值": erec_meta["strategy_label"], "单位": ""},
        ]
    )

    temp_strategy_df = pd.DataFrame(
        [
            {"对象": "Eon", "提取算法": eon_meta["extraction_label"], "基础查表值 (mJ)": eon_meta["e_base_mj"], "标称点能量 (mJ)": eon_meta["e_nom_mj"], "生效温漂系数": eon_meta["effective_temp_coeff"], "温漂修正因子": eon_meta["temp_correction"], "电阻修正因子": eon_meta["rg_correction"], "电压修正因子": eon_meta["voltage_correction"], "最终能量 (mJ)": eon_meta["energy_mj"], "温度策略": eon_meta["strategy_label"]},
            {"对象": "Eoff", "提取算法": eoff_meta["extraction_label"], "基础查表值 (mJ)": eoff_meta["e_base_mj"], "标称点能量 (mJ)": eoff_meta["e_nom_mj"], "生效温漂系数": eoff_meta["effective_temp_coeff"], "温漂修正因子": eoff_meta["temp_correction"], "电阻修正因子": eoff_meta["rg_correction"], "电压修正因子": eoff_meta["voltage_correction"], "最终能量 (mJ)": eoff_meta["energy_mj"], "温度策略": eoff_meta["strategy_label"]},
            {"对象": "Erec", "提取算法": erec_meta["extraction_label"], "基础查表值 (mJ)": erec_meta["e_base_mj"], "标称点能量 (mJ)": erec_meta["e_nom_mj"], "生效温漂系数": erec_meta["effective_temp_coeff"], "温漂修正因子": erec_meta["temp_correction"], "电阻修正因子": erec_meta["rg_correction"], "电压修正因子": erec_meta["voltage_correction"], "最终能量 (mJ)": erec_meta["energy_mj"], "温度策略": erec_meta["strategy_label"]},
        ]
    )

    formula_audit_df = build_formula_audit_df(inputs, dead_meta, eon_meta, eoff_meta, erec_meta)

    input_snapshot_df = pd.DataFrame(
        [
            {"参数": "模块芯片技术类型", "取值": inputs["device_type"], "单位": ""},
            {"参数": "导通数据来源", "取值": inputs["cond_data_type"], "单位": ""},
            {"参数": "导通原测芯片数", "取值": cond_src_count, "单位": ""},
            {"参数": "开关数据来源", "取值": inputs["sw_data_type"], "单位": ""},
            {"参数": "开关原测芯片数", "取值": sw_src_count, "单位": ""},
            {"参数": "后参计算路径", "取值": "半桥臂优先" if ("Module" in inputs["cond_data_type"] or "Module" in inputs["sw_data_type"]) else "单芯直算", "单位": ""},
            {"参数": "导通域评估电流 I_pk_cond", "取值": i_pk_cond_domain, "单位": "A"},
            {"参数": "开关域评估电流 I_pk_sw", "取值": i_pk_sw_domain, "单位": "A"},
            {"参数": "开关域基准电流 I_nom_eval", "取值": i_nom_eval, "单位": "A"},
            {"参数": "目标仿真芯片数", "取值": inputs["n_sim"], "单位": ""},
            {"参数": "系统桥臂数", "取值": inputs["n_arm_system"], "单位": ""},
            {"参数": "热学模式", "取值": inputs["sim_mode"], "单位": ""},
            {"参数": "热学映射口径", "取值": inputs["thermal_model"], "单位": ""},
            {"参数": "主/二极管热参数分离", "取值": inputs["split_thermal_params"], "单位": ""},
            {"参数": "母线电压 V_dc", "取值": inputs["vdc_act"], "单位": "V"},
            {"参数": "输出电流 I_out", "取值": inputs["iout_rms"], "单位": "A"},
            {"参数": "导通参数输入方式", "取值": inputs["cond_param_input_mode"], "单位": ""},
            {"参数": "开关频率 f_sw", "取值": inputs["fsw"], "单位": "Hz"},
            {"参数": "输出频率 f_out", "取值": inputs["fout"], "单位": "Hz"},
            {"参数": "调制系数 M", "取值": inputs["m_index"], "单位": ""},
            {"参数": "功率因数幅值 cos_phi", "取值": inputs["cosphi"], "单位": ""},
            {"参数": "死区时间 t_dead", "取值": inputs["dead_time_us"], "单位": "us"},
            {"参数": "规格书基准 V_nom", "取值": inputs["v_ref"], "单位": "V"},
            {"参数": "规格书基准 T_ref", "取值": inputs["t_ref_dp"], "单位": "℃"},
            {"参数": "R_g,on_ref", "取值": inputs["rg_on_ref"], "单位": "Ω"},
            {"参数": "R_g,off_ref", "取值": inputs["rg_off_ref"], "单位": "Ω"},
            {"参数": "R_on_act", "取值": inputs["rg_on_act"], "单位": "Ω"},
            {"参数": "R_off_act", "取值": inputs["rg_off_act"], "单位": "Ω"},
            {"参数": "开关能量提取算法", "取值": inputs["algo_type"], "单位": ""},
            {"参数": "直线基准电流 I_nom", "取值": inputs["i_nom_ref"], "单位": "A"},
            {"参数": "K_v_on", "取值": inputs["kv_on"], "单位": ""},
            {"参数": "K_v_off", "取值": inputs["kv_off"], "单位": ""},
            {"参数": "K_v_frd", "取值": inputs["kv_frd"], "单位": ""},
            {"参数": "K_i_frd", "取值": inputs["ki_frd"], "单位": ""},
            {"参数": "R_pkg,chip", "取值": inputs["r_pkg_mohm"], "单位": "mΩ"},
            {"参数": "R_arm", "取值": inputs["r_arm_mohm"], "单位": "mΩ"},
            {"参数": "主管温漂系数", "取值": inputs["t_coeff_igbt"], "单位": "1/K"},
            {"参数": "续流温漂系数", "取值": inputs["t_coeff_frd"], "单位": "1/K"},
            {"参数": "页首仿真备注", "取值": inputs["user_notes"], "单位": ""},
            {"参数": "工程师备忘录", "取值": inputs["engineer_memo"], "单位": ""},
        ]
    )
    if "闭环" in inputs["sim_mode"]:
        input_snapshot_df = pd.concat(
            [
                input_snapshot_df,
                pd.DataFrame(
                    [
                        {"参数": "主芯片 RthJC_main", "取值": inputs["rth_jc_main"], "单位": "K/W"},
                        {"参数": "二极管 RthJC_diode", "取值": inputs["rth_jc_diode"], "单位": "K/W"},
                        {"参数": "主芯片 Tc_main", "取值": inputs["t_case_main"], "单位": "℃"},
                        {"参数": "二极管 Tc_diode", "取值": inputs["t_case_diode"], "单位": "℃"},
                        {"参数": "半桥臂总损耗参与结温映射", "取值": "SiC" in inputs["device_type"] and inputs["thermal_model"] == "half_bridge_main_reference", "单位": ""},
                        {"参数": "IGBT仅主开关损耗映射结温", "取值": "SiC" not in inputs["device_type"] and inputs["thermal_model"] == "half_bridge_main_reference", "单位": ""},
                        {"参数": "二极管耦合系数", "取值": inputs["diode_coupling_factor"], "单位": ""},
                        {"参数": "二极管自发热权重", "取值": inputs["diode_self_heating_factor"], "单位": ""},
                    ]
                ),
            ],
            ignore_index=True,
        )
    else:
        input_snapshot_df = pd.concat(
            [input_snapshot_df, pd.DataFrame([{"参数": "固定结温 Tj", "取值": inputs["fixed_tj"], "单位": "℃"}])],
            ignore_index=True,
        )

    excel_sheets = {
        "summary": summary_df,
        "loss_breakdown": loss_breakdown_df,
        "star_ccm_total_heat": star_ccm_df,
        "icepak_heat": icepak_df,
        "linearized_model": linearized_df,
        "iterations": pd.DataFrame(iteration_rows),
        "matrix_health": matrix_health_df,
        "extrapolation_monitor": extrapolation_df,
        "temp_strategy": temp_strategy_df,
        "formula_audit": formula_audit_df,
        "input_snapshot": input_snapshot_df,
        "raw_main_vi": tables["ev_main"],
        "raw_main_ei": tables["ee_main"],
        "raw_diode_vi": tables["ev_diode"],
        "raw_diode_ei": tables["ee_diode"],
        "norm_main_vi": norm_ev_m,
        "norm_main_ei": norm_ee_m,
        "norm_diode_vi": norm_ev_d,
        "norm_diode_ei": norm_ee_d,
    }
    excel_bytes, excel_warning = build_excel_bytes(excel_sheets)

    return {
        "device_type": inputs["device_type"],
        "op_mode": inputs["op_mode"],
        "n_sim": inputs["n_sim"],
        "n_arm_system": inputs["n_arm_system"],
        "user_notes": inputs["user_notes"],
        "engineer_memo": inputs["engineer_memo"],
        "norm_ev_m": norm_ev_m,
        "norm_ev_d": norm_ev_d,
        "norm_ee_m": norm_ee_m,
        "norm_ee_d": norm_ee_d,
        "i_pk_chip": i_pk_chip,
        "dead_meta": dead_meta,
        "cond_param_input_mode": inputs["cond_param_input_mode"],
        "main_model": main_model,
        "diode_model": diode_model,
        "eon_meta": eon_meta,
        "eoff_meta": eoff_meta,
        "erec_meta": erec_meta,
        "p_main_chip": p_main_chip,
        "p_diode_chip": p_diode_chip,
        "p_total_arm": p_total_arm,
        "p_total_system": p_total_system,
        "p_main_switch_position": p_main_switch_position,
        "p_diode_switch_position": p_diode_switch_position,
        "p_main_total_system": p_main_total_system,
        "p_diode_total_system": p_diode_total_system,
        "tj_main_current": tj_main_current,
        "tj_diode_current": tj_diode_current,
        "diode_temp_label": diode_temp_label,
        "dominant_tj": dominant_tj,
        "thermal_meta": thermal_meta,
        "summary_df": summary_df,
        "loss_breakdown_df": loss_breakdown_df,
        "star_ccm_df": star_ccm_df,
        "icepak_df": icepak_df,
        "linearized_df": linearized_df,
        "iteration_df": pd.DataFrame(iteration_rows),
        "matrix_health_df": matrix_health_df,
        "extrapolation_df": extrapolation_df,
        "temp_strategy_df": temp_strategy_df,
        "formula_audit_df": formula_audit_df,
        "input_snapshot_df": input_snapshot_df,
        "extrapolation_messages": extrapolation_messages,
        "excel_bytes": excel_bytes,
        "excel_warning": excel_warning,
        "star_ccm_csv_bytes": star_ccm_df.to_csv(index=False).encode("utf-8-sig"),
        "icepak_csv_bytes": icepak_df.to_csv(index=False).encode("utf-8-sig"),
        "breakdown_csv_bytes": loss_breakdown_df.to_csv(index=False).encode("utf-8-sig"),
    }


st.title("🛡️ 功率模块全工况电热联合仿真平台 (STAR-CCM+ 导向版)")

with st.expander("📝 工程随手记 & 快速操作指南", expanded=True):
    guide_col, note_col = st.columns([1.15, 1.0])
    with guide_col:
        st.markdown(
            """
            **🚀 快速操作流程：**
            1. 左侧边栏先定义 **器件架构、原始数据来源、目标并联芯片数、系统桥臂数**。
            2. 将规格书或测试报告中的主芯片 / 二极管 **V-I、E-I 矩阵** 粘贴进动态表格。
            3. 输入工况：母线电压、相电流、开关频率、调制系数、死区时间。
            4. 若要反拖，请直接切换到 **Regeneration**，不要靠手工输入负 `cos_phi` 来代替。
            5. 点击计算后，重点查看：
               - 主芯片 / 二极管单颗发热率
               - STAR-CCM+ 总热源表
               - 死区修正量
               - 矩阵外推预警

            **📌 关键工程提醒：**
            - `R_pkg,chip` 与 `R_arm` 默认都为 **0 mΩ**，用于避免和原始 V-I 曲线重复计损。
            - SiC 主开关始终保留 **V0 = 0 的纯阻性锁**，这是刻意保留的底层物理规则。
            - 如果 E-I 表已经有多个温度维度，经验温漂系数会被自动锁死为 0。
            """
        )
    with note_col:
        user_notes = st.text_area(
            "🗒️ 仿真备注 (项目 / 规格书版本 / 对标结论)",
            placeholder="例如：A平台 800V SiC 6并联，双脉冲报告 V3.2，STAR-CCM+ 挂总热源……",
            height=170,
        )
        st.caption("建议把结果截图与这段备注一起存档，避免之后忘记上下文。")

with st.sidebar:
    st.header("⚙️ 核心技术架构")
    st.info("工程显式假设：封装内阻 `R_pkg,chip` 与桥臂附加电阻 `R_arm` 默认都为 0。若原始 V-I 已含这些压降，请勿重复录入。")

    device_type = st.radio(
        "1. 模块芯片技术类型",
        ["IGBT + FRD (传统硅基)", "SiC MOSFET (碳化硅)"],
        help="IGBT 保留 V0 + R；SiC 主开关强制 V0 = 0，仅保留纯阻性通道。",
    )

    st.divider()
    st.header("🧮 原始数据规格 (必填)")
    st.warning("程序会根据这里的选择，自动把原始数据拉平成“单芯模型”。")
    cond_data_type = st.radio("A. 导通 V-I 表格代表：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"])
    n_src_cond = st.number_input("V-I 原测模块芯片数", value=6, min_value=1, disabled="单芯片数据" in cond_data_type)

    sw_data_type = st.radio("B. 开关 E-I 表格代表：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"])
    n_src_sw = st.number_input("E-I 原测模块芯片数", value=6, min_value=1, disabled="单芯片数据" in sw_data_type)

    st.divider()
    st.header("🎯 仿真目标规模重构")
    n_sim = st.number_input("目标仿真芯片数 (N_sim)", value=6, min_value=1)
    n_arm_system = st.number_input("系统桥臂数 (N_arm_sys)", value=1, min_value=1, help="三相逆变器通常填 3；只关心单个半桥时填 1。")

    st.divider()
    st.header("🔄 热学计算工作流")
    sim_mode = st.radio("模式选择", ["A. 开环盲算 (已知结温)", "B. 闭环迭代 (已知热阻)"])
    if "闭环" in sim_mode:
        thermal_model_label = st.radio(
            "热学映射口径",
            ["半桥臂热阻主导（推荐）", "主芯片 / 二极管独立双热阻", "主芯片热阻主导 + 二极管耦合"],
        )
        split_thermal_params = thermal_model_label == "主芯片 / 二极管独立双热阻"
        rth_label = "半桥臂参考热阻 Rth_hb (K/W)" if thermal_model_label == "半桥臂热阻主导（推荐）" else "主芯片热阻 RthJC_main (K/W)"
        tc_label = "半桥臂参考壳温 Tc_hb (℃)" if thermal_model_label == "半桥臂热阻主导（推荐）" else "主芯片参考壳温 Tc_main (℃)"
        rth_jc_main = st.number_input(rth_label, value=0.065, min_value=0.0, format="%.4f")
        t_case_main = st.number_input(tc_label, value=65.0)
        if split_thermal_params:
            rth_jc_diode = st.number_input("二极管热阻 RthJC_diode (K/W)", value=0.085, min_value=0.0, format="%.4f")
            t_case_diode = st.number_input("二极管参考壳温 Tc_diode (℃)", value=65.0)
            diode_coupling_factor = 0.0
            diode_self_heating_factor = 1.0
            thermal_model = "dual_rth_independent"
        elif thermal_model_label == "半桥臂热阻主导（推荐）":
            diode_coupling_factor = 0.0
            diode_self_heating_factor = 0.0
            rth_jc_diode = 0.0
            t_case_diode = t_case_main
            thermal_model = "half_bridge_main_reference"
        else:
            diode_coupling_factor = st.number_input("二极管耦合系数", value=0.85, min_value=0.0, max_value=1.5, format="%.2f")
            diode_self_heating_factor = st.number_input("二极管自发热权重", value=0.25, min_value=0.0, max_value=1.5, format="%.2f")
            rth_jc_diode = 0.0
            t_case_diode = t_case_main
            thermal_model = "main_rth_coupled"
        fixed_tj = None
    else:
        fixed_tj = st.number_input("设定全局目标结温 Tj (℃)", value=150.0)
        thermal_model = "open_loop_fixed_tj"
        split_thermal_params = False
        rth_jc_main = None
        rth_jc_diode = None
        t_case_main = None
        t_case_diode = None
        diode_coupling_factor = None
        diode_self_heating_factor = None

    st.divider()
    engineer_memo = st.text_area(
        "🧠 工程师专属备忘录",
        placeholder="例如：STAR-CCM+ 当前用总热源；反拖一定切 Regeneration；SiC 主开关 V0=0 不可改……",
        height=180,
    )

st.divider()
st.header("📊 第一步：特性数据录入 (归一化中心)")
st.info("默认按半桥臂后参优先处理：先计算半桥臂损耗，再折算单芯并做系统级扩容；只有 Bare Die 数据路径才直接按单芯坐标计算。")
cond_param_input_mode = st.radio(
    "导通特性参数输入方式",
    ["1. 查表输入 V-I 曲线 (现有方式)", "2. 手动输入线性化参数 V0 + R"],
    help="第一种方式按 V-I 矩阵自动在当前工作点线性化；第二种方式可直接录入已手算好的 V0 和动态电阻，并按温度/电流插值使用。",
)

col_main, col_diode = st.columns(2)
with col_main:
    st.subheader("🔴 主开关管 (IGBT / SiC)")
    if "手动输入线性化参数" in cond_param_input_mode:
        st.caption("1. 手动线性化导通参数表 (V0 / R_dynamic)")
        ev_main = st.data_editor(DEFAULT_MAIN_LINEAR, num_rows="dynamic", key="v_main_linear")
    else:
        st.caption("1. 导通特性 (Vce / Vds)")
        ev_main = st.data_editor(DEFAULT_MAIN_VI, num_rows="dynamic", key="v_main")
    st.caption("2. 开关能量矩阵 (Eon / Eoff)")
    ee_main = st.data_editor(DEFAULT_MAIN_EI, num_rows="dynamic", key="e_main")

with col_diode:
    st.subheader("🔵 续流二极管 (FRD / Body Diode)")
    if "手动输入线性化参数" in cond_param_input_mode:
        st.caption("1. 手动线性化导通参数表 (V0 / R_dynamic)")
        ev_diode = st.data_editor(DEFAULT_DIODE_LINEAR, num_rows="dynamic", key="v_diode_linear")
    else:
        st.caption("1. 正向压降 (Vf / Vsd)")
        ev_diode = st.data_editor(DEFAULT_DIODE_VI, num_rows="dynamic", key="v_diode")
    st.caption("2. 反向恢复能量 (Erec)")
    ee_diode = st.data_editor(DEFAULT_DIODE_EI, num_rows="dynamic", key="ee_diode")

if "手动输入线性化参数" in cond_param_input_mode and "SiC" in device_type:
    st.info("SiC 主开关仍保留 `V0 = 0` 纯阻性锁。手动表中的主开关 `V0` 会参与展示，但计算时会自动按 0 处理。")

st.divider()
st.header("⚙️ 第二步：全场景工况与物理修正系数")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**⚡ 车辆 / 电驱动工况**")
    op_mode = st.selectbox("🏎️ 运行场景切换", ["电动/巡航 (Motoring)", "制动/反拖 (Regeneration)", "最恶劣堵转 (Stall)"])
    vdc_act = st.number_input("母线 V_dc (V)", value=713.0, min_value=0.0)
    iout_rms = st.number_input("有效值 I_out (A)", value=264.5, min_value=0.0)
    fsw = st.number_input("开关频率 f_sw (Hz)", value=10000.0, min_value=0.0)
    fout = st.number_input("输出频率 f_out (Hz)", value=200.0, min_value=0.0)
    m_index = st.number_input("调制系数 M", value=0.90, min_value=0.0, max_value=1.15)
    cosphi = st.number_input("功率因数幅值 cos_phi", value=0.90, min_value=0.0, max_value=1.0)
    mode = st.selectbox("调制模式选择", ["SVPWM", "SPWM"])
    if fout < 5.0 and "Stall" not in op_mode:
        st.warning(f"当前输出频率 {fout:.2f} Hz 很低，若接近堵转请切换到 Stall 模式。")

with c2:
    st.markdown("**📏 测试基准 / 驱动 / 死区**")
    v_ref = st.number_input("规格书基准 V_nom (V)", value=600.0, min_value=0.001)
    t_ref_dp = st.number_input("规格书基准 T_ref (℃)", value=150.0)
    rg_on_ref = st.number_input("手册 R_g,on (Ω)", value=2.5, min_value=0.0)
    rg_off_ref = st.number_input("手册 R_g,off (Ω)", value=20.0, min_value=0.0)
    rg_on_act = st.number_input("实际 R_on (Ω)", value=2.5, min_value=0.0)
    rg_off_act = st.number_input("实际 R_off (Ω)", value=20.0, min_value=0.0)
    algo_type = st.radio(
        "开关能量提取算法",
        ["1. CAE精确二维插值 (推荐)", "2. 标称点直线比例法 (对标公司报告)"],
        help="CAE 精确法直接按当前 Ipk 与 Tj 查二维矩阵；直线比例法先取 I_nom 标称点，再按电流线性放缩。",
    )
    if "直线比例法" in algo_type:
        i_nom_ref = st.number_input("直线基准电流 I_nom (A)", value=400.0, min_value=0.001)
    else:
        i_nom_ref = 400.0
    dead_time_us = st.number_input("死区时间 t_dead (us)", value=2.0, min_value=0.0, format="%.3f")

with c3:
    st.markdown("**📈 拟合修正指数**")
    kv_on = st.number_input("开通指数 K_v_on", value=1.30)
    kv_off = st.number_input("关断指数 K_v_off", value=1.30)
    kv_frd = st.number_input("续流指数 K_v_frd", value=0.60)
    ki_frd = st.number_input("续流电流指数 K_i_frd", value=0.60)
    kron = st.number_input("电阻系数 K_ron", value=0.30)
    kroff = st.number_input("电阻系数 K_roff", value=0.50)

with c4:
    st.markdown("**🌡️ 温漂 / 封装 / 电路寄生**")
    t_coeff_igbt = st.number_input("主管温漂系数 (1/K)", value=0.003, format="%.4f")
    t_coeff_frd = st.number_input("续流温漂系数 (1/K)", value=0.006 if "IGBT" in device_type else 0.003, format="%.4f")
    r_pkg_mohm = st.number_input("封装内阻 R_pkg,chip (mΩ)", value=0.0, min_value=0.0)
    r_arm_mohm = st.number_input("桥臂附加电阻 R_arm (mΩ)", value=0.0, min_value=0.0)

st.info("工程显式提醒：`R_pkg,chip` 代表单颗封装寄生，`R_arm` 代表公共电路寄生。两者默认都为 0，是为了避免和原始 V-I 曲线重复计损。")
st.warning("死区补偿不是装饰项。只要 `t_dead > 0` 且 `f_sw > 0`，程序就会同时修正有效调制系数和主开关/二极管导通路径分配。")

inputs = {
    "device_type": device_type,
    "cond_param_input_mode": "manual_linearized" if "手动输入线性化参数" in cond_param_input_mode else "lookup_vi",
    "cond_data_type": cond_data_type,
    "n_src_cond": int(n_src_cond),
    "sw_data_type": sw_data_type,
    "n_src_sw": int(n_src_sw),
    "n_sim": int(n_sim),
    "n_arm_system": int(n_arm_system),
    "sim_mode": sim_mode,
    "thermal_model": thermal_model,
    "split_thermal_params": bool(split_thermal_params),
    "rth_jc_main": 0.0 if rth_jc_main is None else float(rth_jc_main),
    "rth_jc_diode": 0.0 if rth_jc_diode is None else float(rth_jc_diode),
    "t_case_main": 0.0 if t_case_main is None else float(t_case_main),
    "t_case_diode": 0.0 if t_case_diode is None else float(t_case_diode),
    "diode_coupling_factor": 0.0 if diode_coupling_factor is None else float(diode_coupling_factor),
    "diode_self_heating_factor": 0.0 if diode_self_heating_factor is None else float(diode_self_heating_factor),
    "fixed_tj": 150.0 if fixed_tj is None else float(fixed_tj),
    "op_mode": op_mode,
    "vdc_act": float(vdc_act),
    "iout_rms": float(iout_rms),
    "fsw": float(fsw),
    "fout": float(fout),
    "m_index": float(m_index),
    "cosphi": float(cosphi),
    "mode": mode,
    "v_ref": float(v_ref),
    "t_ref_dp": float(t_ref_dp),
    "rg_on_ref": float(rg_on_ref),
    "rg_off_ref": float(rg_off_ref),
    "rg_on_act": float(rg_on_act),
    "rg_off_act": float(rg_off_act),
    "algo_type": algo_type,
    "i_nom_ref": float(i_nom_ref),
    "dead_time_us": float(dead_time_us),
    "kv_on": float(kv_on),
    "kv_off": float(kv_off),
    "kv_frd": float(kv_frd),
    "ki_frd": float(ki_frd),
    "kron": float(kron),
    "kroff": float(kroff),
    "t_coeff_igbt": float(t_coeff_igbt),
    "t_coeff_frd": float(t_coeff_frd),
    "r_pkg_mohm": float(r_pkg_mohm),
    "r_arm_mohm": float(r_arm_mohm),
    "user_notes": user_notes,
    "engineer_memo": engineer_memo,
}

tables_for_validation = {}
validation_errors: list[str] = []
validation_warnings: list[str] = []

main_cond_required_cols = [TEMP_COL, CURRENT_COL, "V0 (V)", "R_dynamic (Ω)"] if inputs["cond_param_input_mode"] == "manual_linearized" else [TEMP_COL, CURRENT_COL, "V_drop (V)"]
diode_cond_required_cols = [TEMP_COL, CURRENT_COL, "V0 (V)", "R_dynamic (Ω)"] if inputs["cond_param_input_mode"] == "manual_linearized" else [TEMP_COL, CURRENT_COL, "Vf (V)"]
main_cond_table_name = "主开关管手动线性化参数表" if inputs["cond_param_input_mode"] == "manual_linearized" else "主开关管导通表"
diode_cond_table_name = "二极管手动线性化参数表" if inputs["cond_param_input_mode"] == "manual_linearized" else "二极管导通表"

validated_main_vi, errs, warns = validate_numeric_table(ev_main, main_cond_table_name, main_cond_required_cols)
tables_for_validation["ev_main"] = validated_main_vi
validation_errors.extend(errs)
validation_warnings.extend(warns)

validated_main_ei, errs, warns = validate_numeric_table(ee_main, "主开关管开关能量表", [TEMP_COL, CURRENT_COL, "Eon (mJ)", "Eoff (mJ)"])
tables_for_validation["ee_main"] = validated_main_ei
validation_errors.extend(errs)
validation_warnings.extend(warns)

validated_diode_vi, errs, warns = validate_numeric_table(ev_diode, diode_cond_table_name, diode_cond_required_cols)
tables_for_validation["ev_diode"] = validated_diode_vi
validation_errors.extend(errs)
validation_warnings.extend(warns)

validated_diode_ei, errs, warns = validate_numeric_table(ee_diode, "二极管恢复能量表", [TEMP_COL, CURRENT_COL, "Erec (mJ)"])
tables_for_validation["ee_diode"] = validated_diode_ei
validation_errors.extend(errs)
validation_warnings.extend(warns)

scalar_errors, scalar_warnings = validate_scalar_inputs(inputs)
validation_errors.extend(scalar_errors)
validation_warnings.extend(scalar_warnings)
matrix_health_preview_df = build_matrix_health_df(inputs, tables_for_validation)

with st.expander("🧭 数据矩阵健康度与温漂锁诊断", expanded=False):
    st.dataframe(matrix_health_preview_df, use_container_width=True)
    st.caption("只要某张开关能量表检测到两个及以上温度维度，该表对应的经验温漂系数就会被自动锁死为 0，避免双重放大。")

st.divider()
st.header("🚀 第三步：执行全工况联合仿真")

for warning_text in validation_warnings:
    st.warning(warning_text)

compute_requested = st.button("🚀 执 行 全 工 况 仿 真 计 算", use_container_width=True)

if compute_requested:
    if validation_errors:
        st.session_state.pop("simulation_result", None)
        st.error("输入校验未通过，请先修正以下问题：")
        for err in validation_errors:
            st.write(f"- {err}")
    else:
        result = simulate_system(inputs, tables_for_validation)
        result["warnings"] = validation_warnings
        st.session_state["simulation_result"] = result

result = st.session_state.get("simulation_result")

if result:
    st.success(f"✅ 计算完成！当前状态：{result['op_mode']} | 并联 N: {result['n_sim']}")
    for message in result["extrapolation_messages"]:
        st.warning(message)
    if "SiC" in result["device_type"]:
        st.info("SiC 架构锁已启用：主开关导通模型强制 `V0 = 0`，底层仅保留纯阻性发热项；二极管侧仍保留 `Vf0 + R`。")
    if result["thermal_meta"]["thermal_model"] == "half_bridge_main_reference":
        if "SiC" in result["device_type"]:
            st.info("当前结温按主开关芯片与体二极管组合损耗映射到半桥臂参考热阻。")
        else:
            st.info("当前结温仅按 IGBT 主开关损耗映射到半桥臂参考热阻；FRD 对 IGBT 的热耦合默认视为已包含在该后参中。")
    if result["excel_warning"]:
        st.warning(result["excel_warning"])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("控制结温 Tj,max", f"{result['dominant_tj']:.1f} ℃")
    m2.metric("主芯片结温", f"{result['tj_main_current']:.1f} ℃")
    m3.metric(result["diode_temp_label"], f"{result['tj_diode_current']:.1f} ℃")
    m4.metric("单臂总功耗", f"{result['p_total_arm']:.1f} W")
    m5.metric("系统级总功耗", f"{result['p_total_system']:.1f} W")

    st.divider()
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("🔴 主芯片单颗发热率", f"{result['p_main_chip']:.2f} W")
    p2.metric("🔵 二极管单颗发热率", f"{result['p_diode_chip']:.2f} W")
    p3.metric("🔴 主开关并联位总损耗", f"{result['p_main_switch_position']:.1f} W")
    p4.metric("🔵 续流并联位总损耗", f"{result['p_diode_switch_position']:.1f} W")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("单颗峰值电流 I_pk", f"{result['i_pk_chip']:.2f} A")
    e2.metric("死区占空损失 D_dead", f"{result['dead_meta']['dead_ratio']:.4f}")
    e3.metric("死区修正后 M_eff", f"{result['dead_meta']['m_eff']:.4f}")
    e4.metric("系统级总功耗", f"{result['p_total_system']:.1f} W")

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("主开关动态电阻", f"{result['main_model']['r_eq']:.6f} Ω")
    v2.metric("主开关当前压降", f"{result['main_model']['v_pk']:.4f} V")
    v3.metric("二极管动态电阻", f"{result['diode_model']['r_eq']:.6f} Ω")
    v4.metric("二极管当前压降", f"{result['diode_model']['v_pk']:.4f} V")
    st.caption("以上导通参数均对应当前计算工作点的导通域评估电流 `I_pk_cond`。")

    st.info(
        f"开关能量温度策略：Eon = {result['eon_meta']['strategy_label']}；"
        f"Eoff = {result['eoff_meta']['strategy_label']}；"
        f"Erec = {result['erec_meta']['strategy_label']}。"
    )
    st.caption(
        f"当前结果里：'单颗发热率' = 单个 die；'并联位总损耗' = 一个开关位置上 {result['n_sim']} 颗并联芯片合计。"
        f" 这一步通常更接近公司模块级小程序的口径；系统级总量再按 N_arm_sys = {result['n_arm_system']} 做桥臂缩放。"
    )

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button("⬇️ 下载 STAR-CCM+ 总热源 CSV", data=result["star_ccm_csv_bytes"], file_name="star_ccm_total_heat_source.csv", mime="text/csv", use_container_width=True)
    with d2:
        st.download_button("⬇️ 下载 Icepak 热源 CSV", data=result["icepak_csv_bytes"], file_name="icepak_heat_generation_rate.csv", mime="text/csv", use_container_width=True)
    with d3:
        st.download_button("⬇️ 下载损耗拆分 CSV", data=result["breakdown_csv_bytes"], file_name="loss_breakdown.csv", mime="text/csv", use_container_width=True)
    with d4:
        if result["excel_bytes"] is not None:
            st.download_button("⬇️ 下载完整 Excel", data=result["excel_bytes"], file_name="system_level_thermal_platform_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        else:
            st.info("当前环境未安装 openpyxl，Excel 下载已关闭。")

    tabs = st.tabs(["结果总览", "STAR-CCM+ 总热源", "Icepak 热源", "线性化模型", "热迭代历史", "归一化数据", "矩阵健康度", "外推监视", "公式审计"])

    with tabs[0]:
        st.markdown("**损耗拆分总表**")
        st.dataframe(result["loss_breakdown_df"], use_container_width=True)
        st.markdown("**核心汇总指标**")
        st.dataframe(result["summary_df"], use_container_width=True)
        st.markdown("**输入快照**")
        st.dataframe(result["input_snapshot_df"], use_container_width=True)

    with tabs[1]:
        st.markdown("**面向 STAR-CCM+ 的单颗总热源表**")
        st.dataframe(result["star_ccm_df"], use_container_width=True, height=420)
        st.caption("这张表按每颗 die 给出 Total Heat Source (W)，适合你当前在 STAR-CCM+ 里的使用习惯。")

    with tabs[2]:
        st.markdown("**用于 Icepak / Flotherm 的单颗发热率表**")
        st.dataframe(result["icepak_df"], use_container_width=True, height=420)

    with tabs[3]:
        st.markdown("**当前工作点线性化结果：V0 + R × I**")
        st.dataframe(result["linearized_df"], use_container_width=True)
        st.caption("SiC 主开关在这里会明确显示 `Pure Resistive Lock = ON`。")

    with tabs[4]:
        st.markdown("**闭环热迭代记录**")
        st.dataframe(result["iteration_df"], use_container_width=True)
        st.caption("闭环模式下至少完成 15 次迭代后，才允许按 0.05 ℃ 收敛阈值提前退出。")

    with tabs[5]:
        st.markdown("**归一化后的单芯 V-I / E-I 数据**")
        subtabs = st.tabs(["主芯片 V-I", "主芯片 E-I", "二极管 V-I", "二极管 E-I"])
        with subtabs[0]:
            st.dataframe(result["norm_ev_m"], use_container_width=True)
        with subtabs[1]:
            st.dataframe(result["norm_ee_m"], use_container_width=True)
        with subtabs[2]:
            st.dataframe(result["norm_ev_d"], use_container_width=True)
        with subtabs[3]:
            st.dataframe(result["norm_ee_d"], use_container_width=True)

    with tabs[6]:
        st.markdown("**矩阵维度、归一化来源与温漂锁状态**")
        st.dataframe(result["matrix_health_df"], use_container_width=True)

    with tabs[7]:
        st.markdown("**插值 / 外推边界监视**")
        st.dataframe(result["extrapolation_df"], use_container_width=True)

    with tabs[8]:
        st.markdown("**底层公式 / 开关策略审计**")
        st.dataframe(result["formula_audit_df"], use_container_width=True)
        st.markdown("**开关能量修正明细**")
        st.dataframe(result["temp_strategy_df"], use_container_width=True)

    with st.expander("🧠 工程师专属备忘录回显", expanded=False):
        st.markdown("**页首仿真备注**")
        st.write(result["user_notes"] if result["user_notes"] else "（当前未填写）")
        st.markdown("**侧栏工程师备忘录**")
        st.write(result["engineer_memo"] if result["engineer_memo"] else "（当前未填写）")

st.divider()
with st.expander("🔬 查看底层应用的完整物理公式与工程约定 (长期保留，不准删)", expanded=False):
    st.markdown("### 📘 1. 单芯归一化与扩容重构")
    st.latex(r"I_{chip,src} = \frac{I_{module,src}}{N_{src}}")
    st.latex(r"E_{chip,src} = \frac{E_{module,src}}{N_{src}}")
    st.latex(r"I_{pk,chip} = \sqrt{2}\cdot \frac{I_{out,rms}}{N_{sim}}")
    st.latex(r"P_{system,total} = N_{arm,sys}\cdot P_{arm,total}")
    st.write("先归一化为单芯模型，再乘以 N_sim 与 N_arm_sys 还原目标模块与系统总量。")

    st.markdown("### 📗 2. 导通模型线性化")
    st.latex(r"R_{eq} = \frac{V(I_{pk}) - V(I_{pk}/2)}{I_{pk} - I_{pk}/2}")
    st.latex(r"V_0 = V(I_{pk}) - R_{eq}\cdot I_{pk}")
    st.latex(r"V_{0,SiC}=0 \quad \Rightarrow \quad V_{SiC}(I)\approx R_{ds(on),eq}\cdot I")
    st.write("SiC 主开关不是新公式，而是在你原始 V0 + R 框架上强制令 V0 = 0。")

    st.markdown("### 📙 3. PWM 导通损耗")
    st.markdown("#### 3.1 SPWM")
    st.latex(r"P_{cond,IGBT} = \left(\frac{1}{2\pi} + \frac{M_{eff}\cos\phi}{8}\right)V_{CE0}I_{pk} + \left(\frac{1}{8} + \frac{M_{eff}\cos\phi}{3\pi}\right)R_{tot,IGBT}I_{pk}^2")
    st.latex(r"P_{cond,D} = \left(\frac{1}{2\pi} - \frac{M_{eff}\cos\phi}{8}\right)V_{F0}I_{pk} + \left(\frac{1}{8} - \frac{M_{eff}\cos\phi}{3\pi}\right)R_{tot,D}I_{pk}^2")
    st.latex(r"P_{cond,SiC}^{SPWM} = \left(\frac{1}{8} + \frac{M_{eff}\cos\phi}{3\pi}\right)R_{tot,SiC}I_{pk}^2")
    st.markdown("#### 3.2 SVPWM")
    st.latex(r"P_{cond,IGBT} \approx \frac{M_{eff}\cos\phi}{4}V_{CE0}I_{pk} + \left(\frac{24\cos\phi - 2\sqrt{3}\cos(2\varphi) - 3\sqrt{3}}{24\pi}\right)R_{tot,IGBT}I_{pk}^2")
    st.latex(r"P_{cond,D} \approx \left(\frac{4 - M_{eff}\pi\cos\phi}{4\pi}\right)V_{F0}I_{pk} + \left(\frac{6\pi - 24M_{eff}\cos\phi + 2\sqrt{3}M_{eff}\cos(2\varphi) + 3\sqrt{3}M_{eff}}{24\pi}\right)R_{tot,D}I_{pk}^2")
    st.latex(r"P_{cond,SiC}^{SVPWM} \approx \left(\frac{24\cos\phi - 2\sqrt{3}\cos(2\varphi) - 3\sqrt{3}}{24\pi}\right)R_{tot,SiC}I_{pk}^2")
    st.latex(r"R_{tot} = R_{dynamic} + R_{pkg,chip} + R_{arm,eq}")
    st.latex(r"R_{arm,eq} = R_{arm}\cdot N_{sim}")

    st.markdown("### 📕 4. 死区时间补偿")
    st.latex(r"D_{dead} = 2\cdot t_{dead}\cdot f_{sw}")
    st.latex(r"M_{eff} = M - sgn(i)\cdot K_{mode}\cdot D_{dead}")
    st.latex(r"K_{mode} = \frac{4}{\pi}\;(\mathrm{SVPWM}),\quad \frac{2}{\pi}\;(\mathrm{SPWM})")
    st.latex(r"P_{dead,main}^{avg} = \frac{2}{\pi}V_{0,main}I_{pk} + \frac{1}{2}R_{tot,main}I_{pk}^2")
    st.latex(r"P_{dead,diode}^{avg} = \frac{2}{\pi}V_{0,diode}I_{pk} + \frac{1}{2}R_{tot,diode}I_{pk}^2")
    st.write("程序会同步修正有效调制系数和主开关 / 二极管导通路径重分配。")

    st.markdown("### 📔 5. 开关损耗")
    st.latex(r"E_{base} = E(I_{pk},T_j)\quad \text{或}\quad E(I_{nom},T_j)\cdot \frac{I_{pk}}{I_{nom}}")
    st.latex(r"E_{adj} = E_{base}\cdot \left(\frac{R_{g,act}}{R_{g,ref}}\right)^{K_r}\cdot \left(\frac{V_{dc}}{V_{nom}}\right)^{K_v}")
    st.latex(r"\mathrm{If}\;|\mathcal{T}_{table}| \ge 2,\quad T_{coeff,eff}=0")
    st.latex(r"P_{sw,main} = \frac{f_{sw}}{\pi}(E_{on,adj}+E_{off,adj})")
    st.latex(r"P_{sw,frd} = \frac{f_{sw}}{\pi}E_{rec,adj}\cdot I_{corr}")
    st.write("E-I 表一旦具备多个温度维度，经验温漂系数就会被自动锁死为 0；同时支持“CAE精确二维插值”和“标称点直线比例法”两种能量提取路径。")

    st.markdown("### 📓 6. 堵转极限工况")
    st.latex(r"D_{max} = \frac{1 + M_{eff}}{2}")
    st.latex(r"P_{cond,stall} = D_{max}(V_0I_{pk}+RI_{pk}^2)")
    st.latex(r"P_{cond,stall}^{SiC} = D_{max}R_{tot,SiC}I_{pk}^2")
    st.latex(r"P_{sw,stall} = f_{sw}\cdot E_{adj,total}(I_{pk})")

    st.markdown("### 📒 7. 闭环热迭代")
    st.latex(r"T_{j,main}^{k+1} = T_{c,main} + P_{main,chip}^{k}\cdot R_{thJC,main}")
    st.latex(r"T_{j,diode}^{k+1} = T_{c,diode} + P_{diode,chip}^{k}\cdot R_{thJC,diode}")
    st.latex(r"T_{j,control} = \max(T_{j,main}, T_{j,diode})")
    st.write("闭环模式下至少完成 15 次迭代后，才允许按收敛阈值提前退出。")

    st.markdown("### 📑 8. 数据边界监视")
    st.latex(r"\mathrm{Flag}_{extra} = \mathbb{I}(I_{target}\notin[I_{min},I_{max}] \;\vee\; T_{target}\notin[T_{min},T_{max}])")
    st.write("一旦当前工作点超出原始矩阵边界，结果页就会显式报警。")

    st.markdown("### 📐 9. STAR-CCM+ 总热源映射")
    st.latex(r"P_{chip,total} \;[\mathrm{W}]")
    st.write("当前 STAR-CCM+ 主输出按单颗总热源 Total Heat Source (W) 组织，方便直接映射到各个实体 region。")

    st.markdown("### 🧱 10. 工程约定（必须长期保留）")
    st.markdown(
        """
        - `R_pkg,chip` 默认 `0 mΩ`：只有当原始 V-I 数据没有吸收封装压降时才额外填写。
        - `R_arm` 默认 `0 mΩ`：只有当需要单独补偿外部铜排 / 连接片损耗时才填写。
        - `Regeneration` 用工况开关控制，不要靠把 `cos_phi` 输成负值来冒充反拖。
        - `STAR-CCM+` 当前主输出按 **Total Heat Source (W)** 组织；系统总量通过 `N_arm_sys` 对称桥臂缩放得到。
        """
    )
