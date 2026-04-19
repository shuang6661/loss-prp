import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD系统级电热仿真-物理对齐版", layout="wide")

# =============================================================================
# 默认填充数据
# =============================================================================
DEFAULT_MAIN_VI = pd.DataFrame({"Temp (℃)": [25, 150, 25, 150], "Current (A)": [100.0, 100.0, 600.0, 600.0], "V_drop (V)": [1.10, 1.05, 2.20, 2.50]})
DEFAULT_MAIN_EI = pd.DataFrame({"Temp (℃)": [25, 150, 25, 150], "Current (A)": [100.0, 100.0, 600.0, 600.0], "Eon (mJ)": [5.9, 8.5, 70.0, 95.0], "Eoff (mJ)": [4.9, 7.2, 45.0, 60.0]})
DEFAULT_DIODE_VI = pd.DataFrame({"Temp (℃)": [25, 150, 25, 150], "Current (A)": [100.0, 100.0, 600.0, 600.0], "Vf (V)": [1.20, 1.10, 2.00, 2.20]})
DEFAULT_DIODE_EI = pd.DataFrame({"Temp (℃)": [25, 150, 25, 150], "Current (A)": [100.0, 100.0, 600.0, 600.0], "Erec (mJ)": [1.9, 3.5, 15.0, 25.0]})

TEMP_COL = "Temp (degC)"
CURRENT_COL = "Current (A)"

def clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))

def canonicalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        col_str = str(col)
        if "Temp" in col_str or "temp" in col_str: rename_map[col] = TEMP_COL
        elif "Current" in col_str or col_str.strip().lower() in {"ic (a)", "if (a)", "current"}: rename_map[col] = CURRENT_COL
    return df.rename(columns=rename_map)

def validate_numeric_table(df: pd.DataFrame, table_name: str, required_cols: list[str]):
    cleaned = canonicalize_df_columns(df.copy()).replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
    missing_cols = [col for col in required_cols if col not in cleaned.columns]
    if missing_cols: return cleaned, [f"{table_name} 缺少必要列：{', '.join(missing_cols)}"], []
    raw_required = cleaned[required_cols].copy()
    numeric_required = raw_required.apply(pd.to_numeric, errors="coerce")
    cleaned[required_cols] = numeric_required
    cleaned = cleaned.dropna(subset=required_cols).sort_values([TEMP_COL, CURRENT_COL]).reset_index(drop=True)
    return cleaned, [], []

def normalize_vi_df(df: pd.DataFrame, n_src: int) -> pd.DataFrame:
    res_df = df.copy()
    if n_src > 1: res_df[CURRENT_COL] = res_df[CURRENT_COL] / float(n_src)
    return res_df

def normalize_ei_df(df: pd.DataFrame, n_src: int, e_cols: list[str]) -> pd.DataFrame:
    res_df = df.copy()
    if n_src > 1:
        res_df[CURRENT_COL] = res_df[CURRENT_COL] / float(n_src)
        for col in e_cols:
            if col in res_df.columns: res_df[col] = res_df[col] / float(n_src)
    return res_df

def safe_interp(df: pd.DataFrame, target_i: float, target_t: float, item_name: str) -> float:
    clean_df = canonicalize_df_columns(df.dropna())
    if clean_df.empty or item_name not in clean_df.columns: return 0.0
    temp_list, val_list = [], []
    for temp, group in clean_df.groupby(TEMP_COL):
        sorted_group = group.sort_values(CURRENT_COL)
        if len(sorted_group) >= 2:
            func = interp1d(sorted_group[CURRENT_COL], sorted_group[item_name], kind="linear", fill_value="extrapolate")
            val_list.append(max(0.0, float(func(target_i))))
            temp_list.append(float(temp))
        elif len(sorted_group) == 1:
            val_list.append(max(0.0, float(sorted_group[item_name].iloc[0])))
            temp_list.append(float(temp))

    if len(temp_list) >= 2: return max(0.0, float(interp1d(temp_list, val_list, kind="linear", fill_value="extrapolate")(target_t)))
    elif len(temp_list) == 1: return max(0.0, float(val_list[0]))
    return 0.0

# ================= 核心修复：根据邻近点动态拟合 =================
def get_bracketing_points(i_list, target_i):
    """找到包含 target_i 的最近两个表电流点"""
    i_list = sorted(list(set(i_list)))
    if len(i_list) < 2:
        return (i_list[0], i_list[0]) if i_list else (1e-6, 1e-6)
    if target_i <= i_list[0]:
        return i_list[0], i_list[1]
    if target_i >= i_list[-1]:
        return i_list[-2], i_list[-1]
    for k in range(len(i_list)-1):
        if i_list[k] <= target_i <= i_list[k+1]:
            return i_list[k], i_list[k+1]
    return i_list[0], i_list[1]

def build_linearized_device_model(df: pd.DataFrame, target_i: float, target_t: float, item_name: str, force_zero_intercept: bool):
    """
    【完全对标要求】：不再使用 0 和 Ipk 拉直线，而是寻找最近的两个表电流来计算 Req 和 V0
    """
    clean_df = canonicalize_df_columns(df.dropna())
    i_list = clean_df[CURRENT_COL].unique() if not clean_df.empty else []
    
    # 动态获取距离 Iout(Ipk) 最近的拟合点区间
    i_low, i_high = get_bracketing_points(i_list, target_i)

    v_low = safe_interp(df, i_low, target_t, item_name)
    v_high = safe_interp(df, i_high, target_t, item_name)

    denom = i_high - i_low
    r_eq = max(0.0, (v_high - v_low) / denom) if denom > 1e-12 else 0.0
    v0 = 0.0 if force_zero_intercept else max(0.0, v_low - r_eq * i_low)

    return {"v_pk": v_high, "v_half": v_low, "r_eq": r_eq, "v0": v0, "i_low": i_low, "i_high": i_high}

def calc_switching_energy(df: pd.DataFrame, i_pk: float, tj: float, algo_type: str, item_name: str, vdc: float, vref: float, kv: float, ract: float, rref: float, kr: float, temp_coeff: float, tref: float) -> dict:
    """
    【强制温漂逻辑】：解除一切智能锁，只要 temp_coeff > 0，就必定执行温漂公式
    """
    clean_df = canonicalize_df_columns(df.dropna())
    i_list = clean_df[CURRENT_COL].unique() if not clean_df.empty else []
    
    if "比例法" in algo_type and len(i_list) > 0:
        # 寻找最近的一个点按比例放缩
        i_closest = min(i_list, key=lambda x: abs(x - i_pk))
        e_nom = safe_interp(df, i_closest, tj, item_name)
        e_base = e_nom * (max(float(i_pk), 0.0) / max(i_closest, 1e-6))
        extraction_label = f"最近点比例法 (基准 {i_closest:.1f}A)"
        e_nom_mj = e_nom
    else:
        # CAE两点线性拟合（自带了最近点特征）
        e_base = safe_interp(df, i_pk, tj, item_name)
        i_low, i_high = get_bracketing_points(i_list, i_pk)
        extraction_label = f"就近两点插值 ({i_low:.1f}A~{i_high:.1f}A)"
        e_nom_mj = np.nan

    # === 强行执行温漂放大 ===
    temp_correction = max(0.0, 1.0 + float(temp_coeff) * (tj - tref))
    rg_correction = math.pow(max(ract, 1e-12) / max(rref, 1e-12), kr) if rref > 0 else 1.0
    voltage_correction = math.pow(max(vdc, 1e-12) / max(vref, 1e-12), kv) if vref > 0 else 1.0
    energy_mj = max(0.0, e_base * temp_correction * rg_correction * voltage_correction)

    return {"energy_mj": energy_mj, "e_base_mj": max(0.0, float(e_base)), "e_nom_mj": e_nom_mj, "temp_correction": temp_correction, "rg_correction": rg_correction, "voltage_correction": voltage_correction, "effective_temp_coeff": float(temp_coeff), "extraction_label": extraction_label}
# ================================================================

def calc_dead_time_compensation(mode: str, fsw: float, dead_time_us: float, m_index: float, current_sign: float, vdc: float):
    if fsw <= 0.0 or dead_time_us <= 0.0: return {"dead_ratio": 0.0, "m_eff": m_index, "current_sign": current_sign, "phase_voltage_error_v": 0.0, "modulation_gain": 0.0}
    dead_time_s = dead_time_us * 1e-6
    dead_ratio = clamp(2.0 * dead_time_s * fsw, 0.0, 0.20)
    modulation_gain = 4.0 / math.pi if mode == "SVPWM" else 2.0 / math.pi
    m_eff = clamp(m_index - current_sign * modulation_gain * dead_ratio, 0.0, 1.15)
    return {"dead_ratio": dead_ratio, "m_eff": m_eff, "current_sign": current_sign, "phase_voltage_error_v": 0.5 * vdc * dead_ratio * current_sign, "modulation_gain": modulation_gain}

def calc_pwm_conduction_losses(mode: str, m_eff: float, active_cosphi: float, theta: float, i_pk_chip: float, main_model: dict, diode_model: dict, r_pkg_chip: float, r_arm_chip: float, dead_meta: dict):
    r_main_total = main_model["r_eq"] + r_pkg_chip + r_arm_chip
    r_diode_total = diode_model["r_eq"] + r_pkg_chip + r_arm_chip

    if mode == "SVPWM":
        kv0_m = (m_eff * active_cosphi) / 4.0
        kr_m = m_eff * (24.0 * active_cosphi - 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) - 3.0 * math.sqrt(3.0)) / 24.0
        kv0_d = (4.0 - m_eff * math.pi * active_cosphi) / 4.0
        kr_d = m_eff * (6.0 * math.pi - 24.0 * active_cosphi + 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) + 3.0 * math.sqrt(3.0)) / 24.0
        p_cond_main = (kv0_m * main_model["v0"] * i_pk_chip) + (kr_m * r_main_total * i_pk_chip**2) / math.pi
        p_cond_diode = (kv0_d * diode_model["v0"] * i_pk_chip) / math.pi + (kr_d * r_diode_total * i_pk_chip**2) / math.pi
    else:
        p_cond_main = main_model["v0"] * i_pk_chip * (1.0 / (2.0 * math.pi) + m_eff * active_cosphi / 8.0) + r_main_total * i_pk_chip**2 * (1.0 / 8.0 + m_eff * active_cosphi / (3.0 * math.pi))
        p_cond_diode = diode_model["v0"] * i_pk_chip * (1.0 / (2.0 * math.pi) - m_eff * active_cosphi / 8.0) + r_diode_total * i_pk_chip**2 * (1.0 / 8.0 - m_eff * active_cosphi / (3.0 * math.pi))

    p_cond_main = max(0.0, float(p_cond_main))
    p_cond_diode = max(0.0, float(p_cond_diode))

    dead_ratio = dead_meta["dead_ratio"]
    if dead_ratio > 0.0:
        avg_inst_main = (main_model["v0"] * i_pk_chip * 2.0 / math.pi) + (r_main_total * i_pk_chip**2 * 0.5)
        avg_inst_diode = (diode_model["v0"] * i_pk_chip * 2.0 / math.pi) + (r_diode_total * i_pk_chip**2 * 0.5)
        if dead_meta["current_sign"] >= 0.0:
            p_cond_main = max(0.0, p_cond_main - dead_ratio * avg_inst_main)
            p_cond_diode += dead_ratio * avg_inst_diode
        else:
            p_cond_diode = max(0.0, p_cond_diode - dead_ratio * avg_inst_diode)
            p_cond_main += dead_ratio * avg_inst_main

    return {"p_cond_main": p_cond_main, "p_cond_diode": p_cond_diode, "r_main_total": r_main_total, "r_diode_total": r_diode_total}

def calc_stall_losses(m_eff: float, i_pk_chip: float, main_model: dict, diode_model: dict, r_pkg_chip: float, r_arm_chip: float, dead_meta: dict):
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
        p_cond_diode += dead_ratio * inst_diode

    return {"p_cond_main": p_cond_main, "p_cond_diode": p_cond_diode, "r_main_total": r_main_total, "r_diode_total": r_diode_total, "d_max": d_max}

def simulate_system(inputs: dict, tables: dict):
    cond_src_count = inputs["n_src_cond"] if "Module" in inputs["cond_data_type"] else 1
    sw_src_count = inputs["n_src_sw"] if "Module" in inputs["sw_data_type"] else 1

    norm_ev_m = normalize_vi_df(tables["ev_main"], cond_src_count)
    norm_ev_d = normalize_vi_df(tables["ev_diode"], cond_src_count)
    norm_ee_m = normalize_ei_df(tables["ee_main"], sw_src_count, ["Eon (mJ)", "Eoff (mJ)"])
    norm_ee_d = normalize_ei_df(tables["ee_diode"], sw_src_count, ["Erec (mJ)"])

    i_pk_chip = math.sqrt(2.0) * (inputs["iout_rms"] / inputs["n_sim"]) if inputs["n_sim"] > 0 else 0.0
    r_arm_chip = (inputs["r_arm_mohm"] / 1000.0) * inputs["n_sim"]
    r_pkg_chip = inputs["r_pkg_mohm"] / 1000.0

    active_cosphi = -abs(inputs["cosphi"]) if "Regeneration" in inputs["op_mode"] else abs(inputs["cosphi"])
    active_cosphi = clamp(active_cosphi, -1.0, 1.0)
    theta = math.acos(active_cosphi) if abs(active_cosphi) <= 1.0 else 0.0
    current_sign = -1.0 if "
