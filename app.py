import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD系统级电热仿真-完全体", layout="wide")

# =============================================================================
# 设计守则（请长期保留，后续迭代不要再删）
# 1. 三模态工况必须同时保留：Motoring / Regeneration / Stall。
# 2. 结果必须支持热源分离：主芯片与二极管单颗发热率都要显式输出。
# 3. 原始数据无论来自单芯还是模块，必须先归一化为单芯模型再做扩容评估。
# 4. SVPWM / SPWM 计算必须显式包含死区时间补偿，不能把死区误差藏在黑盒里。
# 5. 工程文档、公式、关键假设、易错参数提醒必须直接展示在界面上。
# 6. 核心公式处必须用详细中文注释说明，便于后续工程师接手维护。
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


def clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def validate_numeric_table(df: pd.DataFrame, table_name: str, required_cols: list[str]):
    """
    对用户粘贴进 data_editor 的原始表格做工程化清洗。

    为什么要单独做这一步：
    1. Streamlit 的表格输入会混入空字符串、None、object 类型。
    2. 插值要求数值列必须是干净的浮点数。
    3. 规格书复制时很容易把同一个温度/电流点贴两次，这里统一做去重平均。
    """
    cleaned = df.copy().replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
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
            bad_values = raw_required.loc[bad_mask, col].astype(str).unique()[:3]
            errors.append(f"{table_name} 的列 {col} 存在非数字内容，例如：{', '.join(bad_values)}")

    cleaned[required_cols] = numeric_required
    cleaned = cleaned.dropna(subset=required_cols)

    if cleaned.empty:
        errors.append(f"{table_name} 没有可用于计算的有效数值行。")
        return cleaned, errors, warnings

    if (cleaned["Current (A)"] < 0).any():
        errors.append(f"{table_name} 中的 Current (A) 不能为负数。")
    if (cleaned["Temp (℃)"] < -273.15).any():
        errors.append(f"{table_name} 中的 Temp (℃) 低于绝对零度，请检查录入。")

    value_cols = [col for col in required_cols if col not in ("Temp (℃)", "Current (A)")]
    for value_col in value_cols:
        if (cleaned[value_col] < 0).any():
            errors.append(f"{table_name} 中的 {value_col} 不能为负数。")

    duplicate_count = cleaned.duplicated(subset=["Temp (℃)", "Current (A)"]).sum()
    if duplicate_count:
        cleaned = cleaned.groupby(["Temp (℃)", "Current (A)"], as_index=False)[value_cols].mean()
        warnings.append(f"{table_name} 存在 {duplicate_count} 组重复温度-电流点，程序已自动取平均值。")

    cleaned = cleaned.sort_values(["Temp (℃)", "Current (A)"]).reset_index(drop=True)

    point_counts = cleaned.groupby("Temp (℃)")["Current (A)"].nunique()
    sparse_temps = [str(temp) for temp, count in point_counts.items() if count < 2]
    if sparse_temps:
        warnings.append(
            f"{table_name} 在温度 {', '.join(sparse_temps)} ℃ 仅有 1 个电流点，程序将退化为单点外推。"
        )

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
    if inputs["vdc_act"] < 0:
        errors.append("母线电压 V_dc 不能为负数。")
    if inputs["iout_rms"] < 0:
        errors.append("有效值 I_out 不能为负数。")
    if inputs["fsw"] < 0:
        errors.append("开关频率 f_sw 不能为负数。")
    if inputs["fout"] < 0:
        errors.append("输出频率 f_out 不能为负数。")
    if not 0.0 <= inputs["m_index"] <= 1.15:
        errors.append("调制系数 M 建议位于 0.0 到 1.15 之间。")
    if not 0.0 <= inputs["cosphi"] <= 1.0:
        errors.append("功率因数 cos_phi 这里按幅值录入，建议位于 0 到 1 之间。")
    if inputs["rg_on_ref"] < 0 or inputs["rg_off_ref"] < 0 or inputs["rg_on_act"] < 0 or inputs["rg_off_act"] < 0:
        errors.append("门极电阻参数不能为负数。")
    if inputs["r_pkg_mohm"] < 0:
        errors.append("封装内阻 R_pkg,chip 不能为负数。")
    if inputs["r_arm_mohm"] < 0:
        errors.append("桥臂附加电阻 R_arm 不能为负数。")
    if inputs["dead_time_us"] < 0:
        errors.append("死区时间 t_dead 不能为负数。")
    if "闭环" in inputs["sim_mode"]:
        if inputs["rth_jc_main"] < 0:
            errors.append("闭环模式下，主芯片热阻 RthJC_main 不能为负数。")
        if inputs["rth_jc_diode"] < 0:
            errors.append("闭环模式下，二极管热阻 RthJC_diode 不能为负数。")
    if inputs["main_die_length_mm"] <= 0 or inputs["main_die_width_mm"] <= 0 or inputs["main_die_thickness_mm"] <= 0:
        errors.append("STAR-CCM+ 主芯片几何尺寸必须大于 0。")
    if inputs["diode_die_length_mm"] <= 0 or inputs["diode_die_width_mm"] <= 0 or inputs["diode_die_thickness_mm"] <= 0:
        errors.append("STAR-CCM+ 二极管几何尺寸必须大于 0。")

    if inputs["fsw"] == 0:
        warnings.append("当前 f_sw = 0 Hz，开关损耗会被计算为 0。")
    if inputs["iout_rms"] == 0:
        warnings.append("当前 I_out = 0 A，结果会趋近空载。")
    if inputs["r_pkg_mohm"] == 0 and inputs["r_arm_mohm"] == 0:
        warnings.append("当前默认未额外计入封装内阻和外部电路电阻，等价假设这些寄生压降已包含在原始 V-I 数据中。")
    if inputs["dead_time_us"] > 0 and inputs["fsw"] > 0:
        dead_ratio = 2.0 * inputs["dead_time_us"] * 1e-6 * inputs["fsw"]
        if dead_ratio > 0.15:
            warnings.append("死区时间相对于开关周期偏大，死区补偿会显著改变有效占空比，请核对驱动设置。")

    return errors, warnings


def describe_temperature_strategy(df: pd.DataFrame, temp_coeff: float) -> dict:
    """
    “防双重放大锁”的显式判定器。

    规则非常明确：
    - 如果表格里已经有 2 个及以上温度维度，就使用二维曲面插值，经验温漂系数视为锁死为 0；
    - 如果表格只有 1 个温度维度，才允许启用经验温漂系数外推。
    """
    unique_temps = sorted(float(t) for t in df["Temp (℃)"].dropna().unique()) if "Temp (℃)" in df.columns else []
    multi_temp = len(unique_temps) >= 2
    effective_temp_coeff = 0.0 if multi_temp else float(temp_coeff)

    return {
        "unique_temps": unique_temps,
        "multi_temp": multi_temp,
        "effective_temp_coeff": effective_temp_coeff,
        "strategy_label": "二维矩阵曲面插值（经验温漂锁死为 0）" if multi_temp else "单温度查表 + 线性温漂外推",
    }


def assess_interp_usage(df: pd.DataFrame, table_name: str, target_i: float, target_t: float) -> dict:
    """
    判断当前工作点是否已经超出数据矩阵边界。

    工程上最怕的一种情况是：程序虽然能算出数值，但其实已经在高电流或高温处做外推，
    而使用者误以为那是“规格书直接覆盖”的结果。所以这里必须显式留痕。
    """
    current_min = float(df["Current (A)"].min()) if "Current (A)" in df.columns and not df.empty else np.nan
    current_max = float(df["Current (A)"].max()) if "Current (A)" in df.columns and not df.empty else np.nan
    temp_min = float(df["Temp (℃)"].min()) if "Temp (℃)" in df.columns and not df.empty else np.nan
    temp_max = float(df["Temp (℃)"].max()) if "Temp (℃)" in df.columns and not df.empty else np.nan

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
    """
    给工程师一个明确的“数据矩阵健康度”回显。
    这样就不会出现“以为自己在做二维插值，实际上只输了一组温度”的误判。
    """
    rows = []

    table_specs = [
        ("主开关导通表", tables["ev_main"], inputs["cond_data_type"], inputs["n_src_cond"] if "Module" in inputs["cond_data_type"] else 1, "纯二维 V-I 曲面插值", None),
        ("主开关开关能量表", tables["ee_main"], inputs["sw_data_type"], inputs["n_src_sw"] if "Module" in inputs["sw_data_type"] else 1, None, inputs["t_coeff_igbt"]),
        ("二极管导通表", tables["ev_diode"], inputs["cond_data_type"], inputs["n_src_cond"] if "Module" in inputs["cond_data_type"] else 1, "纯二维 V-I 曲面插值", None),
        ("二极管恢复能量表", tables["ee_diode"], inputs["sw_data_type"], inputs["n_src_sw"] if "Module" in inputs["sw_data_type"] else 1, None, inputs["t_coeff_frd"]),
    ]

    for table_name, df, source_type, src_count, fixed_strategy, temp_coeff in table_specs:
        temp_count = int(df["Temp (℃)"].nunique()) if "Temp (℃)" in df.columns and not df.empty else 0
        current_count = int(df["Current (A)"].nunique()) if "Current (A)" in df.columns and not df.empty else 0
        current_min = float(df["Current (A)"].min()) if "Current (A)" in df.columns and not df.empty else np.nan
        current_max = float(df["Current (A)"].max()) if "Current (A)" in df.columns and not df.empty else np.nan

        if fixed_strategy is not None:
            strategy_label = fixed_strategy
            effective_temp_coeff = 0.0
        else:
            strategy_meta = describe_temperature_strategy(df, temp_coeff if temp_coeff is not None else 0.0)
            strategy_label = strategy_meta["strategy_label"]
            effective_temp_coeff = strategy_meta["effective_temp_coeff"]

        rows.append(
            {
                "数据表": table_name,
                "原始来源": source_type,
                "归一化并联数 N_src": src_count,
                "温度维数": temp_count,
                "电流维数": current_count,
                "电流范围 (A)": f"{current_min:.2f} ~ {current_max:.2f}" if np.isfinite(current_min) and np.isfinite(current_max) else "N/A",
                "插值/温漂策略": strategy_label,
                "生效温漂系数": effective_temp_coeff,
            }
        )

    return pd.DataFrame(rows)


def normalize_vi_df(df: pd.DataFrame, n_src: int) -> pd.DataFrame:
    """
    导通表归一化逻辑：
    - 如果原始曲线来自模块端子，则同一时刻承受的是模块总电流。
    - 为了构建标准“单芯模型”，必须把电流轴除以并联芯片数。
    - 电压轴不除，因为并联芯片两端电压相同。
    """
    res_df = df.copy()
    if n_src > 1:
        res_df["Current (A)"] = res_df["Current (A)"] / float(n_src)
    return res_df


def normalize_ei_df(df: pd.DataFrame, n_src: int, e_cols: list[str]) -> pd.DataFrame:
    """
    开关能量表归一化逻辑：
    - 模块双脉冲测试给出的 Eon/Eoff/Erec 往往是整臂/整模块总和。
    - 归一化到单芯时，电流轴要除以 n_src，能量轴也要除以 n_src。
    """
    res_df = df.copy()
    if n_src > 1:
        res_df["Current (A)"] = res_df["Current (A)"] / float(n_src)
        for col in e_cols:
            if col in res_df.columns:
                res_df[col] = res_df[col] / float(n_src)
    return res_df


def safe_interp(df: pd.DataFrame, target_i: float, target_t: float, item_name: str) -> float:
    """
    二维安全插值：
    1. 先在每个温度平面内按电流一维插值；
    2. 再在温度维度做二次插值；
    3. 数据不足时自动退化为单点/单温度外推。
    """
    clean_df = df.dropna()
    if clean_df.empty or item_name not in clean_df.columns:
        return 0.0

    if not np.isfinite(target_i) or not np.isfinite(target_t):
        return 0.0

    temp_list: list[float] = []
    val_list: list[float] = []

    for temp, group in clean_df.groupby("Temp (℃)"):
        sorted_group = group.sort_values("Current (A)")
        if len(sorted_group) >= 2:
            func = interp1d(
                sorted_group["Current (A)"],
                sorted_group[item_name],
                kind="linear",
                fill_value="extrapolate",
            )
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
):
    """
    用两点线性化把导通曲线拆成 V0 + R * I。

    这里继续沿用你原始版本的工程思路：
    - 在 Ipk 与 Ipk/2 两个工作点上取样；
    - 用斜率估算等效动态电阻；
    - 用截距估算 Vce0 / Vf0；
    - SiC 主开关强制 V0 = 0，仅保留纯阻性项。
    """
    target_i = max(float(target_i), 1e-6)
    half_i = max(target_i / 2.0, 1e-6)

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
    item_name: str,
    vdc: float,
    vref: float,
    kv: float,
    ract: float,
    rref: float,
    kr: float,
    temp_coeff: float,
    tref: float,
) -> dict:
    """
    开关能量修正逻辑：
    1. 先根据当前芯片峰值电流和结温插值得到基础能量；
    2. 再叠加门极电阻修正；
    3. 再叠加电压指数修正；
    4. 如果原始表格本身没有温度维度，再用线性温漂系数补偿。
    """
    strategy_meta = describe_temperature_strategy(df, temp_coeff)
    e_base = safe_interp(df, i_pk, tj, item_name)
    if e_base <= 0.0:
        return {
            "energy_mj": 0.0,
            "e_base_mj": 0.0,
            "temp_correction": 1.0,
            "rg_correction": 1.0,
            "voltage_correction": 1.0,
            "effective_temp_coeff": strategy_meta["effective_temp_coeff"],
            "strategy_label": strategy_meta["strategy_label"],
            "temp_lock_active": strategy_meta["multi_temp"],
        }

    temp_correction = (
        1.0
        if strategy_meta["multi_temp"]
        else max(0.0, 1.0 + strategy_meta["effective_temp_coeff"] * (tj - tref))
    )
    rg_correction = math.pow(max(ract, 1e-12) / max(rref, 1e-12), kr) if rref > 0 else 1.0
    voltage_correction = math.pow(max(vdc, 1e-12) / max(vref, 1e-12), kv) if vref > 0 else 1.0

    return {
        "energy_mj": max(0.0, e_base * temp_correction * rg_correction * voltage_correction),
        "e_base_mj": max(0.0, e_base),
        "temp_correction": temp_correction,
        "rg_correction": rg_correction,
        "voltage_correction": voltage_correction,
        "effective_temp_coeff": strategy_meta["effective_temp_coeff"],
        "strategy_label": strategy_meta["strategy_label"],
        "temp_lock_active": strategy_meta["multi_temp"],
    }


def calc_dead_time_compensation(mode: str, fsw: float, dead_time_us: float, m_index: float, current_sign: float, vdc: float):
    """
    死区时间补偿逻辑（显式保留，不允许隐藏）：

    对于单相桥臂，一个开关周期内存在两次死区空窗。
    等效占空比损失近似为：
        D_dead = 2 * t_dead * f_sw

    再将其投影到等效调制系数上：
        M_eff = M - sgn(i) * K_mode * D_dead

    这里 K_mode 取：
    - SVPWM: 4/pi
    - SPWM : 2/pi

    同时把死区造成的导通通道重分配显式加入损耗：
    - 正扭矩 Motoring: 默认从主开关导通时间转移到二极管
    - 反拖 Regeneration: 默认从二极管导通时间转移到主开关
    """
    if fsw <= 0.0 or dead_time_us <= 0.0:
        return {
            "dead_time_s": 0.0,
            "dead_ratio": 0.0,
            "modulation_gain": 0.0,
            "delta_m": 0.0,
            "m_eff": m_index,
            "phase_voltage_error_v": 0.0,
            "current_sign": current_sign,
        }

    dead_time_s = dead_time_us * 1e-6
    dead_ratio = clamp(2.0 * dead_time_s * fsw, 0.0, 0.20)
    modulation_gain = 4.0 / math.pi if mode == "SVPWM" else 2.0 / math.pi
    delta_m = current_sign * modulation_gain * dead_ratio
    m_eff = clamp(m_index - delta_m, 0.0, 1.15)

    return {
        "dead_time_s": dead_time_s,
        "dead_ratio": dead_ratio,
        "modulation_gain": modulation_gain,
        "delta_m": delta_m,
        "m_eff": m_eff,
        "phase_voltage_error_v": 0.5 * vdc * dead_ratio * current_sign,
        "current_sign": current_sign,
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
):
    """
    PWM 工况下的导通损耗。

    这里保留你原始手稿里的 SPWM / SVPWM 解析公式，再额外补上两个工程增强：
    1. 把封装内阻 R_pkg 与桥臂附加电阻 R_arm 显式串联进 R 项；
    2. 把死区时间造成的导通通道重分配显式计入，而不是假装没有死区。
    """
    r_main_total = main_model["r_eq"] + r_pkg_chip + r_arm_chip
    r_diode_total = diode_model["r_eq"] + r_pkg_chip + r_arm_chip

    inst_main = main_model["v0"] * i_pk_chip + r_main_total * i_pk_chip**2
    inst_diode = diode_model["v0"] * i_pk_chip + r_diode_total * i_pk_chip**2

    if mode == "SVPWM":
        kv0_m = (m_eff * active_cosphi) / 4.0
        kr_m = (24.0 * active_cosphi - 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) - 3.0 * math.sqrt(3.0)) / 24.0
        kv0_d = (4.0 - m_eff * math.pi * active_cosphi) / 4.0
        kr_d = (
            6.0 * math.pi
            - 24.0 * m_eff * active_cosphi
            + 2.0 * math.sqrt(3.0) * m_eff * math.cos(2.0 * theta)
            + 3.0 * math.sqrt(3.0) * m_eff
        ) / 24.0

        p_cond_main = (kv0_m * main_model["v0"] * i_pk_chip) + (kr_m * r_main_total * i_pk_chip**2) / math.pi
        p_cond_diode = (kv0_d * diode_model["v0"] * i_pk_chip) / math.pi + (kr_d * r_diode_total * i_pk_chip**2) / math.pi
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

    # 死区时间导致导通路径在器件之间切换，这里显式把该项加回去。
    dead_ratio = dead_meta["dead_ratio"]
    dead_shift_from_main = 0.0
    dead_shift_from_diode = 0.0

    if dead_ratio > 0.0:
        if dead_meta["current_sign"] >= 0.0:
            dead_shift_from_main = min(p_cond_main, dead_ratio * inst_main)
            p_cond_main = max(0.0, p_cond_main - dead_shift_from_main)
            p_cond_diode = p_cond_diode + dead_ratio * inst_diode
        else:
            dead_shift_from_diode = min(p_cond_diode, dead_ratio * inst_diode)
            p_cond_diode = max(0.0, p_cond_diode - dead_shift_from_diode)
            p_cond_main = p_cond_main + dead_ratio * inst_main

    return {
        "p_cond_main": p_cond_main,
        "p_cond_diode": p_cond_diode,
        "inst_main": inst_main,
        "inst_diode": inst_diode,
        "dead_shift_from_main": dead_shift_from_main,
        "dead_shift_from_diode": dead_shift_from_diode,
        "r_main_total": r_main_total,
        "r_diode_total": r_diode_total,
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
    堵转工况不再走正弦平均公式，而是按最大占空比近似成“直流重载轰炸”。
    这正是很多极限热设计里最保守、最有意义的工况。
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
        dead_shift_from_main = min(p_cond_main, dead_ratio * inst_main)
        p_cond_main = max(0.0, p_cond_main - dead_shift_from_main)
        p_cond_diode = p_cond_diode + dead_ratio * inst_diode
    else:
        dead_shift_from_main = 0.0

    return {
        "p_cond_main": p_cond_main,
        "p_cond_diode": p_cond_diode,
        "inst_main": inst_main,
        "inst_diode": inst_diode,
        "dead_shift_from_main": dead_shift_from_main,
        "dead_shift_from_diode": 0.0,
        "r_main_total": r_main_total,
        "r_diode_total": r_diode_total,
        "d_max": d_max,
    }


def build_icepak_heat_table(n_sim: int, p_main_chip: float, p_diode_chip: float) -> pd.DataFrame:
    rows = []
    for idx in range(1, n_sim + 1):
        rows.append(
            {
                "Region": f"MainChip_{idx}",
                "Category": "Main Switch",
                "Heat Generation Rate (W)": round(p_main_chip, 6),
                "Count Basis": "Single Die",
            }
        )
        rows.append(
            {
                "Region": f"DiodeChip_{idx}",
                "Category": "Freewheel Diode",
                "Heat Generation Rate (W)": round(p_diode_chip, 6),
                "Count Basis": "Single Die",
            }
        )
    return pd.DataFrame(rows)


def build_star_ccm_heat_table(
    n_sim: int,
    p_main_chip: float,
    p_diode_chip: float,
    main_area_m2: float,
    main_volume_m3: float,
    diode_area_m2: float,
    diode_volume_m3: float,
) -> pd.DataFrame:
    """
    面向 STAR-CCM+ 的热源表。

    STAR-CCM+ 常见的挂载方式不是只看总功率，还会直接用：
    - Solid Region Volumetric Heat Source: W/m^3
    - Boundary Heat Flux: W/m^2
    所以这里一次性把三类量都准备好。
    """
    rows = []
    main_qvol = p_main_chip / max(main_volume_m3, 1e-18)
    diode_qvol = p_diode_chip / max(diode_volume_m3, 1e-18)
    main_qflux = p_main_chip / max(main_area_m2, 1e-18)
    diode_qflux = p_diode_chip / max(diode_area_m2, 1e-18)

    for idx in range(1, n_sim + 1):
        rows.append(
            {
                "Region": f"MainChip_{idx}",
                "Category": "Main Switch",
                "Total Power (W)": round(p_main_chip, 6),
                "Volumetric Heat Source (W/m^3)": round(main_qvol, 6),
                "Area Heat Flux (W/m^2)": round(main_qflux, 6),
                "Area (m^2)": main_area_m2,
                "Volume (m^3)": main_volume_m3,
            }
        )
        rows.append(
            {
                "Region": f"DiodeChip_{idx}",
                "Category": "Freewheel Diode",
                "Total Power (W)": round(p_diode_chip, 6),
                "Volumetric Heat Source (W/m^3)": round(diode_qvol, 6),
                "Area Heat Flux (W/m^2)": round(diode_qflux, 6),
                "Area (m^2)": diode_area_m2,
                "Volume (m^3)": diode_volume_m3,
            }
        )
    return pd.DataFrame(rows)


def build_excel_bytes(sheet_map: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheet_map.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output.getvalue()


def build_formula_audit_df(inputs: dict, result: dict) -> pd.DataFrame:
    """
    把当前计算真正启用的公式开关全部显式列出来。

    这样做的目的不是“好看”，而是避免后续自己或同事隔一段时间再打开工具时，
    忘记这次到底有没有启用 SiC 纯阻性锁、死区补偿、温漂锁、系统级缩放等关键逻辑。
    """
    sic_lock_active = "SiC" in inputs["device_type"]
    mode_label = "SVPWM 马鞍波解析式" if inputs["mode"] == "SVPWM" else "SPWM 正弦解析式"
    thermal_label = (
        "闭环双节点热迭代（主芯片 / 二极管独立热参数）"
        if "闭环" in inputs["sim_mode"]
        else "开环固定结温"
    )

    return pd.DataFrame(
        [
            {"审计项": "运行工况", "当前实现": inputs["op_mode"], "说明": "Motoring / Regeneration / Stall 三模态独立计算"},
            {"审计项": "调制解析式", "当前实现": mode_label, "说明": "保持原始工程公式框架，死区修正体现在 M_eff"},
            {"审计项": "主开关导通模型", "当前实现": "SiC 纯阻性锁 ON" if sic_lock_active else "IGBT 的 V0 + R 线性化", "说明": "SiC 主开关强制 V0 = 0，仅保留 Rds(on) 项"},
            {"审计项": "二极管导通模型", "当前实现": "Vf0 + R 线性化", "说明": "无论 IGBT / SiC 架构，续流路径都保留二极管压降项"},
            {"审计项": "死区补偿", "当前实现": f"启用，M_eff = {result['dead_meta']['m_eff']:.6f}" if result["dead_meta"]["dead_ratio"] > 0 else "关闭", "说明": "同时修正有效调制系数和导通路径重分配"},
            {"审计项": "Eon 温度策略", "当前实现": result["eon_meta"]["strategy_label"], "说明": "若原表含多个温度维度，则经验温漂系数自动锁死为 0"},
            {"审计项": "Eoff 温度策略", "当前实现": result["eoff_meta"]["strategy_label"], "说明": "防双重放大锁已内建"},
            {"审计项": "Erec 温度策略", "当前实现": result["erec_meta"]["strategy_label"], "说明": "防双重放大锁已内建"},
            {"审计项": "热学模式", "当前实现": thermal_label, "说明": "闭环下主芯片和二极管分别按各自单颗损耗迭代结温"},
            {"审计项": "缩放链路", "当前实现": f"N_src -> 单芯归一化 -> N_sim={inputs['n_sim']} -> N_arm_sys={inputs['n_arm_system']}", "说明": "先归一化、再扩容、最后系统级桥臂缩放"},
        ]
    )


def simulate_system(inputs: dict, tables: dict):
    cond_src_count = inputs["n_src_cond"] if "Module" in inputs["cond_data_type"] else 1
    sw_src_count = inputs["n_src_sw"] if "Module" in inputs["sw_data_type"] else 1

    norm_ev_m = normalize_vi_df(tables["ev_main"], cond_src_count)
    norm_ev_d = normalize_vi_df(tables["ev_diode"], cond_src_count)
    norm_ee_m = normalize_ei_df(tables["ee_main"], sw_src_count, ["Eon (mJ)", "Eoff (mJ)"])
    norm_ee_d = normalize_ei_df(tables["ee_diode"], sw_src_count, ["Erec (mJ)"])

    # 这里的 Ipk 仍然沿用你原始版本的系统级定义：
    # 相电流有效值 -> 单芯峰值电流。
    i_pk_chip = math.sqrt(2.0) * (inputs["iout_rms"] / inputs["n_sim"]) if inputs["n_sim"] > 0 else 0.0

    # 桥臂附加电阻是公共电阻，单颗芯片分担的等效串联电阻要乘以并联数。
    # 这是因为公共铜排/母排/焊层上承受的是总电流，而不是单芯电流。
    r_arm_chip = (inputs["r_arm_mohm"] / 1000.0) * inputs["n_sim"]

    # 封装内阻是单颗芯片自身寄生，不乘并联数。
    r_pkg_chip = inputs["r_pkg_mohm"] / 1000.0

    cosphi_mag = abs(inputs["cosphi"])
    active_cosphi = -cosphi_mag if "Regeneration" in inputs["op_mode"] else cosphi_mag
    active_cosphi = clamp(active_cosphi, -1.0, 1.0)
    theta = math.acos(active_cosphi) if abs(active_cosphi) <= 1.0 else 0.0

    current_sign = -1.0 if "Regeneration" in inputs["op_mode"] else 1.0
    dead_meta = calc_dead_time_compensation(
        inputs["mode"],
        inputs["fsw"],
        inputs["dead_time_us"],
        inputs["m_index"],
        current_sign,
        inputs["vdc_act"],
    )

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

    for loop_idx in range(loop_count):
        iter_interp_checks = [
            assess_interp_usage(norm_ev_m, "主开关导通表", i_pk_chip, tj_main_current),
            assess_interp_usage(norm_ev_d, "二极管导通表", i_pk_chip, tj_diode_current),
            assess_interp_usage(norm_ee_m, "主开关开关能量表", i_pk_chip, tj_main_current),
            assess_interp_usage(norm_ee_d, "二极管恢复能量表", i_pk_chip, tj_diode_current),
        ]
        for check in iter_interp_checks:
            if check["table_name"] not in extrapolation_log:
                extrapolation_log[check["table_name"]] = check.copy()
            else:
                extrapolation_log[check["table_name"]]["current_extrapolated"] = (
                    extrapolation_log[check["table_name"]]["current_extrapolated"] or check["current_extrapolated"]
                )
                extrapolation_log[check["table_name"]]["temp_extrapolated"] = (
                    extrapolation_log[check["table_name"]]["temp_extrapolated"] or check["temp_extrapolated"]
                )
                extrapolation_log[check["table_name"]]["any_extrapolated"] = (
                    extrapolation_log[check["table_name"]]["current_extrapolated"]
                    or extrapolation_log[check["table_name"]]["temp_extrapolated"]
                )
                extrapolation_log[check["table_name"]]["target_i"] = max(
                    extrapolation_log[check["table_name"]]["target_i"], check["target_i"]
                )
                extrapolation_log[check["table_name"]]["target_t"] = check["target_t"]

        main_model = build_linearized_device_model(
            norm_ev_m,
            i_pk_chip,
            tj_main_current,
            "V_drop (V)",
            force_zero_intercept="SiC" in inputs["device_type"],
        )
        diode_model = build_linearized_device_model(
            norm_ev_d,
            i_pk_chip,
            tj_diode_current,
            "Vf (V)",
            force_zero_intercept=False,
        )

        if "Stall" in inputs["op_mode"]:
            cond_result = calc_stall_losses(
                dead_meta["m_eff"],
                i_pk_chip,
                main_model,
                diode_model,
                r_pkg_chip,
                r_arm_chip,
                dead_meta,
            )
        else:
            cond_result = calc_pwm_conduction_losses(
                inputs["mode"],
                dead_meta["m_eff"],
                active_cosphi,
                theta,
                i_pk_chip,
                main_model,
                diode_model,
                r_pkg_chip,
                r_arm_chip,
                dead_meta,
            )

        eon_meta = calc_switching_energy(
            norm_ee_m,
            i_pk_chip,
            tj_main_current,
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
            i_pk_chip,
            tj_main_current,
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
            i_pk_chip,
            tj_diode_current,
            "Erec (mJ)",
            inputs["vdc_act"],
            inputs["v_ref"],
            inputs["kv_frd"],
            1.0,
            1.0,
            0.0,
            inputs["t_coeff_frd"],
            inputs["t_ref_dp"],
        )

        eon_adj = eon_meta["energy_mj"]
        eoff_adj = eoff_meta["energy_mj"]
        erec_adj = erec_meta["energy_mj"]

        if "Stall" in inputs["op_mode"]:
            p_sw_main_chip = inputs["fsw"] * ((eon_adj + eoff_adj) / 1000.0)
            p_sw_diode_chip = inputs["fsw"] * (erec_adj / 1000.0)
        else:
            # 保留历史版本中的 Ki_frd 接口，作为基波/峰值之间的工程修正占位。
            # 当 Ipk 正是由正弦 RMS 推导而来时，这个修正通常接近 1。
            i_wave_base = max(i_pk_chip / math.sqrt(2.0), 1e-12)
            i_wave_rms_chip = max(inputs["iout_rms"] / inputs["n_sim"], 1e-12)
            i_corr = math.pow(i_wave_rms_chip / i_wave_base, inputs["ki_frd"]) if inputs["ki_frd"] > 0 else 1.0

            p_sw_main_chip = (inputs["fsw"] / math.pi) * ((eon_adj + eoff_adj) / 1000.0)
            p_sw_diode_chip = (inputs["fsw"] / math.pi) * (erec_adj / 1000.0) * i_corr

        p_main_chip = cond_result["p_cond_main"] + p_sw_main_chip
        p_diode_chip = cond_result["p_cond_diode"] + p_sw_diode_chip
        p_total = (p_main_chip + p_diode_chip) * inputs["n_sim"]

        if "闭环" in inputs["sim_mode"]:
            # 这里不再像早期简化版那样把整臂总损耗直接乘单颗 Rth。
            # 更合理的做法是分别用“单颗主芯片损耗”和“单颗二极管损耗”回推各自结温。
            tj_main_new = inputs["t_case_main"] + p_main_chip * inputs["rth_jc_main"]
            tj_diode_new = inputs["t_case_diode"] + p_diode_chip * inputs["rth_jc_diode"]
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
                "Tj_main_new (℃)": round(tj_main_new, 6),
                "Tj_diode_new (℃)": round(tj_diode_new, 6),
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
    icepak_df = build_icepak_heat_table(inputs["n_sim"], p_main_chip, p_diode_chip)
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
            extrapolation_messages.append(f"{item['table_name']} 发生 {' + '.join(reasons)}，请优先补齐数据矩阵边界。")

    p_total_system = p_total * inputs["n_arm_system"]
    p_main_total_system = p_main_chip * inputs["n_sim"] * inputs["n_arm_system"]
    p_diode_total_system = p_diode_chip * inputs["n_sim"] * inputs["n_arm_system"]

    main_area_m2 = (inputs["main_die_length_mm"] * inputs["main_die_width_mm"]) * 1e-6
    main_volume_m3 = main_area_m2 * (inputs["main_die_thickness_mm"] * 1e-3)
    diode_area_m2 = (inputs["diode_die_length_mm"] * inputs["diode_die_width_mm"]) * 1e-6
    diode_volume_m3 = diode_area_m2 * (inputs["diode_die_thickness_mm"] * 1e-3)
    star_ccm_df = build_star_ccm_heat_table(
        inputs["n_sim"],
        p_main_chip,
        p_diode_chip,
        main_area_m2,
        main_volume_m3,
        diode_area_m2,
        diode_volume_m3,
    )

    loss_breakdown_df = pd.DataFrame(
        [
            {"对象": "主开关芯片", "项目": "导通损耗", "单颗 (W)": cond_result["p_cond_main"], "整臂总计 (W)": cond_result["p_cond_main"] * inputs["n_sim"]},
            {"对象": "主开关芯片", "项目": "开通+关断损耗", "单颗 (W)": p_sw_main_chip, "整臂总计 (W)": p_sw_main_chip * inputs["n_sim"]},
            {"对象": "续流二极管", "项目": "导通损耗", "单颗 (W)": cond_result["p_cond_diode"], "整臂总计 (W)": cond_result["p_cond_diode"] * inputs["n_sim"]},
            {"对象": "续流二极管", "项目": "恢复损耗", "单颗 (W)": p_sw_diode_chip, "整臂总计 (W)": p_sw_diode_chip * inputs["n_sim"]},
            {"对象": "系统级缩放", "项目": "主芯片系统总损耗", "单颗 (W)": np.nan, "整臂总计 (W)": p_main_total_system},
            {"对象": "系统级缩放", "项目": "二极管系统总损耗", "单颗 (W)": np.nan, "整臂总计 (W)": p_diode_total_system},
            {"对象": "系统级缩放", "项目": "系统总功耗", "单颗 (W)": np.nan, "整臂总计 (W)": p_total_system},
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

    summary_df = pd.DataFrame(
        [
            {"指标": "运行场景", "数值": inputs["op_mode"], "单位": ""},
            {"指标": "芯片技术类型", "数值": inputs["device_type"], "单位": ""},
            {"指标": "调制模式", "数值": inputs["mode"], "单位": ""},
            {"指标": "目标并联芯片数 N_sim", "数值": inputs["n_sim"], "单位": ""},
            {"指标": "单颗峰值电流 I_pk", "数值": i_pk_chip, "单位": "A"},
            {"指标": "主芯片结温", "数值": tj_main_current, "单位": "℃"},
            {"指标": "二极管结温", "数值": tj_diode_current, "单位": "℃"},
            {"指标": "控制结温（取最大）", "数值": dominant_tj, "单位": "℃"},
            {"指标": "主芯片单颗发热率", "数值": p_main_chip, "单位": "W"},
            {"指标": "二极管单颗发热率", "数值": p_diode_chip, "单位": "W"},
            {"指标": "STAR-CCM+ 主芯片体积热源", "数值": p_main_chip / max(main_volume_m3, 1e-18), "单位": "W/m^3"},
            {"指标": "STAR-CCM+ 二极管体积热源", "数值": p_diode_chip / max(diode_volume_m3, 1e-18), "单位": "W/m^3"},
            {"指标": "整臂总功耗", "数值": p_total, "单位": "W"},
            {"指标": "系统桥臂数 N_arm_sys", "数值": inputs["n_arm_system"], "单位": ""},
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

    input_snapshot_df = pd.DataFrame(
        [
            {"参数": "模块芯片技术类型", "取值": inputs["device_type"], "单位": ""},
            {"参数": "导通数据来源", "取值": inputs["cond_data_type"], "单位": ""},
            {"参数": "导通原测芯片数", "取值": cond_src_count, "单位": ""},
            {"参数": "开关数据来源", "取值": inputs["sw_data_type"], "单位": ""},
            {"参数": "开关原测芯片数", "取值": sw_src_count, "单位": ""},
            {"参数": "目标仿真芯片数", "取值": inputs["n_sim"], "单位": ""},
            {"参数": "系统桥臂数", "取值": inputs["n_arm_system"], "单位": ""},
            {"参数": "热学模式", "取值": inputs["sim_mode"], "单位": ""},
            {"参数": "母线电压 V_dc", "取值": inputs["vdc_act"], "单位": "V"},
            {"参数": "输出电流 I_out", "取值": inputs["iout_rms"], "单位": "A"},
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
            {"参数": "K_v_on", "取值": inputs["kv_on"], "单位": ""},
            {"参数": "K_v_off", "取值": inputs["kv_off"], "单位": ""},
            {"参数": "K_v_frd", "取值": inputs["kv_frd"], "单位": ""},
            {"参数": "K_i_frd", "取值": inputs["ki_frd"], "单位": ""},
            {"参数": "封装内阻 R_pkg,chip", "取值": inputs["r_pkg_mohm"], "单位": "mΩ"},
            {"参数": "桥臂附加电阻 R_arm", "取值": inputs["r_arm_mohm"], "单位": "mΩ"},
            {"参数": "主芯片长度 L_main", "取值": inputs["main_die_length_mm"], "单位": "mm"},
            {"参数": "主芯片宽度 W_main", "取值": inputs["main_die_width_mm"], "单位": "mm"},
            {"参数": "主芯片厚度 T_main", "取值": inputs["main_die_thickness_mm"], "单位": "mm"},
            {"参数": "二极管长度 L_diode", "取值": inputs["diode_die_length_mm"], "单位": "mm"},
            {"参数": "二极管宽度 W_diode", "取值": inputs["diode_die_width_mm"], "单位": "mm"},
            {"参数": "二极管厚度 T_diode", "取值": inputs["diode_die_thickness_mm"], "单位": "mm"},
            {"参数": "K_ron", "取值": inputs["kron"], "单位": ""},
            {"参数": "K_roff", "取值": inputs["kroff"], "单位": ""},
            {"参数": "主管温漂系数", "取值": inputs["t_coeff_igbt"], "单位": "1/K"},
            {"参数": "续流温漂系数", "取值": inputs["t_coeff_frd"], "单位": "1/K"},
            {"参数": "页首仿真备注", "取值": inputs["user_notes"], "单位": ""},
            {"参数": "侧栏工程师备忘录", "取值": inputs["engineer_memo"], "单位": ""},
        ]
    )
    temp_strategy_df = pd.DataFrame(
        [
            {
                "对象": "Eon",
                "基础查表值 (mJ)": eon_meta["e_base_mj"],
                "生效温漂系数": eon_meta["effective_temp_coeff"],
                "温漂修正因子": eon_meta["temp_correction"],
                "电阻修正因子": eon_meta["rg_correction"],
                "电压修正因子": eon_meta["voltage_correction"],
                "最终能量 (mJ)": eon_meta["energy_mj"],
                "温度策略": eon_meta["strategy_label"],
            },
            {
                "对象": "Eoff",
                "基础查表值 (mJ)": eoff_meta["e_base_mj"],
                "生效温漂系数": eoff_meta["effective_temp_coeff"],
                "温漂修正因子": eoff_meta["temp_correction"],
                "电阻修正因子": eoff_meta["rg_correction"],
                "电压修正因子": eoff_meta["voltage_correction"],
                "最终能量 (mJ)": eoff_meta["energy_mj"],
                "温度策略": eoff_meta["strategy_label"],
            },
            {
                "对象": "Erec",
                "基础查表值 (mJ)": erec_meta["e_base_mj"],
                "生效温漂系数": erec_meta["effective_temp_coeff"],
                "温漂修正因子": erec_meta["temp_correction"],
                "电阻修正因子": erec_meta["rg_correction"],
                "电压修正因子": erec_meta["voltage_correction"],
                "最终能量 (mJ)": erec_meta["energy_mj"],
                "温度策略": erec_meta["strategy_label"],
            },
        ]
    )
    formula_audit_df = build_formula_audit_df(inputs, {"dead_meta": dead_meta, "eon_meta": eon_meta, "eoff_meta": eoff_meta, "erec_meta": erec_meta})

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
                        {"参数": "主/二极管热参数分离", "取值": inputs["split_thermal_params"], "单位": ""},
                    ]
                ),
            ],
            ignore_index=True,
        )
    else:
        input_snapshot_df = pd.concat(
            [
                input_snapshot_df,
                pd.DataFrame([{"参数": "固定结温 Tj", "取值": inputs["fixed_tj"], "单位": "℃"}]),
            ],
            ignore_index=True,
        )

    excel_sheets = {
        "summary": summary_df,
        "loss_breakdown": loss_breakdown_df,
        "icepak_heat": icepak_df,
        "linearized_model": linearized_df,
        "iterations": pd.DataFrame(iteration_rows),
        "matrix_health": matrix_health_df,
        "extrapolation_monitor": extrapolation_df,
        "star_ccm_heat": star_ccm_df,
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

    return {
        "op_mode": inputs["op_mode"],
        "n_sim": inputs["n_sim"],
        "n_arm_system": inputs["n_arm_system"],
        "device_type": inputs["device_type"],
        "user_notes": inputs["user_notes"],
        "engineer_memo": inputs["engineer_memo"],
        "cond_src_count": cond_src_count,
        "sw_src_count": sw_src_count,
        "norm_ev_m": norm_ev_m,
        "norm_ev_d": norm_ev_d,
        "norm_ee_m": norm_ee_m,
        "norm_ee_d": norm_ee_d,
        "i_pk_chip": i_pk_chip,
        "r_pkg_chip": r_pkg_chip,
        "r_arm_chip": r_arm_chip,
        "active_cosphi": active_cosphi,
        "theta": theta,
        "dead_meta": dead_meta,
        "main_model": main_model,
        "diode_model": diode_model,
        "cond_result": cond_result,
        "eon_adj": eon_adj,
        "eoff_adj": eoff_adj,
        "erec_adj": erec_adj,
        "eon_meta": eon_meta,
        "eoff_meta": eoff_meta,
        "erec_meta": erec_meta,
        "p_sw_main_chip": p_sw_main_chip,
        "p_sw_diode_chip": p_sw_diode_chip,
        "p_main_chip": p_main_chip,
        "p_diode_chip": p_diode_chip,
        "p_total": p_total,
        "p_total_system": p_total_system,
        "p_main_total_system": p_main_total_system,
        "p_diode_total_system": p_diode_total_system,
        "tj_main_current": tj_main_current,
        "tj_diode_current": tj_diode_current,
        "dominant_tj": dominant_tj,
        "summary_df": summary_df,
        "loss_breakdown_df": loss_breakdown_df,
        "icepak_df": icepak_df,
        "star_ccm_df": star_ccm_df,
        "matrix_health_df": matrix_health_df,
        "extrapolation_df": extrapolation_df,
        "extrapolation_messages": extrapolation_messages,
        "temp_strategy_df": temp_strategy_df,
        "formula_audit_df": formula_audit_df,
        "linearized_df": linearized_df,
        "iteration_df": pd.DataFrame(iteration_rows),
        "input_snapshot_df": input_snapshot_df,
        "excel_bytes": build_excel_bytes(excel_sheets),
        "icepak_csv_bytes": icepak_df.to_csv(index=False).encode("utf-8-sig"),
        "star_ccm_csv_bytes": star_ccm_df.to_csv(index=False).encode("utf-8-sig"),
        "breakdown_csv_bytes": loss_breakdown_df.to_csv(index=False).encode("utf-8-sig"),
    }


st.title("🛡️ 功率模块全工况电热联合仿真平台 (知识传承版)")

with st.expander("📝 查看工程随手记 & 快速操作指南 (点此做笔记)", expanded=True):
    guide_col, note_col = st.columns([1.15, 1.0])
    with guide_col:
        st.markdown(
            """
            **🚀 快速操作流程：**
            1. **左侧边栏**：设定芯片类型、原始数据来源、目标并联芯片数、热学模式。
            2. **第一步表格**：从规格书/测试报告直接粘贴主芯片与二极管的 V-I、E-I 矩阵。
            3. **第二步工况**：录入母线电压、相电流、调制系数、开关频率、死区时间。  
               **反拖工况请切到 Regeneration，不要靠手工输入负 cos_phi 代替。**
            4. **热学参数**：闭环模式下建议优先使用“主芯片 / 二极管分开热参数”，更贴近真实模块。
            5. **点击计算**：程序会先把原始数据归一化成单芯模型，再执行三模态电热联合求解。
            6. **结果区**：重点查看主芯片/二极管单颗发热率、整臂总损耗、系统级缩放总功耗、死区修正量、Icepak 热源表。

            **📌 工程红线提醒：**
            - `R_pkg,chip` 与 `R_arm` 默认都按 **0 mΩ** 处理，只在原始 V-I 数据没有包含这些寄生压降时再额外填写。
            - 闭环模式下，程序按 **主芯片与二极管各自单颗损耗** 迭代结温，而不是拿整臂总损耗去乘单颗热阻。
            - SiC 主芯片默认按纯阻性通道处理，强制 `V0 = 0`，这是刻意保留的工程规则，不要删。
            - 只要工作点超出原始数据矩阵边界，结果页就会显式提示外推风险，建议补齐矩阵而不是长期依赖外推。
            """
        )
    with note_col:
        user_notes = st.text_area(
            "🗒️ 仿真备注 (项目名 / 规格书版本 / 对标结论)",
            placeholder="例如：A平台 800V SiC 6并联，规格书版本 V3.2，和 XX 模块对标……",
            height=170,
        )
        st.caption("建议把计算结果截图和这段备注一起存档，避免隔一段时间忘掉参数上下文。")

with st.sidebar:
    st.header("⚙️ 核心技术架构")
    st.info("工程显式假设：封装内阻 `R_pkg,chip` 与桥臂附加电阻 `R_arm` 默认均为 0。若原始 V-I 曲线已含这些压降，请不要重复计入。")

    device_type = st.radio(
        "1. 模块芯片技术类型",
        ["IGBT + FRD (传统硅基)", "SiC MOSFET (碳化硅)"],
        help="IGBT 模式保留 Vce0 + R 的线性化；SiC 模式强制主开关 V0 = 0，仅保留纯阻性通道。",
    )

    st.divider()
    st.header("🧮 原始数据规格 (必填)")
    st.warning("这里决定程序如何理解你粘贴进来的规格书数据。单芯数据不会再除以并联数；模块数据会自动归一化为单芯模型。")
    cond_data_type = st.radio(
        "A. 导通 V-I 表格代表：",
        ["单芯片数据 (Bare Die)", "模块半桥数据 (Module)"],
        help="导通曲线如果来自模块端子数据，请务必选择 Module。",
    )
    n_src_cond = st.number_input(
        "V-I 原测模块芯片数",
        value=6,
        min_value=1,
        help="只有当 V-I 原始数据来自模块时，这个并联芯片数才会参与归一化。",
        disabled="单芯片数据" in cond_data_type,
    )

    sw_data_type = st.radio(
        "B. 开关 E-I 表格代表：",
        ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"],
        help="双脉冲测试多数是模块级能量和，程序会自动折算为单芯能量。",
    )
    n_src_sw = st.number_input(
        "E-I 原测模块芯片数",
        value=6,
        min_value=1,
        help="只有当 E-I 原始数据来自模块时，这个并联芯片数才会参与归一化。",
        disabled="单芯片数据" in sw_data_type,
    )

    st.divider()
    st.header("🎯 仿真目标规模重构")
    n_sim = st.number_input(
        "目标仿真芯片数 (N_sim)",
        value=6,
        min_value=1,
        help="程序会先建立单芯模型，再按这个并联数重构目标模块规模，用于扩容/减容评估。",
    )
    n_arm_system = st.number_input(
        "系统桥臂数 (N_arm_sys)",
        value=1,
        min_value=1,
        help="单臂结果乘以此值可得到系统级总损耗。三相逆变器通常填 3；若只关心单个半桥，填 1。",
    )

    st.divider()
    st.header("🔄 热学计算工作流")
    sim_mode = st.radio(
        "模式选择",
        ["A. 开环盲算 (已知结温)", "B. 闭环迭代 (已知热阻)"],
        help="开环：直接指定工作结温；闭环：用单颗损耗和 RthJC 迭代主芯片/二极管结温。",
    )
    if "闭环" in sim_mode:
        split_thermal_params = st.checkbox(
            "主芯片 / 二极管分开热参数",
            value=True,
            help="实际模块里主芯片和二极管的 RthJC 往往不同。默认开启，避免把二者强行等同。",
        )
        rth_jc_main = st.number_input("主芯片热阻 RthJC_main (K/W)", value=0.065, min_value=0.0, format="%.4f")
        t_case_main = st.number_input("主芯片参考壳温 Tc_main (℃)", value=65.0)
        if split_thermal_params:
            rth_jc_diode = st.number_input("二极管热阻 RthJC_diode (K/W)", value=0.085, min_value=0.0, format="%.4f")
            t_case_diode = st.number_input("二极管参考壳温 Tc_diode (℃)", value=65.0)
        else:
            rth_jc_diode = rth_jc_main
            t_case_diode = t_case_main
        fixed_tj = None
    else:
        fixed_tj = st.number_input("设定全局目标结温 Tj (℃)", value=150.0)
        split_thermal_params = False
        rth_jc_main = None
        rth_jc_diode = None
        t_case_main = None
        t_case_diode = None

    st.divider()
    st.header("🧱 STAR-CCM+ 热源几何")
    split_die_geometry = st.checkbox(
        "主芯片 / 二极管分开几何",
        value=True,
        help="STAR-CCM+ 常需要体积热源密度 W/m^3，因此这里按单颗 die 几何计算体积与受热面积。",
    )
    main_die_length_mm = st.number_input("主芯片长度 L_main (mm)", value=10.0, min_value=0.001, format="%.3f")
    main_die_width_mm = st.number_input("主芯片宽度 W_main (mm)", value=10.0, min_value=0.001, format="%.3f")
    main_die_thickness_mm = st.number_input("主芯片厚度 T_main (mm)", value=0.20, min_value=0.001, format="%.3f")
    if split_die_geometry:
        diode_die_length_mm = st.number_input("二极管长度 L_diode (mm)", value=8.0, min_value=0.001, format="%.3f")
        diode_die_width_mm = st.number_input("二极管宽度 W_diode (mm)", value=8.0, min_value=0.001, format="%.3f")
        diode_die_thickness_mm = st.number_input("二极管厚度 T_diode (mm)", value=0.20, min_value=0.001, format="%.3f")
    else:
        diode_die_length_mm = main_die_length_mm
        diode_die_width_mm = main_die_width_mm
        diode_die_thickness_mm = main_die_thickness_mm

    st.divider()
    engineer_memo = st.text_area(
        "🧠 工程师专属备忘录",
        placeholder="记录你自己容易忘的规范：例如 E-I 表默认来自双脉冲整臂、反拖一定切 Regeneration、R_arm 默认 0 ……",
        height=180,
    )

st.divider()
st.header("📊 第一步：特性数据录入 (归一化中心)")
st.info("无论原始数据是单芯还是模块，程序都会先把它“拉平”为标准单芯模型，然后再进行系统级扩容、死区补偿和热迭代。")

col_t, col_d = st.columns(2)
with col_t:
    st.subheader("🔴 主开关管 (IGBT / SiC)")
    st.caption("1. 导通特性 (Vce / Vds)")
    ev_main = st.data_editor(DEFAULT_MAIN_VI, num_rows="dynamic", key="v_main")
    st.caption("2. 开关能量矩阵 (Eon / Eoff)")
    ee_main = st.data_editor(DEFAULT_MAIN_EI, num_rows="dynamic", key="e_main")

with col_d:
    st.subheader("🔵 续流二极管 (FRD / Body Diode)")
    st.caption("1. 正向压降 (Vf / Vsd)")
    ev_diode = st.data_editor(DEFAULT_DIODE_VI, num_rows="dynamic", key="v_diode")
    st.caption("2. 反向恢复能量 (Erec)")
    ee_diode = st.data_editor(DEFAULT_DIODE_EI, num_rows="dynamic", key="ee_diode")

st.divider()
st.header("⚙️ 第二步：全场景工况与物理修正系数")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**⚡ 车辆 / 电驱动工况**")
    op_mode = st.selectbox(
        "🏎️ 运行场景切换",
        ["电动/巡航 (Motoring)", "制动/反拖 (Regeneration)", "最恶劣堵转 (Stall)"],
        help="Motoring：主开关更热；Regeneration：二极管更热；Stall：按最重载占空比做保守评估。",
    )
    vdc_act = st.number_input("母线 V_dc (V)", value=713.0, min_value=0.0)
    iout_rms = st.number_input("有效值 I_out (A)", value=264.5, min_value=0.0)
    fsw = st.number_input("开关频率 f_sw (Hz)", value=10000.0, min_value=0.0)
    fout = st.number_input("输出频率 f_out (Hz)", value=200.0, min_value=0.0)
    m_index = st.number_input("调制系数 M", value=0.90, min_value=0.0, max_value=1.15)
    cosphi = st.number_input("功率因数幅值 cos_phi", value=0.90, min_value=0.0, max_value=1.0)
    mode = st.selectbox("调制模式选择", ["SVPWM", "SPWM"])

    if fout < 5.0 and "Stall" not in op_mode:
        st.warning(f"当前输出频率 {fout:.2f} Hz 很低，若接近堵转请切换到 Stall 模式以获得更保守的热结果。")

with c2:
    st.markdown("**📏 测试基准 / 驱动 / 死区**")
    v_ref = st.number_input("规格书基准 V_nom (V)", value=600.0, min_value=0.001)
    t_ref_dp = st.number_input("规格书基准 T_ref (℃)", value=150.0)
    rg_on_ref = st.number_input("手册 R_g,on (Ω)", value=2.5, min_value=0.0)
    rg_off_ref = st.number_input("手册 R_g,off (Ω)", value=20.0, min_value=0.0)
    rg_on_act = st.number_input("实际 R_on (Ω)", value=2.5, min_value=0.0)
    rg_off_act = st.number_input("实际 R_off (Ω)", value=20.0, min_value=0.0)
    dead_time_us = st.number_input(
        "死区时间 t_dead (us)",
        value=2.0,
        min_value=0.0,
        format="%.3f",
        help="程序会显式把死区占空损失映射到有效调制系数与导通路径重分配中。",
    )

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
    t_coeff_frd = st.number_input(
        "续流温漂系数 (1/K)",
        value=0.006 if "IGBT" in device_type else 0.003,
        format="%.4f",
    )
    r_pkg_mohm = st.number_input(
        "封装内阻 R_pkg,chip (mΩ)",
        value=0.0,
        min_value=0.0,
        help="单颗芯片内部引线/焊层/封装寄生。默认 0，只有当原始 V-I 不含这部分压降时才填写。",
    )
    r_arm_mohm = st.number_input(
        "桥臂附加电阻 R_arm (mΩ)",
        value=0.0,
        min_value=0.0,
        help="外部母排/汇流排/连接片等公共附加电阻。默认 0，若原始 V-I 已含此项请不要重复录入。",
    )

st.info(
    "工程显式提醒：`R_pkg,chip` 代表单颗封装寄生，`R_arm` 代表公共电路寄生。两者默认都为 0，是为了避免和原始 V-I 曲线重复计损。"
)
st.warning(
    "死区补偿不是装饰项。只要 `t_dead > 0` 且 `f_sw > 0`，程序就会同时修正有效调制系数和主开关/二极管导通时间分配。"
)

inputs = {
    "device_type": device_type,
    "cond_data_type": cond_data_type,
    "n_src_cond": int(n_src_cond),
    "sw_data_type": sw_data_type,
    "n_src_sw": int(n_src_sw),
    "n_sim": int(n_sim),
    "n_arm_system": int(n_arm_system),
    "sim_mode": sim_mode,
    "split_thermal_params": bool(split_thermal_params),
    "split_die_geometry": bool(split_die_geometry),
    "rth_jc_main": 0.0 if rth_jc_main is None else float(rth_jc_main),
    "rth_jc_diode": 0.0 if rth_jc_diode is None else float(rth_jc_diode),
    "t_case_main": 0.0 if t_case_main is None else float(t_case_main),
    "t_case_diode": 0.0 if t_case_diode is None else float(t_case_diode),
    "main_die_length_mm": float(main_die_length_mm),
    "main_die_width_mm": float(main_die_width_mm),
    "main_die_thickness_mm": float(main_die_thickness_mm),
    "diode_die_length_mm": float(diode_die_length_mm),
    "diode_die_width_mm": float(diode_die_width_mm),
    "diode_die_thickness_mm": float(diode_die_thickness_mm),
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

validated_main_vi, errs, warns = validate_numeric_table(ev_main, "主开关管导通表", ["Temp (℃)", "Current (A)", "V_drop (V)"])
tables_for_validation["ev_main"] = validated_main_vi
validation_errors.extend(errs)
validation_warnings.extend(warns)

validated_main_ei, errs, warns = validate_numeric_table(
    ee_main, "主开关管开关能量表", ["Temp (℃)", "Current (A)", "Eon (mJ)", "Eoff (mJ)"]
)
tables_for_validation["ee_main"] = validated_main_ei
validation_errors.extend(errs)
validation_warnings.extend(warns)

validated_diode_vi, errs, warns = validate_numeric_table(ev_diode, "二极管导通表", ["Temp (℃)", "Current (A)", "Vf (V)"])
tables_for_validation["ev_diode"] = validated_diode_vi
validation_errors.extend(errs)
validation_warnings.extend(warns)

validated_diode_ei, errs, warns = validate_numeric_table(
    ee_diode, "二极管恢复能量表", ["Temp (℃)", "Current (A)", "Erec (mJ)"]
)
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

if validation_warnings:
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
    for extrapolation_message in result["extrapolation_messages"]:
        st.warning(extrapolation_message)
    if "SiC" in result["device_type"]:
        st.info("SiC 架构锁已启用：主开关导通模型强制 `V0 = 0`，底层仅保留纯阻性 `Rds(on)` 发热项；二极管侧仍保留 `Vf0 + R` 模型。")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("控制结温 Tj,max", f"{result['dominant_tj']:.1f} ℃")
    m2.metric("主芯片结温", f"{result['tj_main_current']:.1f} ℃")
    m3.metric("二极管结温", f"{result['tj_diode_current']:.1f} ℃")
    m4.metric("整臂发热总功耗", f"{result['p_total']:.1f} W")
    m5.metric("死区修正后 M_eff", f"{result['dead_meta']['m_eff']:.4f}")

    st.divider()
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("🔴 主芯片单颗 Heat Generation Rate", f"{result['p_main_chip']:.2f} W")
    p2.metric("🔵 二极管单颗 Heat Generation Rate", f"{result['p_diode_chip']:.2f} W")
    p3.metric("🔴 主芯片整臂总损耗", f"{result['p_main_chip'] * result['n_sim']:.1f} W")
    p4.metric("🔵 二极管整臂总损耗", f"{result['p_diode_chip'] * result['n_sim']:.1f} W")

    extra1, extra2, extra3, extra4 = st.columns(4)
    extra1.metric("单颗峰值电流 I_pk", f"{result['i_pk_chip']:.2f} A")
    extra2.metric("死区占空损失 D_dead", f"{result['dead_meta']['dead_ratio']:.4f}")
    extra3.metric("死区等效相电压误差", f"{result['dead_meta']['phase_voltage_error_v']:.2f} V")
    extra4.metric("系统级总功耗", f"{result['p_total_system']:.1f} W")

    st.info(
        f"开关能量温度策略：Eon = {result['eon_meta']['strategy_label']}；"
        f"Eoff = {result['eoff_meta']['strategy_label']}；"
        f"Erec = {result['erec_meta']['strategy_label']}。"
    )
    st.caption(
        f"系统级总功耗按 N_arm_sys = {result['n_arm_system']} "
        "对单臂结果等比例缩放，适用于对称多桥臂系统的快速总量评估。"
    )

    dl1, dl2, dl3, dl4 = st.columns(4)
    with dl1:
        st.download_button(
            "⬇️ 下载 Icepak 热源 CSV",
            data=result["icepak_csv_bytes"],
            file_name="icepak_heat_generation_rate.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "⬇️ 下载 STAR-CCM+ 热源 CSV",
            data=result["star_ccm_csv_bytes"],
            file_name="star_ccm_heat_source_table.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl3:
        st.download_button(
            "⬇️ 下载损耗拆分 CSV",
            data=result["breakdown_csv_bytes"],
            file_name="loss_breakdown.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl4:
        st.download_button(
            "⬇️ 下载完整 Excel",
            data=result["excel_bytes"],
            file_name="system_level_thermal_platform_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        ["结果总览", "Icepak 热源", "STAR-CCM+ 热源", "线性化模型", "热迭代历史", "归一化数据", "矩阵健康度", "外推监视", "公式审计"]
    )

    with tab1:
        st.markdown("**损耗拆分总表**")
        st.dataframe(result["loss_breakdown_df"], use_container_width=True)
        st.markdown("**核心汇总指标**")
        st.dataframe(result["summary_df"], use_container_width=True)
        st.markdown("**输入快照**")
        st.dataframe(result["input_snapshot_df"], use_container_width=True)

    with tab2:
        st.markdown("**用于 Ansys Icepak / Flotherm 的单芯片发热率表**")
        st.dataframe(result["icepak_df"], use_container_width=True, height=420)
        st.caption("这张表按照“单颗 die 一个热源”的方式展开，便于直接映射到热仿真体区域。")

    with tab3:
        st.markdown("**用于 STAR-CCM+ 的热源表**")
        st.dataframe(result["star_ccm_df"], use_container_width=True, height=420)
        st.caption("这里同时给出单颗总功率、体积热源 W/m^3 和面热流 W/m^2，便于在 STAR-CCM+ 的 solid region / boundary 上直接挂载。")

    with tab4:
        st.markdown("**当前工作点线性化结果：V0 + R × I**")
        st.dataframe(result["linearized_df"], use_container_width=True)
        st.markdown("**说明**")
        st.write(
            "主芯片与二极管都基于当前迭代结温，在 `Ipk` 与 `Ipk/2` 两点上线性化。"
            "SiC 主开关额外强制 `V0 = 0`，保留纯阻性导通模型。"
        )

    with tab5:
        st.markdown("**闭环热迭代记录**")
        st.dataframe(result["iteration_df"], use_container_width=True)
        st.caption("闭环模式下，主芯片与二极管分别按各自单颗损耗迭代结温，至少完成 15 次迭代后才允许按 0.05 ℃ 收敛阈值提前退出。")

    with tab6:
        st.markdown("**归一化后的单芯 V-I / E-I 数据**")
        subtab1, subtab2, subtab3, subtab4 = st.tabs(["主芯片 V-I", "主芯片 E-I", "二极管 V-I", "二极管 E-I"])
        with subtab1:
            st.dataframe(result["norm_ev_m"], use_container_width=True)
        with subtab2:
            st.dataframe(result["norm_ee_m"], use_container_width=True)
        with subtab3:
            st.dataframe(result["norm_ev_d"], use_container_width=True)
        with subtab4:
            st.dataframe(result["norm_ee_d"], use_container_width=True)

    with tab7:
        st.markdown("**矩阵维度、归一化来源与温漂锁状态**")
        st.dataframe(result["matrix_health_df"], use_container_width=True)
        st.caption("这个诊断表就是为了防止把二维温度矩阵和线性温漂系数同时叠加，造成 Eon/Eoff/Erec 双重放大。")

    with tab8:
        st.markdown("**插值/外推边界监视**")
        st.dataframe(result["extrapolation_df"], use_container_width=True)
        st.caption("只要某张表发生电流或温度越界，这里就会标红对应逻辑风险。工程上建议优先补齐矩阵边界，而不是长期依赖外推。")

    with tab9:
        st.markdown("**底层公式 / 开关策略审计**")
        st.dataframe(result["formula_audit_df"], use_container_width=True)
        st.markdown("**开关能量修正明细**")
        st.dataframe(result["temp_strategy_df"], use_container_width=True)
        st.caption("这页的作用就是让你核对：当前计算到底是不是按你原来的底层物理逻辑在跑，尤其是 SiC、温漂锁和死区补偿。")

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
    st.write("先把原始数据归一化成单芯模型，再按 `N_sim` 重构目标模块规模。这是整个系统级扩容评估的根。")

    st.markdown("### 📗 2. 导通模型线性化")
    st.write("程序按 `Ipk` 与 `Ipk/2` 两点把器件导通曲线线性化为 `V_0 + R·I`：")
    st.latex(r"R_{eq} = \frac{V(I_{pk}) - V(I_{pk}/2)}{I_{pk} - I_{pk}/2}")
    st.latex(r"V_0 = V(I_{pk}) - R_{eq}\cdot I_{pk}")
    st.info("SiC 主开关强制取 $V_0 = 0$，仅保留纯阻性导通项，这是工程上常用的简化规则。")
    st.latex(r"V_{0,SiC}=0 \quad \Rightarrow \quad V_{SiC}(I)\approx R_{ds(on),eq}\cdot I")

    st.markdown("### 📙 3. PWM 导通损耗")
    st.markdown("#### 3.1 SPWM")
    st.latex(
        r"P_{cond,IGBT} = \left(\frac{1}{2\pi} + \frac{M_{eff}\cos\phi}{8}\right)V_{CE0}I_{pk} + \left(\frac{1}{8} + \frac{M_{eff}\cos\phi}{3\pi}\right)R_{tot,IGBT}I_{pk}^2"
    )
    st.latex(
        r"P_{cond,D} = \left(\frac{1}{2\pi} - \frac{M_{eff}\cos\phi}{8}\right)V_{F0}I_{pk} + \left(\frac{1}{8} - \frac{M_{eff}\cos\phi}{3\pi}\right)R_{tot,D}I_{pk}^2"
    )
    st.latex(
        r"P_{cond,SiC}^{SPWM} = \left(\frac{1}{8} + \frac{M_{eff}\cos\phi}{3\pi}\right)R_{tot,SiC}I_{pk}^2"
    )
    st.markdown("#### 3.2 SVPWM")
    st.latex(
        r"P_{cond,IGBT} \approx \frac{M_{eff}\cos\phi}{4}V_{CE0}I_{pk} + \left(\frac{24\cos\phi - 2\sqrt{3}\cos(2\varphi) - 3\sqrt{3}}{24\pi}\right)R_{tot,IGBT}I_{pk}^2"
    )
    st.latex(
        r"P_{cond,D} \approx \left(\frac{4 - M_{eff}\pi\cos\phi}{4\pi}\right)V_{F0}I_{pk} + \left(\frac{6\pi - 24M_{eff}\cos\phi + 2\sqrt{3}M_{eff}\cos(2\varphi) + 3\sqrt{3}M_{eff}}{24\pi}\right)R_{tot,D}I_{pk}^2"
    )
    st.latex(
        r"P_{cond,SiC}^{SVPWM} \approx \left(\frac{24\cos\phi - 2\sqrt{3}\cos(2\varphi) - 3\sqrt{3}}{24\pi}\right)R_{tot,SiC}I_{pk}^2"
    )
    st.write("其中：")
    st.latex(r"R_{tot} = R_{dynamic} + R_{pkg,chip} + R_{arm,eq}")
    st.latex(r"R_{arm,eq} = R_{arm} \cdot N_{sim}")
    st.caption("说明：上述 SiC 导通公式不是另起炉灶，而是直接由你原始 IGBT / SVPWM / SPWM 解析式在 `V0=0` 约束下退化得到。")

    st.markdown("### 📕 4. 死区时间补偿（显式保留）")
    st.latex(r"D_{dead} = 2 \cdot t_{dead} \cdot f_{sw}")
    st.latex(r"M_{eff} = M - sgn(i)\cdot K_{mode}\cdot D_{dead}")
    st.latex(r"K_{mode} = \frac{4}{\pi}\;(\mathrm{SVPWM}),\quad \frac{2}{\pi}\;(\mathrm{SPWM})")
    st.write(
        "程序不仅修正有效调制系数，还会把死区造成的导通时间从一个器件显式转移到另一个器件："
        "Motoring 下默认从主开关转移到二极管，Regeneration 下反向处理。"
    )

    st.markdown("### 📔 5. 开关损耗")
    st.latex(
        r"E_{adj} = E_{nom}(I_{pk}, T_j)\cdot \left(\frac{R_{g,act}}{R_{g,ref}}\right)^{K_r}\cdot \left(\frac{V_{dc}}{V_{nom}}\right)^{K_v}"
    )
    st.write("若原始能量表本身缺少温度维度，则再叠加线性温漂补偿：")
    st.latex(r"E_{adj} = E_{adj}\cdot \left[1 + T_{coeff}(T_j - T_{ref})\right]")
    st.write("若原始能量表已经具备两个及以上温度维度，则程序会启动“防双重放大锁”，直接令经验温漂系数失效：")
    st.latex(r"\mathrm{If}\;|\mathcal{T}_{table}| \ge 2,\quad T_{coeff,eff}=0")
    st.latex(r"P_{sw,main} = \frac{f_{sw}}{\pi}(E_{on,adj}+E_{off,adj})")
    st.latex(r"P_{sw,frd} = \frac{f_{sw}}{\pi}E_{rec,adj}\cdot I_{corr}")

    st.markdown("### 📓 6. 堵转极限工况")
    st.latex(r"D_{max} = \frac{1 + M_{eff}}{2}")
    st.latex(r"P_{cond,stall} = D_{max}(V_0I_{pk}+RI_{pk}^2)")
    st.latex(r"P_{cond,stall}^{SiC} = D_{max}R_{tot,SiC}I_{pk}^2")
    st.latex(r"P_{sw,stall} = f_{sw}\cdot E_{adj,total}(I_{pk})")
    st.write("堵转模式故意舍弃正弦平均因子，按最大占空比直流重载处理，用于最保守的热极限评估。")

    st.markdown("### 📒 7. 闭环热迭代")
    st.latex(r"T_{j,main}^{k+1} = T_{c,main} + P_{main,chip}^{k}\cdot R_{thJC,main}")
    st.latex(r"T_{j,diode}^{k+1} = T_{c,diode} + P_{diode,chip}^{k}\cdot R_{thJC,diode}")
    st.latex(r"T_{j,control} = \max(T_{j,main}, T_{j,diode})")
    st.write("也就是说，闭环模式下主芯片和二极管各自按单颗损耗回推结温，不再把整臂总损耗直接套给单颗热阻。")
    st.write("同时程序要求闭环至少完成 15 次以上迭代后，才允许根据收敛条件提前退出，以满足车规级稳态回推的保守性。")

    st.markdown("### 📑 8. 数据边界监视")
    st.latex(r"\mathrm{Flag}_{extra} = \mathbb{I}(I_{target}\notin[I_{min},I_{max}] \;\vee\; T_{target}\notin[T_{min},T_{max}])")
    st.write("程序会持续监视当前工作点是否超出原始矩阵边界，一旦进入外推区，就在结果页显式报警。")

    st.markdown("### 📐 9. STAR-CCM+ 热源映射")
    st.latex(r"q''' = \frac{P_{chip}}{V_{chip}}")
    st.latex(r"q'' = \frac{P_{chip}}{A_{chip}}")
    st.write("其中 $q'''$ 对应 STAR-CCM+ 的体积热源 $W/m^3$，$q''$ 对应表面热流 $W/m^2$。程序按你输入的单颗 die 长宽厚自动换算。")

    st.markdown("### 🧱 10. 工程约定（必须长期保留）")
    st.markdown(
        """
        - `R_pkg,chip` 默认 `0 mΩ`：只有当原始 V-I 数据没有吸收封装压降时才额外填写。
        - `R_arm` 默认 `0 mΩ`：只有当需要单独补偿外部铜排/连接片损耗时才填写。
        - `Regeneration` 用工况开关控制，不要靠手工把 `cos_phi` 输成负值来冒充反拖。
        - 所有结果默认表示 **单个半桥臂**，`Icepak` 热源表按 **单颗 die** 展开。
        - 若填入 `N_arm_sys > 1`，程序只对总量做对称桥臂缩放；单颗 Heat Generation Rate 不会变化。
        """
    )
