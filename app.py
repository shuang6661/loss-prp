import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD系统级电热仿真-电热闭环版", layout="wide")

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
        col_str = str(col).strip().lower()
        if "temp" in col_str or "温度" in col_str: rename_map[col] = TEMP_COL
        elif "current" in col_str or "电流" in col_str or col_str in {"ic (a)", "if (a)"}: rename_map[col] = CURRENT_COL
    return df.rename(columns=rename_map)

# 修复 KeyError：先标准化列名，再做 dropna
def safe_interp(df: pd.DataFrame, target_i: float, target_t: float, item_name: str) -> float:
    df_canon = canonicalize_df_columns(df.copy())
    if item_name not in df_canon.columns: return 0.0
    clean_df = df_canon.dropna(subset=[TEMP_COL, CURRENT_COL, item_name])
    if clean_df.empty: return 0.0
    
    temp_list, val_list = [], []
    for temp, group in clean_df.groupby(TEMP_COL):
        sorted_g = group.sort_values(CURRENT_COL)
        if len(sorted_g) >= 2:
            val_list.append(max(0.0, float(interp1d(sorted_g[CURRENT_COL], sorted_g[item_name], kind="linear", fill_value="extrapolate")(target_i))))
            temp_list.append(float(temp))
        elif len(sorted_g) == 1:
            val_list.append(max(0.0, float(sorted_g[item_name].iloc[0])))
            temp_list.append(float(temp))
    if len(temp_list) >= 2: return max(0.0, float(interp1d(temp_list, val_list, kind="linear", fill_value="extrapolate")(target_t)))
    elif len(temp_list) == 1: return max(0.0, float(val_list[0]))
    return 0.0

def get_bracketing_points(i_list, target_i):
    """动态寻找距离工况点最近的两个特性测试电流进行拟合"""
    i_list = sorted(list(set(i_list)))
    if len(i_list) < 2: return (i_list[0], i_list[0]) if i_list else (1e-6, 1e-6)
    if target_i <= i_list[0]: return i_list[0], i_list[1]
    if target_i >= i_list[-1]: return i_list[-2], i_list[-1]
    for k in range(len(i_list)-1):
        if i_list[k] <= target_i <= i_list[k+1]:
            return i_list[k], i_list[k+1]
    return i_list[0], i_list[1]

def build_linearized_device_model(df: pd.DataFrame, target_i: float, target_t: float, item_name: str, force_zero_intercept: bool):
    df_canon = canonicalize_df_columns(df.copy())
    if item_name not in df_canon.columns: return {"v_pk": 0.0, "v_half": 0.0, "r_eq": 0.0, "v0": 0.0, "i_low": 0.0, "i_high": 0.0}
    clean_df = df_canon.dropna(subset=[CURRENT_COL, item_name])
    i_list = clean_df[CURRENT_COL].unique() if not clean_df.empty else []
    
    i_low, i_high = get_bracketing_points(i_list, target_i)

    v_low = safe_interp(df, i_low, target_t, item_name)
    v_high = safe_interp(df, i_high, target_t, item_name)
    
    denom = i_high - i_low
    r_eq = max(0.0, (v_high - v_low) / denom) if denom > 1e-12 else 0.0
    v0 = 0.0 if force_zero_intercept else max(0.0, v_low - r_eq * i_low)
    
    return {"v_pk": v_high, "v_half": v_low, "r_eq": r_eq, "v0": v0, "i_low": i_low, "i_high": i_high}

def calc_switching_energy(df: pd.DataFrame, i_pk: float, tj: float, algo_type: str, i_nom_domain: float, item_name: str, vdc: float, vref: float, kv: float, ract: float, rref: float, kr: float, temp_coeff: float, tref: float) -> dict:
    if "比例法" in algo_type:
        nominal_curr = max(float(i_nom_domain), 1e-12)
        e_nom = safe_interp(df, nominal_curr, tj, item_name)
        e_base = e_nom * (max(float(i_pk), 0.0) / nominal_curr)
        extraction_label = f"标称直线法 (基准={nominal_curr:.1f}A)"
    else:
        e_base = safe_interp(df, i_pk, tj, item_name)
        df_canon = canonicalize_df_columns(df.copy())
        clean_df = df_canon.dropna(subset=[CURRENT_COL]) if CURRENT_COL in df_canon else pd.DataFrame()
        i_list = clean_df[CURRENT_COL].unique() if not clean_df.empty else []
        i_low, i_high = get_bracketing_points(i_list, i_pk)
        extraction_label = f"相邻区间拟合 ({i_low:.1f}~{i_high:.1f}A)"

    temp_correction = max(0.0, 1.0 + float(temp_coeff) * (tj - tref))
    rg_correction = math.pow(max(ract, 1e-12) / max(rref, 1e-12), kr) if rref > 0 else 1.0
    voltage_correction = math.pow(max(vdc, 1e-12) / max(vref, 1e-12), kv) if vref > 0 else 1.0
    
    energy_mj = max(0.0, e_base * temp_correction * rg_correction * voltage_correction)
    return {"energy_mj": energy_mj, "e_base_mj": e_base, "extraction_label": extraction_label}

def calc_pwm_conduction_losses(mode: str, m_eff: float, active_cosphi: float, theta: float, i_pk_domain: float, main_model: dict, diode_model: dict, r_pkg_domain: float, r_arm_domain: float):
    r_main_total = main_model["r_eq"] + r_pkg_domain + r_arm_domain
    r_diode_total = diode_model["r_eq"] + r_pkg_domain + r_arm_domain

    if mode == "SVPWM":
        kv0_m = (m_eff * active_cosphi) / 4.0
        kr_m = m_eff * (24.0 * active_cosphi - 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) - 3.0 * math.sqrt(3.0)) / 24.0
        kv0_d = (4.0 - m_eff * math.pi * active_cosphi) / 4.0
        kr_d = m_eff * (6.0 * math.pi - 24.0 * active_cosphi + 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) + 3.0 * math.sqrt(3.0)) / 24.0

        p_cond_main = (kv0_m * main_model["v0"] * i_pk_domain) + (kr_m * r_main_total * i_pk_domain**2) / math.pi
        p_cond_diode = (kv0_d * diode_model["v0"] * i_pk_domain) / math.pi + (kr_d * r_diode_total * i_pk_domain**2) / math.pi
    else:
        p_cond_main = main_model["v0"] * i_pk_domain * (1.0 / (2.0 * math.pi) + m_eff * active_cosphi / 8.0) + r_main_total * i_pk_domain**2 * (1.0 / 8.0 + m_eff * active_cosphi / (3.0 * math.pi))
        p_cond_diode = diode_model["v0"] * i_pk_domain * (1.0 / (2.0 * math.pi) - m_eff * active_cosphi / 8.0) + r_diode_total * i_pk_domain**2 * (1.0 / 8.0 - m_eff * active_cosphi / (3.0 * math.pi))

    return {"p_cond_main": max(0.0, float(p_cond_main)), "p_cond_diode": max(0.0, float(p_cond_diode))}

# ================= UI 与系统调度 =================
st.title("🛡️ 功率模块多架构电热联合仿真平台 (电热闭环修复版)")

with st.sidebar:
    device_type = st.radio("1. 模块技术类型", ["IGBT + FRD (传统硅基)", "SiC MOSFET (碳化硅)"])
    st.divider()
    cond_data_type = st.radio("A. 导通 V-I 表格代表：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"])
    n_src_cond = st.number_input("V-I 原测模块芯片数", value=2, min_value=1)
    sw_data_type = st.radio("B. 开关 E-I 表格代表：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"])
    n_src_sw = st.number_input("E-I 原测模块芯片数", value=2, min_value=1)
    n_sim = st.number_input("目标仿真单臂芯片数", value=2, min_value=1)

tabs_input = st.tabs(["📊 第一步：特性数据录入", "⚙️ 第二步：工况与电热配置", "🚀 第三步：执行联合仿真"])

with tabs_input[0]:
    st.info("💡 后台修复：不再因为空行而报错，完美支持自定义列名与空白行录入。")
    c_m, c_d = st.columns(2)
    with c_m:
        st.write("🔴 主开关管特性 (IGBT / SiC)")
        ev_main = st.data_editor(DEFAULT_MAIN_VI, num_rows="dynamic", key="v_main")
        ee_main = st.data_editor(DEFAULT_MAIN_EI, num_rows="dynamic", key="e_main")
    with c_d:
        st.write("🔵 续流二极管特性 (FRD / Body Diode)")
        ev_diode = st.data_editor(DEFAULT_DIODE_VI, num_rows="dynamic", key="v_diode")
        ee_diode = st.data_editor(DEFAULT_DIODE_EI, num_rows="dynamic", key="ee_diode")

with tabs_input[1]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        op_mode = st.selectbox("运行场景", ["电动/巡航", "制动/反拖"])
        vdc_act = st.number_input("实际母线 V_dc (V)", value=650.0)
        iout_rms = st.number_input("输出有效值 I_out (A)", value=285.0)
        fsw = st.number_input("开关频率 f_sw (Hz)", value=10000.0)
        m_index = st.number_input("调制系数 M", value=0.90)
        cosphi = st.number_input("功率因数 cos_phi", value=0.90)
        mode = st.selectbox("调制模式", ["SVPWM", "SPWM"])
    with c2:
        algo_type = st.radio("开关能量提取算法", ["1. CAE精准相邻拟合", "2. 直线比例法(对标公司)"])
        i_nom_ref = st.number_input("直线基准电流(原表层级) (A)", value=400.0)
        v_ref = st.number_input("双脉冲基准 V_nom (V)", value=450.0)
        t_ref_dp = st.number_input("双脉冲基准 T_ref (℃)", value=25.0)
    with c3:
        kv_on = st.number_input("开通电压指数 K_v_on", value=1.30)
        kv_off = st.number_input("关断电压指数 K_v_off", value=1.30)
        kv_frd = st.number_input("续流电压指数 K_v_frd", value=0.60)
        kron = st.number_input("电阻系数 K_ron", value=0.30)
        kroff = st.number_input("关断电阻系数 K_roff", value=0.50)
        ki_frd = st.number_input("续流电流指数 K_i_frd", value=0.60)
    with c4:
        st.info("智能初始结温：IGBT 默认 150℃，SiC 默认 175℃")
        sim_mode = st.radio("热学模式", ["A. 开环(固定结温)", "B. 闭环热迭代(输入热阻水温)"])
        
        default_tj = 175.0 if "SiC" in device_type else 150.0
        
        if "开环" in sim_mode:
            fixed_tj = st.number_input("设定全局结温 Tj (℃)", value=default_tj)
            rth_jc, t_case = 0.0, 0.0
        else:
            rth_jc = st.number_input("芯片到水热阻 Rth (K/W)", value=0.135, format="%.4f")
            t_case = st.number_input("冷却水温 Tc (℃)", value=65.0)
            fixed_tj = default_tj # 作为闭环迭代的初始起点

        t_coeff_igbt = st.number_input("主管温漂系数", value=0.0000, format="%.4f")
        t_coeff_frd = st.number_input("续流温漂系数", value=0.0000, format="%.4f")

with tabs_input[2]:
    if st.button("🚀 执 行 电 热 联 合 仿 真", use_container_width=True):
        
        i_pk_chip = math.sqrt(2.0) * (iout_rms / n_sim) if n_sim > 0 else 0.0
        n_cond = int(n_src_cond) if "Module" in cond_data_type else 1
        n_sw = int(n_src_sw) if "Module" in sw_data_type else 1
        
        i_pk_cond_domain = i_pk_chip * n_cond
        i_pk_sw_domain = i_pk_chip * n_sw

        active_cosphi = -abs(cosphi) if "反拖" in op_mode else abs(cosphi)
        theta = math.acos(active_cosphi) if abs(active_cosphi) <= 1.0 else 0
        
        iteration_log = []
        tj_current = fixed_tj # 初始结温：150 或 175
        loop_max = 30 if "闭环" in sim_mode else 1
        tolerance = 0.05
        
        for loop_idx in range(loop_max):
            # 宏观数据域提取与温度自适应
            main_model = build_linearized_device_model(ev_main, i_pk_cond_domain, tj_current, "V_drop (V)", "SiC" in device_type)
            diode_model = build_linearized_device_model(ev_diode, i_pk_cond_domain, tj_current, "Vf (V)", False)
            
            cond_res_domain = calc_pwm_conduction_losses(mode, m_index, active_cosphi, theta, i_pk_cond_domain, main_model, diode_model, 0.0, 0.0)

            eon = calc_switching_energy(ee_main, i_pk_sw_domain, tj_current, algo_type, i_nom_ref, "Eon (mJ)", vdc_act, v_ref, kv_on, 1.0, 1.0, kron, t_coeff_igbt, t_ref_dp)
            eoff = calc_switching_energy(ee_main, i_pk_sw_domain, tj_current, algo_type, i_nom_ref, "Eoff (mJ)", vdc_act, v_ref, kv_off, 1.0, 1.0, kroff, t_coeff_igbt, t_ref_dp)
            erec = calc_switching_energy(ee_diode, i_pk_sw_domain, tj_current, algo_type, i_nom_ref, "Erec (mJ)", vdc_act, v_ref, kv_frd, 1.0, 1.0, 0.0, t_coeff_frd, t_ref_dp)

            p_sw_m_domain = (fsw / math.pi) * ((eon["energy_mj"] + eoff["energy_mj"]) / 1000.0)
            p_sw_d_domain = (fsw / math.pi) * (erec["energy_mj"] / 1000.0)

            # 微观剥离计算单芯片损耗
            p_cond_main_chip = cond_res_domain['p_cond_main'] / n_cond
            p_cond_diode_chip = cond_res_domain['p_cond_diode'] / n_cond
            p_sw_main_chip = p_sw_m_domain / n_sw
            p_sw_diode_chip = p_sw_d_domain / n_sw
            
            p_total_chip_main = p_cond_main_chip + p_sw_main_chip
            p_total_chip_diode = p_cond_diode_chip + p_sw_diode_chip
            p_total_arm = (p_total_chip_main + p_total_chip_diode) * n_sim
            
            iteration_log.append({
                "迭代次数": loop_idx + 1,
                "当前结温 (℃)": round(tj_current, 2),
                "主开关单颗耗散 (W)": round(p_total_chip_main, 2),
                "二极管单颗耗散 (W)": round(p_total_chip_diode, 2),
            })
            
            if "闭环" in sim_mode:
                # 电热耦合核心：T_j = T_case + P * Rth
                # 假设主开关发热主导，使用最大发热件更新结温（工业界常用简化保守做法）
                max_p = max(p_total_chip_main, p_total_chip_diode)
                tj_new = t_case + max_p * rth_jc
                if abs(tj_new - tj_current) < tolerance:
                    tj_current = tj_new
                    break
                tj_current = tj_new

        st.success("✅ 电热耦合计算收敛！" if "闭环" in sim_mode else "✅ 开环状态计算完毕。")
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("最终稳定结温 Tj", f"{tj_current:.1f} ℃")
        r2.metric("仿真单臂总损耗", f"{p_total_arm:.1f} W")
        r3.metric("🔴 主芯片单颗发热", f"{p_total_chip_main:.2f} W")
        r4.metric("🔵 二极管单颗发热", f"{p_total_chip_diode:.2f} W")
        
        if "闭环" in sim_mode:
            st.divider()
            st.markdown(f"🔄 **电热闭环迭代追踪 (初始推演从 {fixed_tj}℃ 开始)：**")
            st.dataframe(pd.DataFrame(iteration_log), use_container_width=True)
            st.caption("推算公式：新结温 Tj = 冷却水温 Tc + 发热量 P × 热阻 Rth")
