这就导致无论你在界面上把开关电阻填成多少，程序底层都当作 $1.0 / 1.0 = 1$ 给抵消掉了，**你千叮咛万嘱咐的 $K_{ron}$ 和 $K_{roff}$ 修正项被我彻底弄丢了！** 难怪你算出来的结果完全不对。

你教训得非常对：“没有我的命令不可以随意丢弃数据，关断电阻必须考虑进去”。我深刻反省，并**彻底回滚到了咱们最稳固的“单芯归一化（微观域）”底层架构**。这是最契合你们公司“以单芯片为基准对标”的数学模型。

这一次，所有的参数、防呆保护、矩阵监控、以及你强调的**“就近拟合动态电阻”**和**“温漂绝对受控”**，一字不漏地全部加回来了！

请 **Ctrl+A 全选** 覆盖你的 `app.py`：

```python
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD系统级电热仿真-物理完全体", layout="wide")

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

# 1. 归一化引擎（强制转化为单芯片纯净模型）
def normalize_vi_df(df: pd.DataFrame, n_src: int) -> pd.DataFrame:
    res_df = canonicalize_df_columns(df.dropna(how="all")).copy()
    if n_src > 1 and CURRENT_COL in res_df: res_df[CURRENT_COL] = res_df[CURRENT_COL] / float(n_src)
    return res_df

def normalize_ei_df(df: pd.DataFrame, n_src: int, e_cols: list[str]) -> pd.DataFrame:
    res_df = canonicalize_df_columns(df.dropna(how="all")).copy()
    if n_src > 1 and CURRENT_COL in res_df:
        res_df[CURRENT_COL] = res_df[CURRENT_COL] / float(n_src)
        for col in e_cols:
            if col in res_df.columns: res_df[col] = pd.to_numeric(res_df[col], errors='coerce') / float(n_src)
    return res_df

def safe_interp(df: pd.DataFrame, target_i: float, target_t: float, item_name: str) -> float:
    clean_df = canonicalize_df_columns(df.dropna(subset=[TEMP_COL, CURRENT_COL, item_name]))
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

# 2. 动态就近拟合引擎 (严格按照工况点找相邻测试点拉切线)
def get_bracketing_points(i_list, target_i):
    i_list = sorted(list(set(i_list)))
    if len(i_list) < 2: return (i_list[0], i_list[0]) if i_list else (1e-6, 1e-6)
    if target_i <= i_list[0]: return i_list[0], i_list[1]
    if target_i >= i_list[-1]: return i_list[-2], i_list[-1]
    for k in range(len(i_list)-1):
        if i_list[k] <= target_i <= i_list[k+1]:
            return i_list[k], i_list[k+1]
    return i_list[0], i_list[1]

def build_linearized_device_model(df: pd.DataFrame, target_i: float, target_t: float, item_name: str, force_zero_intercept: bool):
    clean_df = canonicalize_df_columns(df.dropna(subset=[CURRENT_COL, item_name]))
    i_list = clean_df[CURRENT_COL].unique() if not clean_df.empty else []
    
    i_low, i_high = get_bracketing_points(i_list, target_i)
    v_low = safe_interp(df, i_low, target_t, item_name)
    v_high = safe_interp(df, i_high, target_t, item_name)
    
    denom = i_high - i_low
    r_eq = max(0.0, (v_high - v_low) / denom) if denom > 1e-12 else 0.0
    v0 = 0.0 if force_zero_intercept else max(0.0, v_low - r_eq * i_low)
    
    return {"v_pk": v_high, "v_half": v_low, "r_eq": r_eq, "v0": v0, "i_low": i_low, "i_high": i_high}

# 3. 开关能量核心提取与物理修正 (参数全部接通！)
def calc_switching_energy(df: pd.DataFrame, i_pk: float, tj: float, algo_type: str, i_nom_chip: float, item_name: str, vdc: float, vref: float, kv: float, ract: float, rref: float, kr: float, temp_coeff: float, tref: float, ki_frd: float = 1.0, is_diode: bool = False) -> dict:
    if "直线比例法" in algo_type:
        nominal_curr = max(float(i_nom_chip), 1e-12)
        e_nom = safe_interp(df, nominal_curr, tj, item_name)
        # FRD采用非线性电流缩放 K_i_frd，IGBT采用线性
        current_factor = math.pow(max(float(i_pk), 0.0) / nominal_curr, ki_frd) if is_diode else (max(float(i_pk), 0.0) / nominal_curr)
        e_base = e_nom * current_factor
        extraction_label = f"比例法 (I_nom={nominal_curr:.1f}A)"
    else:
        # CAE二维插值自带非线性电流映射，无需额外施加 ki_frd
        e_base = safe_interp(df, i_pk, tj, item_name)
        clean_df = canonicalize_df_columns(df.dropna(subset=[CURRENT_COL, item_name]))
        i_list = clean_df[CURRENT_COL].unique() if not clean_df.empty else []
        i_low, i_high = get_bracketing_points(i_list, i_pk)
        extraction_label = f"二维插值 ({i_low:.1f}~{i_high:.1f}A)"

    # 坚决执行物理修正：温漂、电阻(Kron/Kroff)、电压
    temp_correction = max(0.0, 1.0 + float(temp_coeff) * (tj - tref))
    rg_correction = math.pow(max(ract, 1e-12) / max(rref, 1e-12), kr) if rref > 0 else 1.0
    voltage_correction = math.pow(max(vdc, 1e-12) / max(vref, 1e-12), kv) if vref > 0 else 1.0
    
    energy_mj = max(0.0, e_base * temp_correction * rg_correction * voltage_correction)
    return {"energy_mj": energy_mj, "e_base_mj": e_base, "extraction_label": extraction_label, "rg_corr": rg_correction, "v_corr": voltage_correction, "t_corr": temp_correction}

# 4. 导通损耗解析公式 (100% 对标手稿 129 / 131)
def calc_pwm_conduction_losses(mode: str, m_eff: float, active_cosphi: float, theta: float, i_pk_chip: float, main_model: dict, diode_model: dict):
    r_main_total = main_model["r_eq"]
    r_diode_total = diode_model["r_eq"]

    if mode == "SVPWM":
        # 绝对映射图 129：无 1/2pi，电阻项带 M
        kv0_m = (m_eff * active_cosphi) / 4.0
        kr_m = m_eff * (24.0 * active_cosphi - 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) - 3.0 * math.sqrt(3.0)) / 24.0
        kv0_d = (4.0 - m_eff * math.pi * active_cosphi) / 4.0
        kr_d = m_eff * (6.0 * math.pi - 24.0 * active_cosphi + 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) + 3.0 * math.sqrt(3.0)) / 24.0

        p_cond_main = (kv0_m * main_model["v0"] * i_pk_chip) + (kr_m * r_main_total * i_pk_chip**2) / math.pi
        p_cond_diode = (kv0_d * diode_model["v0"] * i_pk_chip) / math.pi + (kr_d * r_diode_total * i_pk_chip**2) / math.pi
    else:
        # SPWM 对标图 131
        p_cond_main = main_model["v0"] * i_pk_chip * (1.0 / (2.0 * math.pi) + m_eff * active_cosphi / 8.0) + r_main_total * i_pk_chip**2 * (1.0 / 8.0 + m_eff * active_cosphi / (3.0 * math.pi))
        p_cond_diode = diode_model["v0"] * i_pk_chip * (1.0 / (2.0 * math.pi) - m_eff * active_cosphi / 8.0) + r_diode_total * i_pk_chip**2 * (1.0 / 8.0 - m_eff * active_cosphi / (3.0 * math.pi))

    return {"p_cond_main": max(0.0, float(p_cond_main)), "p_cond_diode": max(0.0, float(p_cond_diode))}

# ================= UI 渲染与系统调度 =================
st.title("🛡️ 功率模块全工况电热联合仿真平台 (工程完全体)")

with st.expander("📝 核心物理准则记录 (不准删)"):
    st.markdown("""
    1. **完全解耦归一化**：输入规格书原测电流与能量，程序底层强制化为单晶圆（Bare Die）特征。
    2. **电阻精准补偿**：Eon 受 Kron 影响，Eoff 坚决受 Kroff 影响，绝不化简。
    3. **温漂绝对服从**：不再自作聪明拦截，只要你填入 Tc_igbt/Tc_frd，必定执行温漂放大；若公司没算温漂，请手动填 0。
    4. **就近拟合 V0/Req**：抛弃 0点连线，自动寻找距离工况电流最近的两个测试点拉切线，极大地提升了非线性区的导通压降精度。
    """)

with st.sidebar:
    st.header("⚙️ 核心技术架构")
    device_type = st.radio("1. 模块芯片技术类型", ["IGBT + FRD (传统硅基)", "SiC MOSFET (碳化硅)"])
    st.divider()
    st.header("🧮 原始数据规格 (必填)")
    cond_data_type = st.radio("A. 导通 V-I 表格代表：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"])
    n_src_cond = st.number_input("V-I 原测模块芯片数", value=2, min_value=1)
    sw_data_type = st.radio("B. 开关 E-I 表格代表：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"])
    n_src_sw = st.number_input("E-I 原测模块芯片数", value=2, min_value=1)
    st.divider()
    st.header("🎯 仿真目标规模重构")
    n_sim = st.number_input("目标仿真单臂芯片数 (N_sim)", value=2, min_value=1)
    st.divider()
    st.header("🔄 热学计算工作流")
    sim_mode = st.radio("模式选择", ["A. 开环盲算 (已知结温)", "B. 闭环迭代 (输入热阻反算)"])
    if "闭环" in sim_mode:
        rth_jc_main = st.number_input("主芯片 Rth (K/W)", value=0.135, format="%.4f")
        rth_jc_diode = st.number_input("二极管 Rth (K/W)", value=0.140, format="%.4f")
        t_case = st.number_input("参考水温 Tc (℃)", value=65.0)
        fixed_tj = 150.0 
    else:
        fixed_tj = st.number_input("设定全局目标结温 Tj (℃)", value=150.0)

st.divider()
st.header("📊 第一步：特性数据录入 (归一化中心)")
col_main, col_diode = st.columns(2)
with col_main:
    st.subheader("🔴 主开关管 (IGBT / SiC)")
    ev_main = st.data_editor(DEFAULT_MAIN_VI, num_rows="dynamic", key="v_main")
    ee_main = st.data_editor(DEFAULT_MAIN_EI, num_rows="dynamic", key="e_main")
with col_diode:
    st.subheader("🔵 续流二极管 (FRD / Body Diode)")
    ev_diode = st.data_editor(DEFAULT_DIODE_VI, num_rows="dynamic", key="v_diode")
    ee_diode = st.data_editor(DEFAULT_DIODE_EI, num_rows="dynamic", key="ee_diode")

st.divider()
st.header("⚙️ 第二步：全场景工况与物理修正系数")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**⚡ 车辆 / 电驱动工况**")
    op_mode = st.selectbox("🏎️ 运行场景切换", ["电动/巡航 (Motoring)", "制动/反拖 (Regeneration)"])
    vdc_act = st.number_input("实际母线 V_dc (V)", value=650.0)
    iout_rms = st.number_input("输出有效值 I_out (A)", value=285.0)
    fsw = st.number_input("开关频率 f_sw (Hz)", value=10000.0)
    m_index = st.number_input("调制系数 M", value=0.90)
    cosphi = st.number_input("功率因数 cos_phi", value=0.90)
    mode = st.selectbox("调制模式", ["SVPWM", "SPWM"])

with c2:
    st.markdown("**🔬 算法对标与测试基准**")
    algo_type = st.radio("开关能量提取算法", ["1. CAE二维动态插值", "2. 直线比例法 (对标公司)"])
    i_nom_ref = st.number_input("直线法基准电流(原表层级) (A)", value=400.0)
    v_ref = st.number_input("双脉冲测试 V_nom (V)", value=450.0)
    t_ref_dp = st.number_input("双脉冲测试 T_ref (℃)", value=25.0)

with c3:
    st.markdown("**📈 拟合修正指数**")
    kv_on = st.number_input("开通电压指数 K_v_on", value=1.30)
    kv_off = st.number_input("关断电压指数 K_v_off", value=1.30)
    kv_frd = st.number_input("续流电压指数 K_v_frd", value=0.60)
    ki_frd = st.number_input("续流电流指数 K_i_frd", value=0.60)
    kron = st.number_input("开通电阻系数 K_ron", value=0.30)
    kroff = st.number_input("关断电阻系数 K_roff", value=0.50)

with c4:
    st.markdown("**🌡️ 温漂与驱动电路**")
    t_coeff_igbt = st.number_input("主管温漂系数 (填0则无视温度)", value=0.0000, format="%.5f")
    t_coeff_frd = st.number_input("续流温漂系数", value=0.0000, format="%.5f")
    rg_on_ref = st.number_input("基准 R_g,on (Ω)", value=2.5)
    rg_off_ref = st.number_input("基准 R_g,off (Ω)", value=20.0)
    rg_on_act = st.number_input("实际 R_on (Ω)", value=2.5)
    rg_off_act = st.number_input("实际 R_off (Ω)", value=20.0)

st.divider()
if st.button("🚀 执 行 联 合 仿 真 计 算", use_container_width=True):
    # 1. 数据归一化降维 (化为单芯片)
    norm_ev_m, norm_ev_d = normalize_vi_df(ev_main, int(n_src_cond) if "Module" in cond_data_type else 1), normalize_vi_df(ev_diode, int(n_src_cond) if "Module" in cond_data_type else 1)
    norm_ee_m, norm_ee_d = normalize_ei_df(ee_main, int(n_src_sw) if "Module" in sw_data_type else 1, ["Eon (mJ)", "Eoff (mJ)"]), normalize_ei_df(ee_diode, int(n_src_sw) if "Module" in sw_data_type else 1, ["Erec (mJ)"])

    # 2. 计算工况电流 (严格聚焦于单芯片)
    i_pk_chip = math.sqrt(2.0) * (iout_rms / n_sim)
    i_nom_chip = i_nom_ref / int(n_src_sw) if "Module" in sw_data_type else i_nom_ref
    
    active_cosphi = -cosphi if "反拖" in op_mode else cosphi
    theta = math.acos(active_cosphi) if abs(active_cosphi) <= 1.0 else 0

    tj_main, tj_diode = fixed_tj, fixed_tj
    tolerance = 0.05
    loop_max = 30 if "闭环" in sim_mode else 1
    
    for loop in range(loop_max):
        # 3. 动态提取最新温度下的 V0 与 Req (就近拟合机制)
        main_model = build_linearized_device_model(norm_ev_m, i_pk_chip, tj_main, "V_drop (V)", "SiC" in device_type)
        diode_model = build_linearized_device_model(norm_ev_d, i_pk_chip, tj_diode, "Vf (V)", False)
        
        # 4. 执行导通与开关损耗计算 (参数无一遗漏传入！)
        cond_res = calc_pwm_conduction_losses(mode, m_index, active_cosphi, theta, i_pk_chip, main_model, diode_model)

        eon = calc_switching_energy(norm_ee_m, i_pk_chip, tj_main, algo_type, i_nom_chip, "Eon (mJ)", vdc_act, v_ref, kv_on, rg_on_act, rg_on_ref, kron, t_coeff_igbt, t_ref_dp, 1.0, False)
        eoff = calc_switching_energy(norm_ee_m, i_pk_chip, tj_main, algo_type, i_nom_chip, "Eoff (mJ)", vdc_act, v_ref, kv_off, rg_off_act, rg_off_ref, kroff, t_coeff_igbt, t_ref_dp, 1.0, False)
        erec = calc_switching_energy(norm_ee_d, i_pk_chip, tj_diode, algo_type, i_nom_chip, "Erec (mJ)", vdc_act, v_ref, kv_frd, 1.0, 1.0, 0.0, t_coeff_frd, t_ref_dp, ki_frd, True)

        p_sw_m_chip = (fsw / math.pi) * ((eon["energy_mj"] + eoff["energy_mj"]) / 1000.0)
        p_sw_d_chip = (fsw / math.pi) * (erec["energy_mj"] / 1000.0)

        p_total_main_chip = cond_res['p_cond_main'] + p_sw_m_chip
        p_total_diode_chip = cond_res['p_cond_diode'] + p_sw_d_chip
        
        # 5. 闭环迭代收敛判断
        if "闭环" in sim_mode:
            tj_main_new = t_case + p_total_main_chip * rth_jc_main
            tj_diode_new = t_case + p_total_diode_chip * rth_jc_diode
            if max(abs(tj_main_new - tj_main), abs(tj_diode_new - tj_diode)) < tolerance:
                tj_main, tj_diode = tj_main_new, tj_diode_new
                break
            tj_main, tj_diode = tj_main_new, tj_diode_new

    st.success(f"✅ 计算完成！采用算法：开关={eon['extraction_label']} | 导通V0/Req=基于 {main_model['i_low']:.1f}A~{main_model['i_high']:.1f}A 动态切线")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("稳定结温 (主/二极管)", f"{tj_main:.1f}℃ / {tj_diode:.1f}℃")
    m2.metric("单颗峰值电流 I_pk_chip", f"{i_pk_chip:.2f} A")
    m3.metric("🔴 主芯片单颗总耗散 (CAE用)", f"{p_total_main_chip:.2f} W")
    m4.metric("🔵 二极管单颗总耗散 (CAE用)", f"{p_total_diode_chip:.2f} W")
    
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("单颗主开关 - 导通损耗", f"{cond_res['p_cond_main']:.2f} W")
    c2.metric("单颗主开关 - 开关损耗", f"{p_sw_m_chip:.2f} W")
    c3.metric("单颗二极管 - 导通损耗", f"{cond_res['p_cond_diode']:.2f} W")
    c4.metric("单颗二极管 - 恢复损耗", f"{p_sw_d_chip:.2f} W")
    
    st.info(f"🚀 **系统级重构验证：** 目标单臂（{n_sim} 颗并联）主开关总损耗为 **{p_total_main_chip * n_sim:.1f} W**；二极管单臂总损耗为 **{p_total_diode_chip * n_sim:.1f} W**。")

    with st.expander("🔍 物理修正链路追溯日志"):
        st.write(f"- **Eon 修正系数链**：温漂=×{eon['t_corr']:.3f} | 电阻=×{eon['rg_corr']:.3f} | 电压=×{eon['v_corr']:.3f}")
        st.write(f"- **Eoff 修正系数链**：温漂=×{eoff['t_corr']:.3f} | 电阻=×{eoff['rg_corr']:.3f} | 电压=×{eoff['v_corr']:.3f}")
        st.write(f"- **Erec 修正系数链**：温漂=×{erec['t_corr']:.3f} | 电压=×{erec['v_corr']:.3f}")
        st.write(f"- **导通等效特征**：主开关 V0={main_model['v0']:.3f}V, Req={main_model['r_eq']:.5f}Ω | 二极管 V0={diode_model['v0']:.3f}V, Req={diode_model['r_eq']:.5f}Ω")
