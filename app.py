import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD模块全维仿真-防忘备注版", layout="wide")
st.title("🛡️ 功率模块多架构电热联合仿真平台 (带备注专家版)")

# ==========================================
# 侧边栏: 全局架构与目标定义
# ==========================================
with st.sidebar:
    st.header("⚙️ 核心技术架构")
    device_type = st.radio("1. 模块芯片技术类型", ["IGBT + FRD (传统硅基)", "SiC MOSFET (碳化硅)"], 
                           help="选择 IGBT 会启用 FRD 二极管损耗计算；选择 SiC 会启用体二极管逻辑并强制 Vce0 = 0V")
    
    st.divider()
    st.header("🧮 原始数据规格定义")
    st.caption("⚠️ 此处需告知程序：你右侧填写的表格代表什么层级的数据")
    
    cond_data_type = st.radio("A. 导通 V-I 表格来源：", ["单芯片数据 (Bare Die)", "模块半桥数据 (Module)"],
                             help="若填入的是裸片测试数据选单芯；若填入整模块端子测试数据选模块")
    n_src_cond = st.number_input("V-I 原测模块芯片数", value=6, min_value=1, 
                                 help="若上方选单芯，此处填 1；若选模块，填入该模块内部并联的芯片数")
    
    sw_data_type = st.radio("B. 开关 E-I 表格来源：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"],
                            help="双脉冲测试通常为整模块数据")
    n_src_sw = st.number_input("E-I 原测模块芯片数", value=6, min_value=1,
                               help="填入双脉冲测试时该模块内部并联的芯片数")

    st.divider()
    st.header("🎯 仿真目标重构")
    n_sim = st.number_input("目标仿真芯片数 (N_sim)", value=6, min_value=1, 
                            help="【扩容/减容核心】你想评估多少并联芯片的模块？填入此数，损耗会自动按此规模折算")
        st.divider()
    st.header("🔄 热学计算工作流")
    sim_mode = st.radio("模式选择", ["A. 开环盲算 (已知结温)", "B. 闭环迭代 (已知热阻)"])
    if "闭环" in sim_mode:
        rth_jc = st.number_input("封装热阻 RthJC (K/W)", value=0.065, format="%.4f", help="芯片结到壳的热阻")
        t_case = st.number_input("基板/水温 Tc (℃)", value=65.0, help="散热系统的冷却温度")
    else:
        fixed_tj = st.number_input("设定全局目标结温 Tj (℃)", value=150.0, help="在此固定温度下评估损耗")

# ==========================================
# 第一部分: 数据录入
# ==========================================
st.divider()
st.header("📊 第一步：特性数据录入")
st.info("💡 **填表说明**：支持从 Excel 复制粘贴 (Ctrl+C -> Ctrl+V)。程序会根据左侧定义的【原始数据规格】自动将数据归一化为单芯片模型。")

col_t, col_d = st.columns(2)
with col_t:
    st.subheader("🔴 主开关管 (IGBT / SiC)")
    v_df_main = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'V_drop (V)': [1.1, 1.05, 2.2, 2.5]})
    st.caption("1. 导通压降 Vce / Vds (横轴：电流，纵轴：压降)")
    ev_main = st.data_editor(v_df_main, num_rows="dynamic", key="v_main")
    
    e_df_main = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Eon (mJ)': [5.9, 8.5, 70.0, 95.0], 'Eoff (mJ)': [4.9, 7.2, 45.0, 60.0]})
    st.caption("2. 开关能量矩阵 (对应双脉冲原始数据)")
    ee_main = st.data_editor(e_df_main, num_rows="dynamic", key="e_main")

with col_d:
    st.subheader("🔵 续流二极管 (FRD / Body Diode)")
    v_df_diode = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Vf (V)': [1.2, 1.1, 2.0, 2.2]})
    st.caption("1. 正向压降 Vf / Vsd")
    ev_diode = st.data_editor(v_df_diode, num_rows="dynamic", key="v_diode")
    
    e_df_diode = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Erec (mJ)': [1.9, 3.5, 15.0, 25.0]})
    st.caption("2. 反向恢复能量 Erec")
    ee_diode = st.data_editor(e_df_diode, num_rows="dynamic", key="e_diode")

# ==========================================
# 第二部分: 工况与全维物理系数
# ==========================================
st.divider()
st.header("⚙️ 第二步：工况与物理修正系数配置")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**⚡ 运行工况**")
    vdc_act = st.number_input("实际母线 V_dc (V)", value=713.0, help="逆变器运行的直流电压")
    iout_rms = st.number_input("输出有效值 I_out (A)", value=264.5, help="三相交流输出电流的有效值")
    fsw = st.number_input("开关频率 f_sw (Hz)", value=10000, help="载波频率")
    m_index = st.number_input("调制系数 M", value=0.90, min_value=0.0, max_value=1.1, help="调制度")
    cosphi = st.number_input("功率因数 cos_phi", value=0.90, min_value=-1.0, max_value=1.0, help="负载电流相位差余弦值")
    mode = st.selectbox("调制模式", ["SVPWM", "SPWM"], help="空间矢量调制或正弦脉宽调制")

with c2:
    st.markdown("**📐 基准与电阻修正**")
    v_ref = st.number_input("规格书测试 V_nom (V)", value=600.0, help="测双脉冲时的基准母线电压")
    t_ref_dp = st.number_input("规格书测试 T_ref_dp (℃)", value=150.0, help="测双脉冲时的基准结温")
    rg_on_ref = st.number_input("手册基准 R_g,on (Ω)", value=2.5, help="规格书测试 Eon 用的电阻")
    rg_off_ref = st.number_input("手册基准 R_g,off (Ω)", value=20.0, help="规格书测试 Eoff 用的电阻")
    rg_on_act = st.number_input("实际应用 R_on (Ω)", value=2.5, help="你电路板上实际使用的开通电阻")
    rg_off_act = st.number_input("实际应用 R_off (Ω)", value=20.0, help="你电路板上实际使用的关断电阻")

with c3:
    st.markdown("**📈 电压与电流指数**")
    kv_on = st.number_input("开通电压指数 K_v_on", value=1.30, help="能量随电压变化的敏感度，通常 1.2-1.4")
    kv_off = st.number_input("关断电压指数 K_v_off", value=1.30)
    kv_frd = st.number_input("续流电压指数 K_v_frd", value=0.60)
    ki_frd = st.number_input("续流电流指数 K_i_frd", value=0.60)
    
    st.markdown("**🧱 结构参数**")
    r_arm_mohm = st.number_input("桥臂内阻 R_arm (mΩ)", value=0.0, 
                                 help="【默认为 0】若对标纯芯片则填 0；若考虑模块端子压降则填入封装内阻")

with c4:
    st.markdown("**🌡️ 电阻与温度系数**")
    kron = st.number_input("开通电阻系数 K_ron", value=0.30, help="电阻对开通损耗的影响指数")
    kroff = st.number_input("关断电阻系数 K_roff", value=0.50, help="电阻对关断损耗的影响指数")
    t_coeff_igbt = st.number_input("主管温漂 T_c_igbt", value=0.003, format="%.4f", help="IGBT/SiC 能量随温升的线性比例")
    t_coeff_frd = st.number_input("续流温漂 T_cerr", value=0.006 if "IGBT" in device_type else 0.003, format="%.4f", 
                                  help="二极管反向恢复能量温漂系数")

# ==========================================
# 第三部分: 核心算法与求解 (带归一化逻辑)
# ==========================================
st.divider()
st.header("🚀 第三步：执行联合仿真")

def normalize_vi_df(df, n_src):
    res_df = df.copy()
    if n_src > 1: res_df['Current (A)'] = res_df['Current (A)'] / float(n_src)
    return res_df

def normalize_ei_df(df, n_src, e_cols):
    res_df = df.copy()
    if n_src > 1:
        res_df['Current (A)'] = res_df['Current (A)'] / float(n_src)
        for col in e_cols:
            if col in res_df.columns: res_df[col] = res_df[col] / float(n_src)
    return res_df

def safe_interp(df, target_i, target_t, item_name):
    clean_df = df.dropna()
    if clean_df.empty: return 0.0
    temp_list, val_list = [], []
    for temp, group in clean_df.groupby('Temp (℃)'):
        sorted_g = group.sort_values('Current (A)')
        if len(sorted_g) >= 2:
            f = interp1d(sorted_g['Current (A)'], sorted_g[item_name], kind='linear', fill_value="extrapolate")
            val_list.append(max(0.0, float(f(target_i))))
            temp_list.append(temp)
        elif len(sorted_g) == 1:
            val_list.append(max(0.0, float(sorted_g[item_name].iloc[0])))
            temp_list.append(temp)
    if len(temp_list) >= 2:
        return max(0.0, float(interp1d(temp_list, val_list, fill_value="extrapolate")(target_t)))
    elif len(temp_list) == 1: return val_list[0]
    return 0.0

def calc_sw_power_analytical(df, i_pk_lookup, tj, item_name, fsw, vdc_act, v_ref, kv, r_act, r_ref, k_r, t_coeff, t_ref_dp):
    e_base_pk = safe_interp(df, i_pk_lookup, tj, item_name)
    clean_df = df.dropna()
    active_t_coeff = 0.0 if (not clean_df.empty and len(clean_df['Temp (℃)'].unique()) >= 2) else t_coeff
    r_corr = math.pow(r_act / r_ref, k_r) if r_ref > 0 else 1.0
    t_corr = 1.0 + active_t_coeff * (tj - t_ref_dp)
    return (1.0 / math.pi) * fsw * (e_base_pk * r_corr * t_corr / 1000.0) * ((vdc_act / v_ref) ** kv)

if st.button("🚀 执 行 全 参 数 仿 真 计 算 (一镜到底)", use_container_width=True):
    # 底层归一化
    norm_ev_main = normalize_vi_df(ev_main, n_src_cond)
    norm_ev_diode = normalize_vi_df(ev_diode, n_src_cond)
    norm_ee_main = normalize_ei_df(ee_main, n_src_sw, ['Eon (mJ)', 'Eoff (mJ)'])
    norm_ee_diode = normalize_ei_df(ee_diode, n_src_sw, ['Erec (mJ)'])
    # 目标层级重构
    i_pk_chip = math.sqrt(2) * (iout_rms / n_sim)
    theta = math.acos(cosphi)
    r_arm_chip_equiv = (r_arm_mohm / 1000.0) * n_sim
    
    loop_count = 1 if "开环" in sim_mode else 15
    tj_current = fixed_tj if "开环" in sim_mode else t_case + 5.0

    for _ in range(loop_count):
        # 1. 单芯参数提取
        v_pk_m = safe_interp(norm_ev_main, i_pk_chip, tj_current, 'V_drop (V)')
        v_hf_m = safe_interp(norm_ev_main, i_pk_chip/2, tj_current, 'V_drop (V)')
        r_m_chip = (v_pk_m - v_hf_m) / (i_pk_chip / 2) if i_pk_chip > 0 else 0
        v0_m = 0.0 if "SiC" in device_type else max(0.0, v_pk_m - r_m_chip * i_pk_chip)
        r_m_total = r_m_chip + r_arm_chip_equiv

        v_pk_d = safe_interp(norm_ev_diode, i_pk_chip, tj_current, 'Vf (V)')
        v_hf_d = safe_interp(norm_ev_diode, i_pk_chip/2, tj_current, 'Vf (V)')
        r_d_chip = (v_pk_d - v_hf_d) / (i_pk_chip / 2) if i_pk_chip > 0 else 0
        v0_d = max(0.0, v_pk_d - r_d_chip * i_pk_chip)
        r_d_total = r_d_chip + r_arm_chip_equiv

        # 2. 解析公式积分 (单芯级)
        if mode == "SVPWM":
            p_cond_v0_m = (m_index * i_pk_chip * v0_m * cosphi) / 4.0
            coeff_r_m = (24*cosphi - 2*math.sqrt(3)*math.cos(2*theta) - 3*math.sqrt(3)) / 24.0
            p_cond_main_chip = p_cond_v0_m + coeff_r_m * (m_index * r_m_total * i_pk_chip**2) / math.pi
            
            p_cond_v0_d = ((4 - m_index * math.pi * cosphi) / 4.0) * (v0_d * i_pk_chip) / math.pi
            coeff_r_d = (6*math.pi - 24*m_index*cosphi + 2*math.sqrt(3)*m_index*math.cos(2*theta) + 3*math.sqrt(3)*m_index) / 24.0
            p_cond_diode_chip = p_cond_v0_d + coeff_r_d * (r_d_total * i_pk_chip**2) / math.pi
        else:
            p_cond_main_chip = v0_m * i_pk_chip * (1/(2*math.pi) + m_index*cosphi/8) + r_m_total * i_pk_chip**2 * (1/8 + m_index*cosphi/(3*math.pi))
            p_cond_diode_chip = v0_d * i_pk_chip * (1/(2*math.pi) - m_index*cosphi/8) + r_d_total * i_pk_chip**2 * (1/8 - m_index*cosphi/(3*math.pi))

        p_on_chip = calc_sw_power_analytical(norm_ee_main, i_pk_chip, tj_current, 'Eon (mJ)', fsw, vdc_act, v_ref, kv_on, rg_on_act, rg_on_ref, kron, t_coeff_igbt, t_ref_dp)
        p_off_chip = calc_sw_power_analytical(norm_ee_main, i_pk_chip, tj_current, 'Eoff (mJ)', fsw, vdc_act, v_ref, kv_off, rg_off_act, rg_off_ref, kroff, t_coeff_igbt, t_ref_dp)
        
        i_corr_factor = (iout_rms / ( (i_pk_chip*n_sim) / math.sqrt(2))) ** ki_frd if ki_frd > 0 else 1.0
        p_sw_diode_chip = calc_sw_power_analytical(norm_ee_diode, i_pk_chip, tj_current, 'Erec (mJ)', fsw, vdc_act, v_ref, kv_frd, 1.0, 1.0, 0.0, t_coeff_frd, t_ref_dp) * i_corr_factor
                # 3. 结果放大至目标级
        p_total = (p_cond_main_chip + p_on_chip + p_off_chip + p_cond_diode_chip + p_sw_diode_chip) * n_sim
        
        if "闭环" in sim_mode:
            tj_new = t_case + p_total * rth_jc
            if abs(tj_new - tj_current) < 0.05: break
            tj_current = tj_new

    # --- 最终面板 ---
    st.success(f"✅ 计算完成！已基于归一化单芯模型重构为 **{n_sim}芯** 模块损耗数据。")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("仿真收敛结温 Tj", f"{tj_current:.1f} ℃")
    r2.metric("主开关导通总损耗", f"{p_cond_main_chip*n_sim:.1f} W")
    r3.metric("主开关动态总损耗", f"{(p_on_chip+p_off_chip)*n_sim:.1f} W")
    r4.metric("模块桥臂总功耗", f"{p_total:.1f} W")

    st.divider()
    with st.expander("🔬 查看底层计算逻辑与应用的数学公式 (永久对标手册)"):
        st.markdown("#### 1. 导通损耗 (Conduction Loss)")
        if mode == "SVPWM":
            st.write("已启用 **SVPWM** 空间矢量调制马鞍波解析积分：")
            st.latex(r"K_{v0} = \frac{M \cos\phi}{4}, \quad K_r = \frac{24\cos\phi - 2\sqrt{3}\cos(2\varphi) - 3\sqrt{3}}{24}")
        else:
            st.write("已启用 **SPWM** 正弦波调制解析积分：")
            st.latex(r"K_{v0} = \frac{1}{2\pi} + \frac{M\cos\phi}{8}, \quad K_r = \frac{1}{8} + \frac{M\cos\phi}{3\pi}")
        st.write(f"注：当前仿真内阻 $R_{{arm}} = {r_arm_mohm} \text{{ m}}\Omega$ 已叠加至单芯电阻项进行综合积分。")
            
        st.markdown("#### 2. 开关损耗 (Switching Loss)")
        st.write("解析积分公式（1/π 线性近似推导）：")
        st.latex(r"P_{sw} = \frac{1}{\pi} f_{sw} \cdot E(I_{pk}) \cdot \left(\frac{R_{act}}{R_{ref}}\right)^{K_r} \cdot \left[1 + T_c(T_j - T_{ref\_dp})\right] \cdot \left(\frac{V_{dc}}{V_{ref}}\right)^{K_v}")
        st.info("智能温漂锁说明：若数据表格输入了多温维度，Tc 系数将自动置 0，以插值结果为准。")
        
