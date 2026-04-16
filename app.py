import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD系统级电热仿真-全工况版", layout="wide")
st.title("🛡️ 功率模块全工况电热联合仿真平台 (系统级架构版)")

# ==========================================
# 侧边栏: 全局架构与目标定义
# ==========================================
with st.sidebar:
    st.header("⚙️ 核心技术架构")
    device_type = st.radio("1. 模块芯片技术类型", ["IGBT + FRD (传统硅基)", "SiC MOSFET (碳化硅)"])
    
    st.divider()
    st.header("🧮 原始数据规格定义")
    st.caption("⚠️ 告知程序：右侧录入的表格代表什么层级")
    cond_data_type = st.radio("A. 导通 V-I 表格来源：", ["单芯片数据 (Bare Die)", "模块半桥数据 (Module)"])
    n_src_cond = st.number_input("V-I 原测模块芯片数", value=6, min_value=1) if "模块" in cond_data_type else 1
    
    sw_data_type = st.radio("B. 开关 E-I 表格来源：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"])
    n_src_sw = st.number_input("E-I 原测模块芯片数", value=6, min_value=1) if "模块" in sw_data_type else 1

    st.divider()
    st.header("🎯 仿真目标重构")
    n_sim = st.number_input("目标仿真芯片数 (N_sim)", value=6, min_value=1, help="评估不同并联数量(扩容/缩水)的损耗")

    st.divider()
    st.header("🔄 热学计算工作流")
    sim_mode = st.radio("模式选择", ["A. 开环盲算 (已知结温)", "B. 闭环迭代 (已知热阻)"])
    if "闭环" in sim_mode:
        rth_jc = st.number_input("封装热阻 RthJC (K/W)", value=0.065, format="%.4f")
        t_case = st.number_input("基板/水温 Tc (℃)", value=65.0)
    else:
        fixed_tj = st.number_input("设定全局目标结温 Tj (℃)", value=150.0)

# ==========================================
# 第一部分: 数据录入
# ==========================================
st.divider()
st.header("📊 第一步：特性数据录入")
col_t, col_d = st.columns(2)

with col_t:
    st.subheader("🔴 主开关管 (IGBT / SiC)")
    v_df_main = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'V_drop (V)': [1.1, 1.05, 2.2, 2.5]})
    ev_main = st.data_editor(v_df_main, num_rows="dynamic", key="v_main")
    
    e_df_main = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Eon (mJ)': [5.9, 8.5, 70.0, 95.0], 'Eoff (mJ)': [4.9, 7.2, 45.0, 60.0]})
    ee_main = st.data_editor(e_df_main, num_rows="dynamic", key="e_main")

with col_d:
    st.subheader("🔵 续流二极管 (FRD / Body Diode)")
    v_df_diode = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Vf (V)': [1.2, 1.1, 2.0, 2.2]})
    ev_diode = st.data_editor(v_df_diode, num_rows="dynamic", key="v_diode")
    
    e_df_diode = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Erec (mJ)': [1.9, 3.5, 15.0, 25.0]})
    ee_diode = st.data_editor(e_df_diode, num_rows="dynamic", key="e_diode")

# ==========================================
# 第二部分: 工况与全维物理系数
# ==========================================
st.divider()
st.header("⚙️ 第二步：运行工况与物理修正系数配置")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**⚡ 运行工况 (全场景)**")
    op_mode = st.selectbox("🏎️ 车辆运行状态", ["电动/巡航 (Motoring)", "制动/反拖 (Regeneration)", "最恶劣堵转 (Stall / 0Hz)"])
    vdc_act = st.number_input("实际母线 V_dc (V)", value=713.0)
    iout_rms = st.number_input("输出有效值 I_out (A)", value=264.5)
    fsw = st.number_input("开关频率 f_sw (Hz)", value=10000)
    fout = st.number_input("输出频率 f_out (Hz)", value=200.0)
    m_index = st.number_input("调制系数 M", value=0.90)
    cosphi = st.number_input("功率因数 cos_phi (绝对值)", value=0.90, min_value=0.0, max_value=1.0)
    mode = st.selectbox("调制模式", ["SVPWM", "SPWM"])

    if fout < 5.0 and "堵转" not in op_mode:
        st.warning(f"⚠️ 输出频率 {fout}Hz 过低，正弦波发热不均，建议切换至【最恶劣堵转】模式评估峰值结温。")

with c2:
    st.markdown("**📐 基准与电阻修正**")
    v_ref = st.number_input("规格书 V_nom (V)", value=600.0)
    t_ref_dp = st.number_input("规格书 T_ref_dp (℃)", value=150.0)
    rg_on_ref = st.number_input("手册 R_g,on (Ω)", value=2.5)
    rg_off_ref = st.number_input("手册 R_g,off (Ω)", value=20.0)
    rg_on_act = st.number_input("实际 R_on (Ω)", value=2.5)
    rg_off_act = st.number_input("实际 R_off (Ω)", value=20.0)

with c3:
    st.markdown("**📈 电压与电流指数**")
    kv_on = st.number_input("开通电压指数 K_v_on", value=1.30)
    kv_off = st.number_input("关断电压指数 K_v_off", value=1.30)
    kv_frd = st.number_input("续流电压指数 K_v_frd", value=0.60)
    ki_frd = st.number_input("续流电流指数 K_i_frd", value=0.60)
    r_arm_mohm = st.number_input("桥臂内阻 R_arm (mΩ)", value=0.0) 

with c4:
    st.markdown("**🌡️ 电阻与温度系数**")
    kron = st.number_input("开通电阻 K_ron", value=0.30)
    kroff = st.number_input("关断电阻 K_roff", value=0.50)
    t_coeff_igbt = st.number_input("主管温漂 T_c_igbt", value=0.003, format="%.4f")
    t_coeff_frd = st.number_input("续流温漂 T_c_rec", value=0.006 if "IGBT" in device_type else 0.003, format="%.4f")

# ==========================================
# 第三部分: 核心求解器
# ==========================================
st.divider()
st.header("🚀 第三步：执行整车工况联合仿真")

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

def calc_sw_analytical(df, i_pk, tj, item_name, fsw, vdc, vref, kv, ract, rref, kr, tc, tref):
    e_base = safe_interp(df, i_pk, tj, item_name)
    clean_df = df.dropna()
    active_tc = 0.0 if (not clean_df.empty and len(clean_df['Temp (℃)'].unique()) >= 2) else tc
    rcorr = math.pow(ract / rref, kr) if rref > 0 else 1.0
    tcorr = 1.0 + active_tc * (tj - tref)
    return e_base * rcorr * tcorr * math.pow(vdc / vref, kv)

if st.button("🚀 执 行 全 工 况 仿 真 计 算", use_container_width=True):
    # 底层归一化
    norm_ev_main = normalize_vi_df(ev_main, n_src_cond)
    norm_ev_diode = normalize_vi_df(ev_diode, n_src_cond)
    norm_ee_main = normalize_ei_df(ee_main, n_src_sw, ['Eon (mJ)', 'Eoff (mJ)'])
    norm_ee_diode = normalize_ei_df(ee_diode, n_src_sw, ['Erec (mJ)'])

    # 工况环境判断
    i_pk_chip = math.sqrt(2) * (iout_rms / n_sim)
    r_arm_chip = (r_arm_mohm / 1000.0) * n_sim
    
    # 核心：反拖模式自动翻转功率因数
    active_cosphi = -cosphi if "反拖" in op_mode else cosphi
    theta = math.acos(active_cosphi) if abs(active_cosphi) <= 1.0 else math.acos(1.0)
    
    loop_count = 1 if "开环" in sim_mode else 15
    tj_current = fixed_tj if "开环" in sim_mode else t_case + 5.0

    for _ in range(loop_count):
        # 1. 提取动态特性
        v_pk_m = safe_interp(norm_ev_main, i_pk_chip, tj_current, 'V_drop (V)')
        v_hf_m = safe_interp(norm_ev_main, i_pk_chip/2, tj_current, 'V_drop (V)')
        r_m_chip = (v_pk_m - v_hf_m) / (i_pk_chip / 2) if i_pk_chip > 0 else 0
        v0_m = 0.0 if "SiC" in device_type else max(0.0, v_pk_m - r_m_chip * i_pk_chip)
        rm_tot = r_m_chip + r_arm_chip

        v_pk_d = safe_interp(norm_ev_diode, i_pk_chip, tj_current, 'Vf (V)')
        v_hf_d = safe_interp(norm_ev_diode, i_pk_chip/2, tj_current, 'Vf (V)')
        r_d_chip = (v_pk_d - v_hf_d) / (i_pk_chip / 2) if i_pk_chip > 0 else 0
        v0_d = max(0.0, v_pk_d - r_d_chip * i_pk_chip)
        rd_tot = r_d_chip + r_arm_chip

        # 2. 核心工况分流计算
        if "堵转" in op_mode:
            # 极限堵转模式：无 1/π 正弦平均，连续承受峰值电流发热
            d_max = 0.5 * (1 + m_index)  # 峰值占空比
            d_min = 1.0 - d_max
            
            p_cond_main_chip = d_max * (v0_m * i_pk_chip + rm_tot * i_pk_chip**2)
            p_cond_diode_chip = d_min * (v0_d * i_pk_chip + rd_tot * i_pk_chip**2)
            
            # 开关能量在堵转下不除以 π，直接乘以频率
            e_on_adj = calc_sw_analytical(norm_ee_main, i_pk_chip, tj_current, 'Eon (mJ)', 1, vdc_act, v_ref, kv_on, rg_on_act, rg_on_ref, kron, t_coeff_igbt, t_ref_dp)
            e_off_adj = calc_sw_analytical(norm_ee_main, i_pk_chip, tj_current, 'Eoff (mJ)', 1, vdc_act, v_ref, kv_off, rg_off_act, rg_off_ref, kroff, t_coeff_igbt, t_ref_dp)
            p_sw_main_chip = fsw * (e_on_adj + e_off_adj) / 1000.0
            
            e_rec_adj = calc_sw_analytical(norm_ee_diode, i_pk_chip, tj_current, 'Erec (mJ)', 1, vdc_act, v_ref, kv_frd, 1.0, 1.0, 0.0, t_coeff_frd, t_ref_dp)
            p_sw_diode_chip = fsw * (e_rec_adj / 1000.0)

        else:
            # 电动与反拖模式：标准解析积分
            if mode == "SVPWM":
                kv0_m = (m_index * active_cosphi) / 4.0
                kr_m = (24*active_cosphi - 2*math.sqrt(3)*math.cos(2*theta) - 3*math.sqrt(3)) / 24.0
                p_cond_main_chip = (kv0_m * v0_m * i_pk_chip) + (kr_m * rm_tot * i_pk_chip**2) / math.pi
                
                kv0_d = (4 - m_index * math.pi * active_cosphi) / 4.0
                kr_d = (6*math.pi - 24*m_index*active_cosphi + 2*math.sqrt(3)*m_index*math.cos(2*theta) + 3*math.sqrt(3)*m_index) / 24.0
                p_cond_diode_chip = (kv0_d * v0_d * i_pk_chip) / math.pi + (kr_d * rd_tot * i_pk_chip**2) / math.pi
            else: # SPWM
                p_cond_main_chip = v0_m*i_pk_chip*(1/(2*math.pi) + m_index*active_cosphi/8) + rm_tot*i_pk_chip**2*(1/8 + m_index*active_cosphi/(3*math.pi))
                p_cond_diode_chip = v0_d*i_pk_chip*(1/(2*math.pi) - m_index*active_cosphi/8) + rd_tot*i_pk_chip**2*(1/8 - m_index*active_cosphi/(3*math.pi))

            # 正弦平均开关损耗 (带有 1/π)
            e_on_adj = calc_sw_analytical(norm_ee_main, i_pk_chip, tj_current, 'Eon (mJ)', 1, vdc_act, v_ref, kv_on, rg_on_act, rg_on_ref, kron, t_coeff_igbt, t_ref_dp)
            e_off_adj = calc_sw_analytical(norm_ee_main, i_pk_chip, tj_current, 'Eoff (mJ)', 1, vdc_act, v_ref, kv_off, rg_off_act, rg_off_ref, kroff, t_coeff_igbt, t_ref_dp)
            p_sw_main_chip = (1.0 / math.pi) * fsw * (e_on_adj + e_off_adj) / 1000.0
            
            e_rec_adj = calc_sw_analytical(norm_ee_diode, i_pk_chip, tj_current, 'Erec (mJ)', 1, vdc_act, v_ref, kv_frd, 1.0, 1.0, 0.0, t_coeff_frd, t_ref_dp)
            i_corr = (iout_rms / ((i_pk_chip*n_sim) / math.sqrt(2))) ** ki_frd if ki_frd > 0 else 1.0
            p_sw_diode_chip = (1.0 / math.pi) * fsw * (e_rec_adj / 1000.0) * i_corr

        # 3. 结果扩容放大
        p_total_main = (p_cond_main_chip + p_sw_main_chip) * n_sim
        p_total_diode = (p_cond_diode_chip + p_sw_diode_chip) * n_sim
        p_total = p_total_main + p_total_diode
        
        if "闭环" in sim_mode:
            tj_new = t_case + p_total * rth_jc
            if abs(tj_new - tj_current) < 0.05: break
            tj_current = tj_new

    # ==========================================
    # 结果展示面板
    # ==========================================
    st.success(f"✅ 整车工况执行完毕！当前状态：**{op_mode}** (架构: {device_type} | 扩容倍率: {n_sim})")
    
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("仿真极限结温 Tj", f"{tj_current:.1f} ℃")
    r2.metric("整臂发热总功率", f"{p_total:.1f} W")
    r3.metric("🔴 主开关管总损耗", f"{p_total_main:.1f} W")
    r4.metric("🔵 续流二极管总损耗", f"{p_total_diode:.1f} W")

    st.divider()
    st.markdown("#### 📉 CAE 热仿真单芯发热率 (直接填入热模型)")
    col_m, col_d = st.columns(2)
    with col_m:
        st.info(f"**🔴 IGBT / SiC 芯片单颗发热 (占比 {p_total_main/p_total*100:.1f}%)**" if p_total>0 else "**🔴 主芯片单颗发热**")
        st.write(f"- 导通分项: {p_cond_main_chip:.1f} W | 开关分项: {p_sw_main_chip:.1f} W")
        st.markdown(f"> ### **{p_cond_main_chip + p_sw_main_chip:.2f} W / 颗**")
    with col_d:
        st.info(f"**🔵 FRD / 续流二极管单颗发热 (占比 {p_total_diode/p_total*100:.1f}%)**" if p_total>0 else "**🔵 二极管单颗发热**")
        st.write(f"- 导通分项: {p_cond_diode_chip:.1f} W | 恢复分项: {p_sw_diode_chip:.1f} W")
        st.markdown(f"> ### **{p_cond_diode_chip + p_sw_diode_chip:.2f} W / 颗**")

    # 底层透传
    st.divider()
    with st.expander("🔬 查看底层工况物理引擎逻辑 (点击展开验证)"):
        st.markdown("#### 1. 整车工况映射逻辑")
        if "堵转" in op_mode:
            st.error("🚨 **已启用【极限堵转模型】**：此时输出频率相当于 0Hz，芯片无法均摊正弦波发热，直接承受持续的峰值电流冲击！")
            st.latex(r"P_{sw\_stall} = f_{sw} \cdot E(I_{pk}) \quad \text{(已舍弃 1/}\pi\text{ 均值积分)}")
        elif "反拖" in op_mode:
            st.success("🔄 **已启用【反拖/制动模型】**：内部已将 $\cos\phi$ 倒转，系统进入发电状态，热负荷将瞬间由 IGBT 转移至 FRD 二极管！")
        else:
            st.info("🚗 **已启用【常规电动模型】**：标准巡航或加速态，主发热源为 IGBT/SiC。")
        
        st.markdown("#### 2. 系统级归一化扩容")
        st.write(f"当前仿真设定为 **{n_sim}** 并联架构，所有热源数据已自动进行缩放换算。")
