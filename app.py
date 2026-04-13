import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD模块电热耦合-全维度版", layout="wide")
st.title("🔬 功率模块电热耦合仿真 - 多方案灵活对标平台")

# --- 1. 方案选择 ---
st.sidebar.header("配置方案选择")
scheme = st.sidebar.radio("选择计算方案", ["芯片级方案 (Chip-level)", "模块级方案 (Module-level)"])
n_chips = st.sidebar.number_input("每臂芯片并联数 (N)", value=6 if scheme == "芯片级方案 (Chip-level)" else 1, min_value=1)

# --- 2. 纵向累计数据录入 ---
st.header("1. 特性数据录入 (纵向累计排列)")
col_data1, col_data2 = st.columns(2)

with col_data1:
    st.subheader("📊 静态压降数据 (Vce/Vf)")
    v_df = pd.DataFrame({
        'Temp (℃)': [25, 25, 125, 125, 150, 150, 175, 175],
        'Current (A)': [100, 600, 100, 600, 100, 600, 100, 600],
        'V_drop (V)': [1.1, 2.2, 1.05, 2.4, 1.0, 2.5, 0.95, 2.65]
    })
    ev_df = st.data_editor(v_df, num_rows="dynamic", key="v_table")

with col_data2:
    st.subheader("⚡ 开关能量数据 (Eon+Eoff+Erec)")
    e_df = pd.DataFrame({
        'Temp (℃)': [25, 25, 150, 150, 175, 175],
        'Current (A)': [100, 600, 100, 600, 100, 600],
        'Energy (mJ)': [5.0, 45.0, 8.5, 75.0, 10.0, 90.0]
    })
    ee_df = st.data_editor(e_df, num_rows="dynamic", key="e_table")

# --- 3. 结构与工况参数 ---
st.divider()
st.header("2. 结构参数与工况设置")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**结构参数**")
    r_arm = st.number_input("桥臂内阻 (mΩ)", value=0.5) / 1000
    rth_jc = st.number_input("RthJC (K/W)", value=0.065, format="%.4f")
    t_case = st.number_input("基板温度 Tc (℃)", value=65.0)

with c2:
    st.markdown("**运行工况**")
    vdc = st.number_input("直流电压 Vdc (V)", value=780.0)
    v_ref = st.number_input("测试基准 Vref (V)", value=510.0)
    iout = st.number_input("输出有效值 Iout (Arms)", value=187.0)

with c3:
    st.markdown("**控制参数**")
    fsw = st.number_input("开关频率 fsw (Hz)", value=10000)
    m_index = st.number_input("调制度 M", value=0.92)
    cosphi = st.number_input("功率因数 cosφ", value=0.92)

with c4:
    st.markdown("**修正指数**")
    kv_exp = st.number_input("电压修正指数 Kv", value=1.3)
    rg_corr = st.number_input("门极电阻修正系数", value=1.0)
    mode = st.selectbox("调制模式", ["SVPWM", "PWM"])

# --- 4. 核心计算引擎 (多级插值与迭代) ---
def advanced_interp(df, target_i, target_t):
    """处理纵向累计数据的二级插值"""
    clean_df = df.dropna()
    groups = clean_df.groupby('Temp (℃)')
    
    # 存储每个温度点下的电流维度插值结果
    temp_list = []
    val_list = []
    
    for temp, group in groups:
        sorted_group = group.sort_values('Current (A)')
        if len(sorted_group) >= 2:
            f = interp1d(sorted_group['Current (A)'], sorted_group.iloc[:, 2], 
                         kind='linear', fill_value="extrapolate")
            val_list.append(float(f(target_i)))
            temp_list.append(temp)
    
    if len(temp_list) >= 2:
        f_t = interp1d(temp_list, val_list, kind='linear', fill_value="extrapolate")
        return float(f_t(target_t))
    elif len(temp_list) == 1:
        return val_list[0]
    return 0.0

if st.button("🚀 开始电热闭环仿真"):
    tj_loop = t_case + 5.0
    
    # 根据方案确定查找电流
    i_lookup = iout / n_chips if scheme == "芯片级方案 (Chip-level)" else iout
    
    for _ in range(12):
        # 1. 导通损耗
        v_drop_base = advanced_interp(ev_df, i_lookup, tj_loop)
        # 总压降 = 查表值 + 桥臂电阻压降
        v_total = v_drop_base + iout * r_arm
        
        # 调制系数计算
        phi = math.acos(cosphi)
        if mode == "SVPWM":
            k_v0, k_r = 0.25*m_index*cosphi, (24*cosphi - 2*math.sqrt(3)*math.cos(2*phi) - 3*math.sqrt(3))/24
        else:
            k_v0, k_r = (1/(2*math.pi)) + (m_index*cosphi/8), (1/8) + (m_index*cosphi/(3*math.pi))
        
        # 导通损耗 (如果是芯片级，需要乘以芯片数)
        # 注意：v_total 是对应的单臂或单芯压降，iout 是总电流
        p_cond = v_total * iout * (k_v0 * 4 + k_r * 2) / 2

        # 2. 开关损耗
        esw_base = advanced_interp(ee_df, i_lookup, tj_loop)
        v_corr = math.pow(vdc / v_ref, kv_exp)
        # 总开关损耗 (如果是芯片级方案，esw_base 是单颗芯片能量，总能量需乘以 N)
        p_sw_total = (1/math.pi) * fsw * (esw_base * (n_chips if scheme == "芯片级方案 (Chip-level)" else 1) / 1000) * v_corr * rg_corr
        
        p_total = p_sw_total + p_cond
        tj_new = t_case + p_total * rth_jc
        
        if abs(tj_new - tj_loop) < 0.05: break
        tj_loop = tj_new

    # --- 结果展示 ---
    st.divider()
    res1, res2, res3, res4 = st.columns(4)
    res1.metric("稳定结温 Tj", f"{tj_loop:.2f} ℃")
    res2.metric("总损耗 P_total", f"{p_total:.2f} W")
    res3.metric("导通损耗 P_cond", f"{p_cond:.2f} W")
    res4.metric("开关损耗 P_sw", f"{p_sw_total:.2f} W")
    
    st.success(f"方案：{scheme} | 查找电流：{i_lookup:.2f}A")
