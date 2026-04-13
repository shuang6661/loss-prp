import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD仿真数字化平台-终极对标版", layout="wide")
st.title("🛡️ 功率模块全维度损耗与结温仿真平台 (终极整合版)")

# --- 1. 方案选择与基本配置 ---
st.sidebar.header("配置与方案 (Scheme)")
scheme = st.sidebar.radio("选择计算方案", ["芯片级方案 (Chip-level)", "模块级方案 (Module-level)"])
n_chips = st.sidebar.number_input("并联芯片数 (N)", value=6 if scheme == "芯片级方案 (Chip-level)" else 1, min_value=1)
st.sidebar.divider()
st.sidebar.markdown("**对标提醒**：芯片级方案会根据 N 自动分摊电流。模块级方案直接使用总电流查表。")

# --- 2. 原始特性矩阵录入 (通态特性 + 开关特性) ---
st.header("1. 原始特性矩阵录入 (Datasheet Inputs)")
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.subheader("📉 通态特性 (Vce/Vf 矩阵) [V]")
    st.write("支持纵向累计录入，程序会自动进行多温插值。")
    # 默认数据：包含不同温度、不同电流的采样点
    v_df = pd.DataFrame({
        'Temp (℃)': [25, 25, 150, 150, 175, 175],
        'Current (A)': [100.0, 600.0, 100.0, 600.0, 100.0, 600.0],
        'V_drop (V)': [1.10, 2.20, 1.05, 2.50, 1.00, 2.65]
    })
    ev_df = st.data_editor(v_df, num_rows="dynamic", key="v_table_final")

with col_d2:
    st.subheader("⚡ 开关特性 (Eon, Eoff, Erec 矩阵) [mJ]")
    st.write("对应你规格书中的双脉冲实测数据。")
    e_df = pd.DataFrame({
        'Temp (℃)': [25, 25, 150, 150, 175, 175],
        'Current (A)': [100.0, 800.0, 100.0, 800.0, 100.0, 800.0],
        'Eon (mJ)': [5.9, 121.0, 8.5, 160.0, 10.0, 185.0],
        'Eoff (mJ)': [4.9, 65.5, 7.2, 85.0, 8.5, 95.0],
        'Erec (mJ)': [1.9, 4.6, 3.5, 8.0, 4.5, 10.0]
    })
    ee_df = st.data_editor(e_df, num_rows="dynamic", key="e_table_final")

# --- 3. 物理修正系数与工况设置 ---
st.divider()
st.header("2. 物理修正系数与工况设置")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**母线与驱动 (Reference)**")
    vdc_act = st.number_input("实际 Vdc (V)", value=780.0)
    v_ref = st.number_input("测试基准 Vref (V)", value=510.0)
    rg_ref = st.number_input("测试基准 Rg_ref (Ω)", value=5.0)
    rg_on_act = st.number_input("实际开通 Rg_on (Ω)", value=5.0)
    rg_off_act = st.number_input("实际关断 Rg_off (Ω)", value=25.0)

with c2:
    st.markdown("**运行工况 (Operation)**")
    iout_rms = st.number_input("输出有效值 Iout (Arms)", value=187.0)
    fsw = st.number_input("开关频率 fsw (Hz)", value=10000)
    m_index = st.number_input("调制度 M", value=0.92)
    cosphi = st.number_input("功率因数 cosφ", value=0.92)

with c3:
    st.markdown("**热学与内阻 (Thermal)**")
    rth_jc = st.number_input("热阻 RthJC (K/W)", value=0.065, format="%.4f")
    t_case = st.number_input("基板温度 Tc (℃)", value=65.0)
    r_arm = st.number_input("桥臂/母排内阻 (mΩ)", value=0.5) / 1000.0

with c4:
    st.markdown("**修正指数 (Exponents)**")
    kv_on = st.number_input("Kv_on (Eon电压指数)", value=1.3)
    kv_off = st.number_input("Kv_off (Eoff电压指数)", value=1.2)
    kv_rec = st.number_input("Kv_rec (Erec电压指数)", value=0.6)
    mode = st.selectbox("调制模式", ["SVPWM", "SPWM/PWM"])

# --- 4. 核心计算引擎 (多维插值 + 闭环迭代) ---
def advanced_interp(df, target_i, target_t, item_name):
    clean_df = df.dropna()
    temp_list, val_list = [], []
    for temp, group in clean_df.groupby('Temp (℃)'):
        sorted_g = group.sort_values('Current (A)')
        if len(sorted_g) >= 2:
            f = interp1d(sorted_g['Current (A)'], sorted_g[item_name], kind='linear', fill_value="extrapolate")
            val_list.append(float(f(target_i)))
            temp_list.append(temp)
    if len(temp_list) >= 2:
        return float(interp1d(temp_list, val_list, fill_value="extrapolate")(target_t))
    elif len(temp_list) == 1: return val_list[0]
    return 0.0

if st.button("🚀 执行全参数电热闭环仿真"):
    tj_loop = t_case + 5.0
    i_lookup = iout_rms / n_chips if scheme == "芯片级方案 (Chip-level)" else iout_rms
    i_pk = math.sqrt(2) * iout_rms
    theta = math.acos(cosphi)
    
    # 迭代计算
    for _ in range(12):
        # A. 导通损耗
        v_drop = advanced_interp(ev_df, i_lookup, tj_loop, 'V_drop (V)')
        v_total = v_drop + iout_rms * r_arm
        
        if mode == "SVPWM":
            kv0, kr = m_index*cosphi/4, (24*cosphi - 2*math.sqrt(3)*math.cos(2*theta) - 3*math.sqrt(3))/24
        else:
            kv0, kr = 1/(2*math.pi) + m_index*cosphi/8, 1/8 + m_index*cosphi/(3*math.pi)
        
        p_cond = v_total * iout_rms * (kv0 * 4 + kr * 2) / 2

        # B. 开关损耗 (分项修正)
        e_on = advanced_interp(ee_df, i_lookup, tj_loop, 'Eon (mJ)')
        e_off = advanced_interp(ee_df, i_lookup, tj_loop, 'Eoff (mJ)')
        e_rec = advanced_interp(ee_df, i_lookup, tj_loop, 'Erec (mJ)')
        
        mult = n_chips if scheme == "芯片级方案 (Chip-level)" else 1
        p_on = (1/math.pi) * fsw * (e_on * mult / 1000) * (vdc_act/v_ref)**kv_on * (rg_on_act/rg_ref)**1.0
        p_off = (1/math.pi) * fsw * (e_off * mult / 1000) * (vdc_act/v_ref)**kv_off * (rg_off_act/rg_ref)**0.8
        p_rec = (1/math.pi) * fsw * (e_rec * mult / 1000) * (vdc_act/v_ref)**kv_rec * (rg_on_act/rg_ref)**0.5
        
        p_total = p_on + p_off + p_rec + p_cond
        tj_new = t_case + p_total * rth_jc
        
        if abs(tj_new - tj_loop) < 0.05: break
        tj_loop = tj_
