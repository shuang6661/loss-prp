import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD纯芯片级损耗对标", layout="wide")
st.title("🛡️ 功率模块纯芯片级损耗对标平台")

st.sidebar.header("配置方案")
scheme = st.sidebar.radio("选择计算方案", ["芯片级方案 (Chip-level)", "模块级方案 (Module-level)"])
n_chips = st.sidebar.number_input("并联芯片数 (N)", value=6 if scheme == "芯片级方案 (Chip-level)" else 1, min_value=1)

# --- 1. 原始特性矩阵录入 ---
st.header("1. 原始特性矩阵录入 (V-I & E-I 数据)")
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.subheader("📉 通态特性矩阵 (V_drop)")
    v_df = pd.DataFrame({
        'Temp (℃)': [25, 25, 150, 150],
        'Current (A)': [100.0, 600.0, 100.0, 600.0],
        'V_drop_T (V)': [1.10, 2.20, 1.05, 2.50], # IGBT压降
        'V_drop_D (V)': [1.20, 2.00, 1.10, 2.20]  # Diode压降
    })
    ev_df = st.data_editor(v_df, num_rows="dynamic", key="v_table_pure")

with col_d2:
    st.subheader("⚡ 开关特性矩阵 (E_sw)")
    e_df = pd.DataFrame({
        'Temp (℃)': [25, 25, 150, 150],
        'Current (A)': [100.0, 600.0, 100.0, 600.0],
        'Eon (mJ)': [5.9, 70.0, 8.5, 95.0],
        'Eoff (mJ)': [4.9, 45.0, 7.2, 60.0],
        'Erec (mJ)': [1.9, 15.0, 3.5, 25.0]
    })
    ee_df = st.data_editor(e_df, num_rows="dynamic", key="e_table_pure")

# --- 2. 物理参数与工况设置 ---
st.divider()
st.header("2. 工况设置与修正指数")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**电气工况**")
    vdc_act = st.number_input("实际母线 Vdc (V)", value=780.0)
    iout_rms = st.number_input("输出有效值 Iout (Arms)", value=187.0)
    fsw = st.number_input("开关频率 fsw (Hz)", value=10000)
    fout = st.number_input("输出基波频率 fout (Hz)", value=50.0)

with c2:
    st.markdown("**调制参数**")
    m_index = st.number_input("调制度 M", value=0.92)
    cosphi = st.number_input("功率因数 cosφ", value=0.92)
    mode = st.selectbox("调制模式", ["SVPWM", "SPWM/PWM"])

with c3:
    st.markdown("**门极电阻与测试基准**")
    v_ref = st.number_input("测试基准电压 Vref (V)", value=510.0)
    rg_ref = st.number_input("规格书测试 Rg_ref (Ω)", value=5.0, help="厂家测双脉冲时用的电阻")
    rg_on_act = st.number_input("实际开通 Rg_on (Ω)", value=5.0)
    rg_off_act = st.number_input("实际关断 Rg_off (Ω)", value=25.0)

with c4:
    st.markdown("**热阻与修正系数**")
    rth_jc = st.number_input("热阻 RthJC (K/W)", value=0.065, format="%.4f")
    t_case = st.number_input("基板温度 Tc (℃)", value=65.0)
    kv_exp = st.number_input("电压修正指数 Kv", value=1.3)
    kr_exp = st.number_input("电阻修正指数 Kr", value=1.0)
    # --- 3. 核心计算引擎 (严防负值 + 公式对标) ---
def advanced_interp(df, target_i, target_t, item_name):
    clean_df = df.dropna()
    temp_list, val_list = [], []
    for temp, group in clean_df.groupby('Temp (℃)'):
        sorted_g = group.sort_values('Current (A)')
        if len(sorted_g) >= 2:
            f = interp1d(sorted_g['Current (A)'], sorted_g[item_name], kind='linear', fill_value="extrapolate")
            val = max(0.0, float(f(target_i))) # 物理锁1：杜绝外推负值
            val_list.append(val)
            temp_list.append(temp)
    if len(temp_list) >= 2:
        return max(0.0, float(interp1d(temp_list, val_list, fill_value="extrapolate")(target_t))) # 物理锁2
    elif len(temp_list) == 1: 
        return max(0.0, val_list[0])
    return 0.0

if st.button("🚀 执行纯芯片级计算"):
    tj_loop = t_case + 5.0
    i_lookup = iout_rms / n_chips if scheme == "芯片级方案 (Chip-level)" else iout_rms
    i_pk = math.sqrt(2) * i_lookup # 计算查表用的单芯/单臂峰值电流
    theta = math.acos(cosphi)
    
    for _ in range(12):
        # A. 动态提取 V0 和 r (完美对接你的公式)
        # 通过在 I_pk 和 I_pk/2 处插值，反推 V0 和 r_on
        v_t_pk = advanced_interp(ev_df, i_pk, tj_loop, 'V_drop_T (V)')
        v_t_half = advanced_interp(ev_df, i_pk/2, tj_loop, 'V_drop_T (V)')
        r_t = (v_t_pk - v_t_half) / (i_pk / 2) if i_pk > 0 else 0
        v0_t = v_t_pk - r_t * i_pk

        v_d_pk = advanced_interp(ev_df, i_pk, tj_loop, 'V_drop_D (V)')
        v_d_half = advanced_interp(ev_df, i_pk/2, tj_loop, 'V_drop_D (V)')
        r_d = (v_d_pk - v_d_half) / (i_pk / 2) if i_pk > 0 else 0
        v0_d = v_d_pk - r_d * i_pk

        # B. 调制系数解析 (严格遵守你的源Sheet公式)
        if mode == "SVPWM":
            kv0_t, kr_t = m_index*cosphi/4, (24*cosphi - 2*math.sqrt(3)*math.cos(2*theta) - 3*math.sqrt(3))/24
            # 二极管 SVPWM 复杂公式
            kv0_d = (4 - m_index*math.pi*cosphi)/(4*math.pi)
            kr_d = (6*math.pi - 24*m_index*cosphi + 2*math.sqrt(3)*m_index*math.cos(2*theta) + 3*math.sqrt(3)*m_index)/(24*math.pi)
        else:
            kv0_t, kr_t = 1/(2*math.pi) + m_index*cosphi/8, 1/8 + m_index*cosphi/(3*math.pi)
            kv0_d, kr_d = 1/(2*math.pi) - m_index*cosphi/8, 1/8 - m_index*cosphi/(3*math.pi)
        
        # 导通损耗 (放大回总电流维度)
        mult = n_chips if scheme == "芯片级方案 (Chip-level)" else 1
        p_cond_t = (v0_t * i_pk * kv0_t + r_t * i_pk**2 * kr_t) * mult
        p_cond_d = (v0_d * i_pk * kv0_d + r_d * i_pk**2 * kr_d) * mult
        p_cond = p_cond_t + p_cond_d

        # C. 开关损耗 (查表 + 修正)
        e_on = advanced_interp(ee_df, i_lookup, tj_loop, 'Eon (mJ)')
        e_off = advanced_interp(ee_df, i_lookup, tj_loop, 'Eoff (mJ)')
        e_rec = advanced_interp(ee_df, i_lookup, tj_loop, 'Erec (mJ)')
        
        p_on = (1/math.pi) * fsw * (e_on * mult / 1000) * (vdc_act/v_ref)**kv_exp * (rg_on_act/rg_ref)**kr_exp
        p_off = (1/math.pi) * fsw * (e_off * mult / 1000) * (vdc_act/v_ref)**kv_exp * (rg_off_act/rg_ref)**kr_exp
        p_rec = (1/math.pi) * fsw * (e_rec * mult / 1000) * (vdc_act/v_ref)**kv_exp * (rg_on_act/rg_ref)**kr_exp
        p_sw = p_on + p_off + p_rec

        # D. 结温闭环
        p_total = p_sw + p_cond
        tj_new = t_case + p_total * rth_jc
        if abs(tj_new - tj_loop) < 0.05: break
        tj_loop = tj_new
            # --- 结果展示 ---
    st.divider()
    st.subheader("3. 对标结果分析")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("稳定结温 Tj", f"{tj_loop:.2f} ℃")
    r2.metric("总导通损耗 P_cond", f"{p_cond:.2f} W")
    r3.metric("总开关损耗 P_sw", f"{p_sw:.2f} W")
    r4.metric("模块总损耗", f"{p_total:.2f} W")

    with st.expander("🔬 查看提取的芯片参数与应用的解析公式"):
        st.write(f"当前从 V-I 矩阵自动反推的硅片参数 (Tj={tj_loop:.1f}℃):")
        st.write(f"IGBT: Vce0 = {v0_t:.3f} V, r_ce = {r_t*1000:.2f} mΩ")
        st.write(f"Diode: Vf0 = {v0_d:.3f} V, r_f = {r_d*1000:.2f} mΩ")
        st.latex(r"P_{IGBT\_cond} = \frac{1}{\pi} V_{CE0} I_{pk} K_{v0\_T} + r_{CE} I_{pk}^2 K_{r\_T}")
