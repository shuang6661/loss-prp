import streamlit as st
import numpy as np
import math

# 设置页面
st.set_page_config(page_title="BYD仿真数字化平台-离线版", layout="wide")
st.title("🔋 功率损耗仿真数字化平台 (Pro版)")

# --- 侧边栏：器件数据库（二次插值拟合区） ---
st.sidebar.header("1. 器件规格书数据 (后台)")
device_type = st.sidebar.selectbox("选择器件类型", ["SiC MOSFET", "IGBT"])
n_chips = st.sidebar.number_input("半桥并联芯片数量", value=6, min_value=1)

with st.sidebar.expander("二次插值：导通特性 (I vs V)"):
    st.write("输入三个测试点 [I, V]")
    i_pts = np.array([st.number_input("I1", value=100.0), st.number_input("I2", value=450.0), st.number_input("I3", value=900.0)])
    v_pts = np.array([st.number_input("V1", value=1.1), st.number_input("V2", value=1.8), st.number_input("V3", value=2.9)])
    v_coeffs = np.polyfit(i_pts, v_pts, 2) # 计算 a, b, c

with st.sidebar.expander("二次插值：开关能量 (I vs Esw)"):
    st.write("输入三个测试点 [I, E_total(mJ)]")
    ei_pts = np.array([100.0, 450.0, 900.0])
    e_pts = np.array([st.number_input("E1", value=10.5), st.number_input("E2", value=45.1), st.number_input("E3", value=95.0)])
    e_coeffs = np.polyfit(ei_pts, e_pts, 2) # 计算 A, B, C

# --- 主界面：工况输入 ---
st.header("2. 实时工况输入")
col1, col2, col3 = st.columns(3)
with col1:
    vdc = st.number_input("母线电压 Vdc (V)", value=780.0)
    iout = st.number_input("电流有效值 Iout (Arms)", value=187.0)
    fsw = st.number_input("开关频率 fsw (Hz)", value=10000)
with col2:
    mode = st.selectbox("调制模式", ["SVPWM", "PWM"])
    m_index = st.number_input("调制系数 M", value=0.92)
    cosphi = st.number_input("功率因数 cosφ", value=0.92)
with col3:
    tj = st.number_input("当前结温 Tj (℃)", value=175.0)
    v_ref = st.number_input("测试电压基准 Vref (V)", value=650.0)
    # 针对不同器件的电流修正系数
    ki_sw = 0.6 if device_type == "SiC MOSFET" else 1.0 

# --- 核心计算引擎 ---
if st.button("🚀 执行全功能计算"):
    # 1. 导通损耗：二次插值 + 多芯折算 + 调制积分
    # 折算系数至 N 芯
    a = v_coeffs[0] / (n_chips**2)
    b = v_coeffs[1] / n_chips
    c = v_coeffs[2]
    
    phi = math.acos(cosphi)
    i_pk = math.sqrt(2) * iout
    
    if mode == "SVPWM":
        k1 = 0.25 * m_index * cosphi
        k2 = (24*cosphi - 2*math.sqrt(3)*math.cos(2*phi) - 3*math.sqrt(3))/24
        k3 = (m_index * cosphi * (3 + math.cos(2*phi))) / (12 * math.pi)
    else:
        k1 = (1/(2*math.pi)) + (m_index * cosphi / 8)
        k2 = (1/8) + (m_index * cosphi / (3 * math.pi))
        k3 = 2 / (3 * math.pi)

    # 导通损耗解析积分
    p_cond = a * k3 * (i_pk**3) + b * k2 * (i_pk**2) + c * k1 * i_pk

    # 2. 开关损耗：二次拟合 + 电压/结温/电流三重修正
    e_base = e_coeffs[0] * (iout**2) + e_coeffs[1] * iout + e_coeffs[2]
    kt_sw = 1 + 0.003 * (tj - 25) # 温度补偿
    kv_sw = math.pow(vdc / v_ref, 1.3) # 电压修正
    
    # 结合有效值到峰值的积分转换 (1/pi)
    p_sw = (1 / math.pi) * fsw * (e_base / 1000) * kt_sw * kv_sw

    # --- 结果展示区 ---
    st.divider()
    st.subheader("3. 仿真结果报告")
    res1, res2, res3 = st.columns(3)
    res1.metric("导通损耗 (P_cond)", f"{p_cond:.2f} W")
    res2.metric("开关损耗 (P_sw)", f"{p_sw:.2f} W")
    res3.metric("模块总损耗", f"{p_cond + p_sw:.2f} W")

    # 4 芯/6 芯对比分析提示
    st.info(f"分析：当前工况下单颗芯片分担电流为 {iout/n_chips:.2f} Arms。如果仿真结温偏低，请检查 Tj={tj}℃ 时的二次项系数是否偏移。")