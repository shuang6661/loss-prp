import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD全架构仿真-并联控制版", layout="wide")
st.title("🛡️ 功率模块多架构电热联合仿真平台 (精确并联控制版)")

# --- 侧边栏：全局架构与工作流 ---
with st.sidebar:
    st.header("⚙️ 核心架构设置")
    device_type = st.radio("1. 模块芯片技术架构", ["IGBT + FRD (传统硅基)", "SiC MOSFET (碳化硅)"])
    
    st.divider()
    st.header("🧮 数据录入方案")
    scheme = st.radio("2. 查表数据层级", ["芯片级方案 (Chip-level)", "模块级方案 (Module-level)"])
    n_chips = st.number_input("并联芯片数 (N)", value=6 if "芯片" in scheme else 1, min_value=1)
    st.info("注：芯片级方案会自动将总电流除以 N 进行单芯查表，最后总损耗乘以 N。")

    st.divider()
    st.header("🔄 仿真工作流选择")
    sim_mode = st.radio("3. 热学计算模式", [
        "A. 开环盲算 (已知目标结温，求损耗导出给热仿真)", 
        "B. 闭环迭代 (已有封装热阻，求最终稳态结温)"
    ])
    
    if "闭环" in sim_mode:
        rth_jc = st.number_input("输入封装热阻 RthJC (K/W)", value=0.065, format="%.4f")
        t_case = st.number_input("基板/水温 Tc (℃)", value=65.0)
    else:
        fixed_tj = st.number_input("设定全局目标结温 Tj (℃)", value=150.0)

# --- 页面主体 ---
tab1, tab2, tab3 = st.tabs(["📊 第一步：特性数据录入", "⚙️ 第二步：工况与高级系数", "🚀 第三步：执行联合仿真"])

# ==========================================
# TAB 1: 数据录入
# ==========================================
with tab1:
    st.info("💡 **快捷录入**：在 Excel 选中数据 (Ctrl+C)，点击下方表格左上角空单元格粘贴 (Ctrl+V) 即可自动扩充。")
    col_t, col_d = st.columns(2)
    
    with col_t:
        st.subheader("🔴 主管特性 (IGBT / SiC)")
        v_df_main = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'V_drop (V)': [1.1, 1.05, 2.2, 2.5]})
        st.write("1. 通态压降 (Vce / Vds)")
        ev_main = st.data_editor(v_df_main, num_rows="dynamic", key="v_main")
        
        e_df_main = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Eon (mJ)': [5.9, 8.5, 70.0, 95.0], 'Eoff (mJ)': [4.9, 7.2, 45.0, 60.0]})
        st.write("2. 开关能量 (Eon, Eoff)")
        ee_main = st.data_editor(e_df_main, num_rows="dynamic", key="e_main")

    with col_d:
        if "IGBT" in device_type:
            st.subheader("🔵 FRD 二极管特性")
            v_df_diode = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Vf (V)': [1.2, 1.1, 2.0, 2.2]})
            st.write("1. 正向压降 (Vf)")
            ev_diode = st.data_editor(v_df_diode, num_rows="dynamic", key="v_diode")
            
            e_df_diode = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Erec (mJ)': [1.9, 3.5, 15.0, 25.0]})
            st.write("2. 反向恢复能量 (Erec)")
            ee_diode = st.data_editor(e_df_diode, num_rows="dynamic", key="e_diode")
        else:
            st.subheader("🔵 SiC 体二极管 / 同步整流")
            v_df_diode = pd.DataFrame({'Temp (℃)': [25, 150], 'Current (A)': [600.0, 600.0], 'Vsd (V)': [3.5, 4.0]})
            ev_diode = st.data_editor(v_df_diode, num_rows="dynamic", key="v_sic_diode")

# ==========================================
# TAB 2: 工况与高级参数
# ==========================================
with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. 基本电气工况**")
        vdc_act = st.number_input("实际母线 Vdc (V)", value=713.0)
        iout_rms = st.number_input("输出有效值 Iout (Arms)", value=264.5)
        fsw = st.number_input("开关频率 fsw (Hz)", value=10000)
        m_index = st.number_input("调制度 M", value=0.9)
        cosphi = st.number_input("功率因数 cosφ", value=0.9)
        mode = st.selectbox("调制模式", ["SVPWM", "PWM"])

    with c2:
        st.markdown("**2. 基准与门极电阻**")
        v_ref = st.number_input("数据手册测试 Vref (V)", value=600.0)
        rg_on_ref = st.number_input("手册测试 Rg_on (Ω)", value=2.5)
        rg_off_ref = st.number_input("手册测试 Rg_off (Ω)", value=20.0)

    with c3:
        st.markdown("**3. 高级系数 (对标计算书)**")
        kv_main = st.number_input("主管开关电压系数 (Kv)", value=1.3)
        if "IGBT" in device_type:
            kv_frd = st.number_input("FRD 开关电压系数 (Kv)", value=0.6)

# ==========================================
# TAB 3: 核心求解器
# ==========================================
with tab3:
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

    def calc_sw_power_integral(df, i_pk_lookup, tj, item_name, fsw, vdc_act, v_ref, kv):
        """正弦半波数值积分 (基于分摊后的查找电流)"""
        theta_arr = np.linspace(0, math.pi, 100)
        i_inst = i_pk_lookup * np.sin(theta_arr)
        
        e_sum = 0.0
        for iv in i_inst:
            e_sum += safe_interp(df, iv, tj, item_name)
            
        e_mean_mj = e_sum / len(i_inst)
        p_sw_lookup = fsw * 0.5 * (e_mean_mj / 1000.0) * ((vdc_act / v_ref) ** kv)
        return p_sw_lookup

    if st.button("🚀 执行联合仿真", use_container_width=True):
        # --- 核心倍率逻辑：根据芯片级/模块级确定查表电流 ---
        mult = n_chips if "芯片" in scheme else 1
        i_lookup_rms = iout_rms / mult
        i_pk = math.sqrt(2) * i_lookup_rms # 查表用的峰值电流
        
        theta = math.acos(cosphi)
        loop_count = 1 if "开环" in sim_mode else 15
        tj_current = fixed_tj if "开环" in sim_mode else t_case + 5.0

        for _ in range(loop_count):
            # --- 1. 动态提取硅片参数 V0 和 r ---
            v_pk_m = safe_interp(ev_main, i_pk, tj_current, 'V_drop (V)')
            v_hf_m = safe_interp(ev_main, i_pk/2, tj_current, 'V_drop (V)')
            r_m = (v_pk_m - v_hf_m) / (i_pk / 2) if i_pk > 0 else 0
            v0_m = max(0.0, v_pk_m - r_m * i_pk)

            if "IGBT" in device_type:
                v_pk_d = safe_interp(ev_diode, i_pk, tj_current, 'Vf (V)')
                v_hf_d = safe_interp(ev_diode, i_pk/2, tj_current, 'Vf (V)')
                r_d = (v_pk_d - v_hf_d) / (i_pk / 2) if i_pk > 0 else 0
                v0_d = max(0.0, v_pk_d - r_d * i_pk)

            # --- 2. 解析公式严绝对标 (单芯/单体导通损耗) ---
            if mode == "SVPWM":
                p_cond_v0_m = 0.25 * m_index * i_pk * v0_m * cosphi
                coeff_r_m = (24*cosphi - 2*math.sqrt(3)*math.cos(2*theta) - 3*math.sqrt(3)) / 24
                p_cond_r_m = coeff_r_m * (m_index * r_m * i_pk**2) / math.pi
                p_cond_main_lookup = p_cond_v0_m + p_cond_r_m
                
                if "IGBT" in device_type:
                    p_cond_v0_d = ((4 - m_index * math.pi * cosphi) / 4) * (v0_d * i_pk) / math.pi
                    coeff_r_d = (6*math.pi - 24*m_index*cosphi + 2*math.sqrt(3)*m_index*math.cos(2*theta) + 3*math.sqrt(3)*m_index) / 24
                    p_cond_r_d = coeff_r_d * (r_d * i_pk**2) / math.pi
                    p_cond_diode_lookup = p_cond_v0_d + p_cond_r_d

            else:
                p_cond_main_lookup = v0_m * i_pk * (1/(2*math.pi) + m_index*cosphi/8) + r_m * i_pk**2 * (1/8 + m_index*cosphi/(3*math.pi))
                if "IGBT" in device_type:
                    p_cond_diode_lookup = v0_d * i_pk * (1/(2*math.pi) - m_index*cosphi/8) + r_d * i_pk**2 * (1/8 - m_index*cosphi/(3*math.pi))
                                # --- 3. 开关损耗计算 (单芯/单体数值积分) ---
            p_on_lookup = calc_sw_power_integral(ee_main, i_pk, tj_current, 'Eon (mJ)', fsw, vdc_act, v_ref, kv_main)
            p_off_lookup = calc_sw_power_integral(ee_main, i_pk, tj_current, 'Eoff (mJ)', fsw, vdc_act, v_ref, kv_main)
            p_sw_main_lookup = p_on_lookup + p_off_lookup

            if "IGBT" in device_type:
                p_sw_diode_lookup = calc_sw_power_integral(ee_diode, i_pk, tj_current, 'Erec (mJ)', fsw, vdc_act, v_ref, kv_frd)
            else:
                p_cond_diode_lookup = 0.0
                p_sw_diode_lookup = 0.0

            # --- 4. 倍率还原 (单芯 -> 模块) ---
            p_cond_main = p_cond_main_lookup * mult
            p_sw_main = p_sw_main_lookup * mult
            p_cond_diode = p_cond_diode_lookup * mult
            p_sw_diode = p_sw_diode_lookup * mult

            p_total = p_cond_main + p_sw_main + p_cond_diode + p_sw_diode
            
            # --- 5. 闭环反馈 ---
            if "闭环" in sim_mode:
                tj_new = t_case + p_total * rth_jc
                if abs(tj_new - tj_current) < 0.05: break
                tj_current = tj_new

        # --- 结果展示面板 ---
        st.success(f"✅ 算法同步完成！当前倍率控制：方案为 [{scheme}]，实际应用并联放大系数 N = {mult}")
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("目标/收敛结温 Tj", f"{tj_current:.1f} ℃")
        r2.metric("主管导通总损耗", f"{p_cond_main:.1f} W")
        r3.metric("主管开关总损耗", f"{p_sw_main:.1f} W")
        r4.metric("器件总发热功率", f"{p_total:.1f} W")

        if "IGBT" in device_type:
            st.divider()
            st.markdown("#### 🔵 FRD 二极管损耗明细")
            dr1, dr2, dr3 = st.columns(3)
            dr1.metric("FRD 导通总损耗", f"{p_cond_diode:.1f} W")
            dr2.metric("FRD 开关(恢复)总损耗", f"{p_sw_diode:.1f} W")
            dr3.metric("FRD 总占比", f"{(p_cond_diode+p_sw_diode)/p_total * 100:.1f} %" if p_total > 0 else "0 %")
