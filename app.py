import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD全架构仿真-精准映射版", layout="wide")
st.title("🛡️ 功率模块多架构电热联合仿真平台 (公式精准映射版)")

# --- 侧边栏：全局架构与工作流 ---
with st.sidebar:
    st.header("⚙️ 核心架构设置")
    device_type = st.radio("1. 模块芯片技术架构", ["IGBT + FRD (传统硅基)", "SiC MOSFET (碳化硅)"])
    
    st.divider()
    st.header("🧮 数据录入方案")
    scheme = st.radio("2. 查表数据层级", ["芯片级方案 (Chip-level)", "模块级方案 (Module-level)"])
    n_chips = st.number_input("并联芯片数 (N)", value=6 if "芯片" in scheme else 1, min_value=1)
    
    st.divider()
    st.header("🔄 仿真工作流选择")
    sim_mode = st.radio("3. 热学计算模式", ["A. 开环盲算 (已知结温求损耗)", "B. 闭环迭代 (已知热阻求结温)"])
    
    if "闭环" in sim_mode:
        rth_jc = st.number_input("输入封装热阻 RthJC (K/W)", value=0.065, format="%.4f")
        t_case = st.number_input("基板/水温 Tc (℃)", value=65.0)
    else:
        fixed_tj = st.number_input("设定全局目标结温 Tj (℃)", value=150.0)

# --- 页面主体 ---
tab1, tab2, tab3 = st.tabs(["📊 第一步：特性数据录入", "⚙️ 第二步：工况与变量映射", "🚀 第三步：执行联合仿真"])

# ==========================================
# TAB 1: 数据录入
# ==========================================
with tab1:
    st.info("💡 快捷录入：Excel 中 Ctrl+C 选中数据，点击下方表格空单元格 Ctrl+V 粘贴。")
    col_t, col_d = st.columns(2)
    
    with col_t:
        st.subheader("🔴 主管特性 (IGBT / SiC)")
        v_df_main = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'V_drop (V)': [1.1, 1.05, 2.2, 2.5]})
        ev_main = st.data_editor(v_df_main, num_rows="dynamic", key="v_main", help="Vce 或 Vds")
        
        e_df_main = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Eon (mJ)': [5.9, 8.5, 70.0, 95.0], 'Eoff (mJ)': [4.9, 7.2, 45.0, 60.0]})
        ee_main = st.data_editor(e_df_main, num_rows="dynamic", key="e_main")

    with col_d:
        st.subheader("🔵 二极管特性 (FRD / Body Diode)")
        v_df_diode = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Vf (V)': [1.2, 1.1, 2.0, 2.2]})
        ev_diode = st.data_editor(v_df_diode, num_rows="dynamic", key="v_diode")
        
        e_df_diode = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Erec (mJ)': [1.9, 3.5, 15.0, 25.0]})
        ee_diode = st.data_editor(e_df_diode, num_rows="dynamic", key="e_diode")

# ==========================================
# TAB 2: 工况与变量映射 (严格对标 Python 映射表)
# ==========================================
with tab2:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**电气工况**")
        vdc_act = st.number_input("实际母线 V_dc (V)", value=713.0)
        iout_rms = st.number_input("输出有效值 I_out (A)", value=264.5)
        fsw = st.number_input("开关频率 f_sw (Hz)", value=10000)
        m_index = st.number_input("调制系数 M", value=0.9)
        cosphi = st.number_input("功率因数 cos_phi", value=0.9)
        mode = st.selectbox("调制模式", ["SVPWM", "SPWM"])

    with c2:
        st.markdown("**测试基准 (Nominal)**")
        v_ref = st.number_input("双脉冲基准 V_nom (V)", value=600.0)
        t_ref_dp = st.number_input("双脉冲基准 T_ref_dp (℃)", value=150.0)
        rg_on_ref = st.number_input("规格书 R_g,on (Ω)", value=2.5)
        rg_off_ref = st.number_input("规格书 R_g,off (Ω)", value=20.0)
        rg_on_act = st.number_input("实际应用 R_on (Ω)", value=2.5)
        rg_off_act = st.number_input("实际应用 R_off (Ω)", value=20.0)

    with c3:
        st.markdown("**电压与电流指数**")
        kv_igbt = st.number_input("主电压指数 K_v_igbt", value=1.3)
        kv_frd = st.number_input("续流电压指数 K_v_frd", value=0.6)
        ki_frd = st.number_input("续流电流指数 K_i_frd", value=0.6)

    with c4:
        st.markdown("**电阻与温度修正系数**")
        kron = st.number_input("开通电阻系数 K_ron", value=0.3)
        kroff = st.number_input("关断电阻系数 K_roff", value=0.5)
        if "IGBT" in device_type:
            t_coeff_rec = st.number_input("FRD 温漂 T_cerr", value=0.006, format="%.4f")
        else:
            t_coeff_rec = st.number_input("SiC 温漂 T_cesw", value=0.003, format="%.4f")

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

    def calc_sw_power_integral(df, i_pk_lookup, tj, item_name, fsw, vdc_act, v_ref, kv, r_act, r_ref, k_r, t_coeff, t_ref_dp):
        """正弦半波数值积分 (集成电阻与温度复合修正)"""
        theta_arr = np.linspace(0, math.pi, 100)
        i_inst = i_pk_lookup * np.sin(theta_arr)
        
        e_sum = 0.0
        for iv in i_inst:
            e_base = safe_interp(df, iv, tj, item_name)
            # 引入电阻修正 (R_act / R_ref)^Kr
            r_corr = math.pow(r_act / r_ref, k_r) if r_ref > 0 else 1.0
            # 引入温漂修正 E_rec_adj
            t_corr = 1.0 + t_coeff * (tj - t_ref_dp)
            e_sum += (e_base * r_corr * t_corr)
            
        e_mean_mj = e_sum / len(i_inst)
        # 积分平均功率
        p_sw_lookup = fsw * 0.5 * (e_mean_mj / 1000.0) * ((vdc_act / v_ref) ** kv)
        return p_sw_lookup

    if st.button("🚀 执行精准公式映射仿真", use_container_width=True):
        mult = n_chips if "芯片" in scheme else 1
        i_lookup_rms = iout_rms / mult
        i_pk = math.sqrt(2) * i_lookup_rms 
        theta = math.acos(cosphi)
        
        loop_count = 1 if "开环" in sim_mode else 15
        tj_current = fixed_tj if "开环" in sim_mode else t_case + 5.0

        for _ in range(loop_count):
            # --- 1. 动态提取硅片参数 V0 和 r ---
            v_pk_m = safe_interp(ev_main, i_pk, tj_current, 'V_drop (V)')
            v_hf_m = safe_interp(ev_main, i_pk/2, tj_current, 'V_drop (V)')
            r_m = (v_pk_m - v_hf_m) / (i_pk / 2) if i_pk > 0 else 0
            
            # 【核心逻辑修正】SiC 模块时，强制 V_CE0 = 0V
            if "SiC" in device_type:
                v0_m = 0.0
            else:
                v0_m = max(0.0, v_pk_m - r_m * i_pk)

            # 二极管提取
            v_pk_d = safe_interp(ev_diode, i_pk, tj_current, 'Vf (V)')
            v_hf_d = safe_interp(ev_diode, i_pk/2, tj_current, 'Vf (V)')
            r_d = (v_pk_d - v_hf_d) / (i_pk / 2) if i_pk > 0 else 0
            v0_d = max(0.0, v_pk_d - r_d * i_pk)
                        # 二极管提取
            v_pk_d = safe_interp(ev_diode, i_pk, tj_current, 'Vf (V)')
            v_hf_d = safe_interp(ev_diode, i_pk/2, tj_current, 'Vf (V)')
            r_d = (v_pk_d - v_hf_d) / (i_pk / 2) if i_pk > 0 else 0
            v0_d = max(0.0, v_pk_d - r_d * i_pk)

            # --- 2. 解析公式严绝对标 (单芯/单体导通损耗) ---
            if mode == "SVPWM":
                # 主管 SVPWM (电压分母为 4)
                p_cond_v0_m = (m_index * i_pk * v0_m * cosphi) / 4.0
                coeff_r_m = (24*cosphi - 2*math.sqrt(3)*math.cos(2*theta) - 3*math.sqrt(3)) / 24.0
                p_cond_r_m = coeff_r_m * (m_index * r_m * i_pk**2) / math.pi
                p_cond_main_lookup = p_cond_v0_m + p_cond_r_m
                
                # 二极管 SVPWM
                p_cond_v0_d = ((4 - m_index * math.pi * cosphi) / 4.0) * (v0_d * i_pk) / math.pi
                coeff_r_d = (6*math.pi - 24*m_index*cosphi + 2*math.sqrt(3)*m_index*math.cos(2*theta) + 3*math.sqrt(3)*m_index) / 24.0
                p_cond_r_d = coeff_r_d * (r_d * i_pk**2) / math.pi
                p_cond_diode_lookup = p_cond_v0_d + p_cond_r_d

            else:
                # SPWM 
                p_cond_main_lookup = v0_m * i_pk * (1/(2*math.pi) + m_index*cosphi/8) + r_m * i_pk**2 * (1/8 + m_index*cosphi/(3*math.pi))
                p_cond_diode_lookup = v0_d * i_pk * (1/(2*math.pi) - m_index*cosphi/8) + r_d * i_pk**2 * (1/8 - m_index*cosphi/(3*math.pi))

            # --- 3. 开关损耗计算 (电阻调整 + 数值积分) ---
            # Eon 修正
            p_on_lookup = calc_sw_power_integral(ee_main, i_pk, tj_current, 'Eon (mJ)', fsw, vdc_act, v_ref, kv_igbt, rg_on_act, rg_on_ref, kron, 0.0, t_ref_dp)
            # Eoff 修正
            p_off_lookup = calc_sw_power_integral(ee_main, i_pk, tj_current, 'Eoff (mJ)', fsw, vdc_act, v_ref, kv_igbt, rg_off_act, rg_off_ref, kroff, 0.0, t_ref_dp)
            p_sw_main_lookup = p_on_lookup + p_off_lookup

            # Erec 修正 (包含 T_cerr / T_cesw)
            # 引入电流指数 K_i_frd 对基波放大
            i_corr_factor = (iout_rms / (i_pk/math.sqrt(2))) ** ki_frd if ki_frd > 0 else 1.0
            p_sw_diode_base = calc_sw_power_integral(ee_diode, i_pk, tj_current, 'Erec (mJ)', fsw, vdc_act, v_ref, kv_frd, 1.0, 1.0, 0.0, t_coeff_rec, t_ref_dp)
            p_sw_diode_lookup = p_sw_diode_base * i_corr_factor

            # --- 4. 倍率还原 ---
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
        st.success(f"✅ 公式逻辑已全面挂载！已执行 {mode} 非线性解析，包含 SiC零压降检测与温漂修正。")
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("仿真收敛结温 Tj", f"{tj_current:.1f} ℃")
        r2.metric(f"主管导通总损耗", f"{p_cond_main:.1f} W")
        r3.metric(f"主管开关总损耗", f"{p_sw_main:.1f} W")
        r4.metric("器件总发热功率", f"{p_total:.1f} W")

        st.divider()
        st.markdown("#### 🔵 续流二极管 (FRD / Body Diode) 损耗明细")
        dr1, dr2, dr3 = st.columns(3)
        dr1.metric("二极管 导通损耗", f"{p_cond_diode:.1f} W")
        dr2.metric("二极管 开关(恢复)损耗", f"{p_sw_diode:.1f} W")
        dr3.metric("二极管 总占比", f"{(p_cond_diode+p_sw_diode)/p_total * 100:.1f} %" if p_total > 0 else "0 %")
        
        with st.expander("🔬 查看底层应用的逻辑状态"):
            st.write(f"▪ 模块类型识别: **{device_type}**")
            st.write(f"▪ SVPWM 电压分母修正: **已启用 (分母为 4)**")
            if "SiC" in device_type:
                st.write(f"▪ SiC 纯阻性模式: **已启用 (强制 V_ce = 0V)**")
            st.write(f"▪ 开通电阻修正 E_on_adj: **已生效 (Kron = {kron})**")
            st.write(f"▪ 关断电阻修正 E_off_adj: **已生效 (Kroff = {kroff})**")
            st.write(f"▪ 恢复温漂补偿: **已生效 (T_coeff = {t_coeff_rec})**")
                   # --- (将这段代码粘贴在 app.py 的最下方) ---
        st.divider()
        with st.expander("🔬 查看底层计算逻辑与应用的数学公式 (点击展开)"):
            st.markdown("#### 1. 导通损耗 (Conduction Loss)")
            if mode == "SVPWM":
                st.write("已启用 **SVPWM** 空间矢量调制马鞍波解析积分：")
                st.latex(r"K_{v0\_IGBT} = \frac{M \cos\phi}{4}")
                st.latex(r"K_{r\_IGBT} = \frac{24\cos\phi - 2\sqrt{3}\cos(2\varphi) - 3\sqrt{3}}{24}")
            else:
                st.write("已启用 **SPWM** 正弦波调制解析积分：")
                st.latex(r"K_{r\_IGBT} = \frac{1}{8} + \frac{M\cos\phi}{3\pi}")
            
            st.markdown("#### 2. 开关损耗 (Switching Loss - 数值积分)")
            st.write("已启用半正弦波瞬态电流离散积分，并叠加复合修正：")
            st.latex(r"P_{on} = \frac{f_{sw}}{2} \int_0^\pi E_{on}(i) \cdot \left(\frac{R_{on}}{R_{g,on}}\right)^{K_{ron}} \cdot \left(\frac{V_{dc}}{V_{nom}}\right)^{K_v} d\theta")
            st.latex(r"P_{off} = \frac{f_{sw}}{2} \int_0^\pi E_{off}(i) \cdot \left(\frac{R_{off}}{R_{g,off}}\right)^{K_{roff}} \cdot \left(\frac{V_{dc}}{V_{nom}}\right)^{K_v} d\theta")
            
            st.markdown("#### 3. 续流反向恢复 (Diode Recovery)")
            if "IGBT" in device_type:
                st.write("已启用 FRD 专属温漂与电流基波放大系数：")
                st.latex(r"E_{rec\_adj} = E_{rec}(i) \cdot \left[1 + T_{cerr}(T_j - T_{ref\_dp})\right]")
                st.latex(r"P_{rec} \propto \left(\frac{V_{dc}}{V_{nom}}\right)^{K_{v\_frd}} \cdot \left(\frac{I_{out}}{I_{pk}/\sqrt{2}}\right)^{K_{i\_frd}}")
            else:
                st.write("碳化硅 (SiC) 模式下，采用体二极管或同步整流逻辑，续流开通损耗极小。")
