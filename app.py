import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD系统级电热仿真-完全体", layout="wide")

# ==========================================
# 顶部：工程随手记与操作手册 (防止遗忘)
# ==========================================
st.title("🛡️ 功率模块全工况电热联合仿真平台 (知识传承版)")

with st.expander("📝 查看工程随手记 & 快速操作指南 (点此做笔记)", expanded=True):
    col_note1, col_note2 = st.columns([1, 1])
    with col_note1:
        st.markdown("""
        **🚀 快速操作流程：**
        1. **左侧边栏**：设定芯片类型 (IGBT/SiC) 和数据来源 (单芯/模块)。
        2. **第一步表格**：从规格书/测试报告复制粘贴 V-I 和 E-I 矩阵。
        3. **第二步工况**：输入实际母线电压、电流。**反拖**工况选“Regeneration”。
        4. **点击计算**：直接获取用于 **Icepak/Flotherm** 的热源功率数据。
        """)
    with col_note2:
        user_notes = st.text_area("🗒️ 仿真备注 (例如：XX项目、XX规格书对标记录...)", 
                                  placeholder="在这里输入你的实验笔记，方便截图保存...", height=120)
        st.caption("提示：笔记仅限当前页面有效，建议计算完成后连同结果一起截图存档。")

# ==========================================
# 侧边栏: 核心逻辑定义
# ==========================================
with st.sidebar:
    st.header("⚙️ 核心技术架构")
    device_type = st.radio("1. 模块芯片技术类型", ["IGBT + FRD (传统硅基)", "SiC MOSFET (碳化硅)"], 
                           help="【物理逻辑】IGBT模式会计算二极管反向恢复损耗；SiC模式会自动将Vce0压降强制置0（纯阻性特性）。")
    
    st.divider()
    st.header("🧮 原始数据规格 (必填)")
    st.warning("⚠️ 此处决定了程序如何‘理解’你填入的数据表格。")
    cond_data_type = st.radio("A. 导通 V-I 表格代表：", ["单芯片数据 (Bare Die)", "模块半桥数据 (Module)"],
                             help="如果你填的是单颗晶圆测试数据，选单芯；如果是成品模块端子数据，选模块。")
    n_src_cond = st.number_input("V-I 原测模块芯片数", value=6, min_value=1, help="测试该曲线时，模块内部实际上并联了几颗芯片？")
    
    sw_data_type = st.radio("B. 开关 E-I 表格代表：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"],
                            help="双脉冲测试一般给出的是整臂/整模块的能量和。")
    n_src_sw = st.number_input("E-I 原测模块芯片数", value=6, min_value=1, help="双脉冲测试时，该模块内部实际并联的芯片数。")

    st.divider()
    st.header("🎯 仿真目标规模重构")
    n_sim = st.number_input("目标仿真芯片数 (N_sim)", value=6, min_value=1, 
                            help="【扩容/减容】你想评估多大规模的模块？程序会自动根据此并联数折算损耗。")

    st.divider()
    st.header("🔄 热学计算工作流")
    sim_mode = st.radio("模式选择", ["A. 开环盲算 (已知结温)", "B. 闭环迭代 (已知热阻)"],
                        help="开环模式下你可以手动强制芯片在150℃下发热；闭环模式根据热阻和水温反算真实平衡结温。")
    if "闭环" in sim_mode:
        rth_jc = st.number_input("封装热阻 RthJC (K/W)", value=0.065, format="%.4f", help="单颗芯片的热阻 Rth")
        t_case = st.number_input("基板/水温 Tc (℃)", value=65.0, help="散热板表面的参考温度")
    else:
        fixed_tj = st.number_input("设定全局目标结温 Tj (℃)", value=150.0)

# ==========================================
# 第一部分: 数据录入
# ==========================================
st.divider()
st.header("📊 第一步：特性数据录入 (归一化中心)")
st.info("💡 **操作提醒**：无论原始数据是单芯还是模块，程序都会在此将其‘拉平’为标准单芯模型。")

col_t, col_d = st.columns(2)
with col_t:
    st.subheader("🔴 主开关管 (IGBT / SiC)")
    v_df_main = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'V_drop (V)': [1.1, 1.05, 2.2, 2.5]})
    st.caption("1. 导通特性 (Vce 或 Vds)")
    ev_main = st.data_editor(v_df_main, num_rows="dynamic", key="v_main")
    
    e_df_main = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Eon (mJ)': [5.9, 8.5, 70.0, 95.0], 'Eoff (mJ)': [4.9, 7.2, 45.0, 60.0]})
    st.caption("2. 开关能量矩阵 (Eon, Eoff)")
    ee_main = st.data_editor(e_df_main, num_rows="dynamic", key="e_main")

with col_d:
    st.subheader("🔵 续流二极管 (FRD / Body Diode)")
    v_df_diode = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Vf (V)': [1.2, 1.1, 2.0, 2.2]})
    st.caption("1. 正向压降 (Vf 或 Vsd)")
    ev_diode = st.data_editor(v_df_diode, num_rows="dynamic", key="v_diode")
    
    e_df_diode = pd.DataFrame({'Temp (℃)': [25, 150, 25, 150], 'Current (A)': [100.0, 100.0, 600.0, 600.0], 'Erec (mJ)': [1.9, 3.5, 15.0, 25.0]})
    st.caption("2. 反向恢复能量 (Erec)")
    ee_diode = st.data_editor(e_df_diode, num_rows="dynamic", key="ee_diode")

# ==========================================
# 第二部分: 工况与高级系数 (带详细备注)
# ==========================================
st.divider()
st.header("⚙️ 第二步：全场景工况与物理修正系数")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**⚡ 车辆/电驱动工况**")
    op_mode = st.selectbox("🏎️ 运行场景切换", ["电动/巡航 (Motoring)", "制动/反拖 (Regeneration)", "最恶劣堵转 (Stall)"],
                           help="电动模式IGBT主热；反拖模式二极管主热；堵转模式计算峰值发热极限。")
    vdc_act = st.number_input("母线 V_dc (V)", value=713.0, help="逆变器运行电压")
    iout_rms = st.number_input("有效值 I_out (A)", value=264.5, help="输出相电流有效值")
    fsw = st.number_input("开关频率 f_sw (Hz)", value=10000)
    fout = st.number_input("输出频率 f_out (Hz)", value=200.0, help="通常200Hz以上。若低于5Hz建议切至堵转模式。")
    m_index = st.number_input("调制系数 M", value=0.90, help="0.0-1.1之间，通常取0.9")
    cosphi = st.number_input("功率因数 cos_phi", value=0.90, help="电流与电压相位差余弦")
    mode = st.selectbox("调制模式选择", ["SVPWM", "SPWM"])

    if fout < 5.0 and "堵转" not in op_mode:
        st.warning(f"⚠️ 当前输出频率 {fout}Hz 极低，建议切换至【最恶劣堵转】模式。")

with c2:
    st.markdown("**📏 测试基准与阻值**")
    v_ref = st.number_input("规格书基准 V_nom (V)", value=600.0, help="测双脉冲时的电压环境")
    t_ref_dp = st.number_input("规格书基准 T_ref (℃)", value=150.0, help="测双脉冲时的结温环境")
    rg_on_ref = st.number_input("手册 R_g,on (Ω)", value=2.5, help="规格书测试Eon对应的电阻")
    rg_off_ref = st.number_input("手册 R_g,off (Ω)", value=20.0, help="规格书测试Eoff对应的电阻")
    rg_on_act = st.number_input("实际 R_on (Ω)", value=2.5, help="你设计的门极驱动开通电阻")
    rg_off_act = st.number_input("实际 R_off (Ω)", value=20.0, help="你设计的门极驱动关断电阻")

with c3:
    st.markdown("**📈 拟合修正指数**")
    kv_on = st.number_input("开通指数 K_v_on", value=1.30, help="电压对开通能量的影响斜率")
    kv_off = st.number_input("关断指数 K_v_off", value=1.30)
    kv_frd = st.number_input("续流指数 K_v_frd", value=0.60, help="电压对Erec的影响斜率")
    ki_frd = st.number_input("续流电流指数 K_i_frd", value=0.60, help="电流对Erec的影响斜率")
    r_arm_mohm = st.number_input("桥臂内阻 R_arm (mΩ)", value=0.0, help="【默认为0】若包含端子压降损耗则填入此数")

with c4:
    st.markdown("**🌡️ 温度与电阻系数**")
    kron = st.number_input("电阻系数 K_ron", value=0.30, help="电阻偏离基准时对能量的放大指数")
    kroff = st.number_input("电阻系数 K_roff", value=0.50)
    t_coeff_igbt = st.number_input("主管温漂系数 (1/K)", value=0.003, format="%.4f", help="IGBT能量随温升比例")
    t_coeff_frd = st.number_input("续流温漂系数 (1/K)", value=0.006 if "IGBT" in device_type else 0.003, format="%.4f", help="二极管能量随温升比例")

# ==========================================
# 第三部分: 核心计算引擎 (保持精密逻辑)
# ==========================================
st.divider()
st.header("🚀 第三步：执行全工况联合仿真")

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

def calc_sw_logic(df, i_pk, tj, item_name, fsw, vdc, vref, kv, ract, rref, kr, tc, tref):
    e_base = safe_interp(df, i_pk, tj, item_name)
    active_tc = 0.0 if (not df.dropna().empty and len(df.dropna()['Temp (℃)'].unique()) >= 2) else tc
    rcorr = math.pow(ract / rref, kr) if rref > 0 else 1.0
    tcorr = 1.0 + active_tc * (tj - tref)
    return e_base * rcorr * tcorr * math.pow(vdc / vref, kv)

if st.button("🚀 执 行 全 工 况 仿 真 计 算", use_container_width=True):
    # 底层归一化
    norm_ev_m, norm_ev_d = normalize_vi_df(ev_main, n_src_cond), normalize_vi_df(ev_diode, n_src_cond)
    norm_ee_m, norm_ee_d = normalize_ei_df(ee_main, n_src_sw, ['Eon (mJ)', 'Eoff (mJ)']), normalize_ei_df(ee_diode, n_src_sw, ['Erec (mJ)'])

    # 工况环境
    i_pk_chip = math.sqrt(2) * (iout_rms / n_sim)
    r_arm_chip = (r_arm_mohm / 1000.0) * n_sim
    active_cosphi = -cosphi if "反拖" in op_mode else cosphi
    theta = math.acos(active_cosphi) if abs(active_cosphi) <= 1.0 else 0
    
    loop_count = 1 if "开环" in sim_mode else 15
    tj_current = fixed_tj if "开环" in sim_mode else t_case + 5.0

    for _ in range(loop_count):
        v_pk_m = safe_interp(norm_ev_m, i_pk_chip, tj_current, 'V_drop (V)')
        v_hf_m = safe_interp(norm_ev_m, i_pk_chip/2, tj_current, 'V_drop (V)')
        r_m_chip = (v_pk_m - v_hf_m) / (i_pk_chip / 2) if i_pk_chip > 0 else 0
        v0_m = 0.0 if "SiC" in device_type else max(0.0, v_pk_m - r_m_chip * i_pk_chip)
        
        v_pk_d = safe_interp(norm_ev_d, i_pk_chip, tj_current, 'Vf (V)')
        v_hf_d = safe_interp(norm_ev_d, i_pk_chip/2, tj_current, 'Vf (V)')
        r_d_chip = (v_pk_d - v_hf_d) / (i_pk_chip / 2) if i_pk_chip > 0 else 0
        v0_d = max(0.0, v_pk_d - r_d_chip * i_pk_chip)

        if "堵转" in op_mode:
            d_max = 0.5 * (1 + m_index)
            p_cond_m_chip = d_max * (v0_m * i_pk_chip + (r_m_chip + r_arm_chip) * i_pk_chip**2)
            p_cond_d_chip = (1-d_max) * (v0_d * i_pk_chip + (r_d_chip + r_arm_chip) * i_pk_chip**2)
            e_sum = calc_sw_logic(norm_ee_m, i_pk_chip, tj_current, 'Eon (mJ)', 1, vdc_act, v_ref, kv_on, rg_on_act, rg_on_ref, kron, t_coeff_igbt, t_ref_dp) + \
                    calc_sw_logic(norm_ee_m, i_pk_chip, tj_current, 'Eoff (mJ)', 1, vdc_act, v_ref, kv_off, rg_off_act, rg_off_ref, kroff, t_coeff_igbt, t_ref_dp)
            p_sw_m_chip = fsw * (e_sum / 1000.0)
            p_sw_d_chip = fsw * (calc_sw_logic(norm_ee_d, i_pk_chip, tj_current, 'Erec (mJ)', 1, vdc_act, v_ref, kv_frd, 1.0, 1.0, 0.0, t_coeff_frd, t_ref_dp) / 1000.0)
        else:
            if mode == "SVPWM":
                kv0_m, kr_m = (m_index * active_cosphi)/4.0, (24*active_cosphi - 2*math.sqrt(3)*math.cos(2*theta) - 3*math.sqrt(3))/24.0
                kv0_d, kr_d = (4 - m_index*math.pi*active_cosphi)/4.0, (6*math.pi - 24*m_index*active_cosphi + 2*math.sqrt(3)*m_index*math.cos(2*theta) + 3*math.sqrt(3)*m_index)/24.0
                p_cond_m_chip = (kv0_m * v0_m * i_pk_chip) + (kr_m * (r_m_chip + r_arm_chip) * i_pk_chip**2) / math.pi
                p_cond_d_chip = (kv0_d * v0_d * i_pk_chip) / math.pi + (kr_d * (r_d_chip + r_arm_chip) * i_pk_chip**2) / math.pi
            else:
                p_cond_m_chip = v0_m*i_pk_chip*(1/(2*math.pi) + m_index*active_cosphi/8) + (r_m_chip+r_arm_chip)*i_pk_chip**2*(1/8 + m_index*active_cosphi/(3*math.pi))
                p_cond_d_chip = v0_d*i_pk_chip*(1/(2*math.pi) - m_index*active_cosphi/8) + (r_d_chip+r_arm_chip)*i_pk_chip**2*(1/8 - m_index*active_cosphi/(3*math.pi))
            
            e_sum = calc_sw_logic(norm_ee_m, i_pk_chip, tj_current, 'Eon (mJ)', 1, vdc_act, v_ref, kv_on, rg_on_act, rg_on_ref, kron, t_coeff_igbt, t_ref_dp) + \
                    calc_sw_logic(norm_ee_m, i_pk_chip, tj_current, 'Eoff (mJ)', 1, vdc_act, v_ref, kv_off, rg_off_act, rg_off_ref, kroff, t_coeff_igbt, t_ref_dp)
            p_sw_m_chip = (1.0/math.pi) * fsw * (e_sum / 1000.0)
            i_corr = (iout_rms / ((i_pk_chip*n_sim) / math.sqrt(2))) ** ki_frd if ki_frd > 0 else 1.0
            p_sw_d_chip = (1.0/math.pi) * fsw * (calc_sw_logic(norm_ee_d, i_pk_chip, tj_current, 'Erec (mJ)', 1, vdc_act, v_ref, kv_frd, 1.0, 1.0, 0.0, t_coeff_frd, t_ref_dp) / 1000.0) * i_corr

        p_total = (p_cond_m_chip + p_sw_m_chip + p_cond_d_chip + p_sw_d_chip) * n_sim
        if "闭环" in sim_mode:
            tj_new = t_case + p_total * rth_jc
            if abs(tj_new - tj_current) < 0.05: break
            tj_current = tj_new

    # ==========================================
    # 结果展示面板
    # ==========================================
    st.success(f"✅ 计算完成！当前状态：{op_mode} | 并联 N: {n_sim}")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("收敛结温 Tj", f"{tj_current:.1f} ℃")
    r2.metric("整臂发热总功耗", f"{p_total:.1f} W")
    r3.metric("🔴 主芯片总损耗", f"{(p_cond_m_chip + p_sw_m_chip)*n_sim:.1f} W")
    r4.metric("🔵 二极管总损耗", f"{(p_cond_d_chip + p_sw_d_chip)*n_sim:.1f} W")

    st.divider()
    col_m, col_d = st.columns(2)
    with col_m:
        st.info(f"🔴 主芯片单颗发热率: **{p_cond_m_chip + p_sw_m_chip:.2f} W**")
    with col_d:
        st.info(f"🔵 二极管单颗发热率: **{p_cond_d_chip + p_sw_d_chip:.2f} W**")

    # ==========================================
    # 底部：底层物理公式全量展开展示 (知识传承基石)
    # ==========================================
    with st.expander("🔬 查看底层应用的完整物理公式 (与手稿笔记 100% 对标)", expanded=False):
        st.markdown("### 📘 导通损耗 (Conduction Loss)")
        st.write("程序已基于多维插值动态提取 $V_{CE0}/V_{f0}$ 及等效内阻 $R_{CE}/R_F$。其中 $I_{pk} = \sqrt{2}I_{out}$。")
        
        st.markdown("#### 1. SPWM 调制模型 (正弦波平均解析)")
        st.latex(r"P_{cond\_IGBT} = \left(\frac{1}{2\pi} + \frac{M\cos\phi}{8}\right) \cdot V_{CE0} \cdot I_{pk} + \left(\frac{1}{8} + \frac{M\cos\phi}{3\pi}\right) \cdot R_{CE} \cdot I_{pk}^2")
        st.latex(r"P_{cond\_D} = \left(\frac{1}{2\pi} - \frac{M\cos\phi}{8}\right) \cdot V_{F0} \cdot I_{pk} + \left(\frac{1}{8} - \frac{M\cos\phi}{3\pi}\right) \cdot R_{F} \cdot I_{pk}^2")

        st.markdown("#### 2. SVPWM 调制模型 (空间矢量马鞍波解析)")
        st.latex(r"P_{cond\_IGBT} \approx \frac{1}{4} M I_{pk} V_{CE0} \cos\phi + \left( \frac{24\cos\phi - 2\sqrt{3}\cos(2\varphi) - 3\sqrt{3}}{24\pi} \right) M R_{CE} I_{pk}^2")
        st.latex(r"P_{cond\_D} \approx \left(\frac{4 - M\pi\cos\phi}{4\pi}\right) V_{F0} I_{pk} + \left(\frac{6\pi - 24M\cos\phi + 2\sqrt{3}M\cos(2\varphi) + 3\sqrt{3}M}{24\pi}\right) R_F I_{pk}^2")
        st.info("💡 补充：若选择 **SiC 模块** 且处于开通状态，程序底层将强制令 $V_{CE0} = 0$，仅保留纯电阻损耗项。")

        st.markdown("---")
        st.markdown("### 📕 开关损耗 (Switching Loss)")
        st.write("程序已对基础查表能量进行 **驱动电阻补偿、电压拟合指数补偿、温漂漂移补偿** 的三维修正：")
        
        st.markdown("#### 1. 主开关管开通与关断损耗 ($P_{sw\_on}$ & $P_{sw\_off}$)")
        st.latex(r"E_{on\_adj} = E_{on\_nom}(I_{pk}) \cdot \left(\frac{R_{on\_act}}{R_{g,on\_ref}}\right)^{K_{ron}} \cdot \left[1 + T_{c\_igbt}(T_j - T_{ref\_dp})\right]")
        st.latex(r"P_{sw\_on} = \frac{1}{\pi} f_{sw} E_{on\_adj} \cdot \left(\frac{V_{dc}}{V_{nom}}\right)^{K_{v\_on}}")
        st.write("*(关断损耗 $P_{sw\_off}$ 同理，分别代入 $E_{off\_nom}$、关断电阻及其特有系数 $K_{roff}, K_{v\_off}$)*")

        st.markdown("#### 2. 续流二极管反向恢复损耗 ($P_{sw\_FRD}$)")
        st.write("已集成用于修正不同工况电流偏差的基波放大指数 $K_{i\_frd}$：")
        st.latex(r"E_{rec\_adj} = E_{rec\_nom}(I_{pk}) \cdot \left[1 + T_{cerr}(T_j - T_{ref\_dp})\right]")
        st.latex(r"P_{sw\_FRD} = \frac{1}{\pi} f_{sw} E_{rec\_adj} \cdot \left(\frac{I_{out}}{I_{pk}/\sqrt{2}}\right)^{K_{i\_frd}} \cdot \left(\frac{V_{dc}}{V_{nom}}\right)^{K_{v\_frd}}")
        
        st.markdown("---")
        st.markdown("### 🛑 极限堵转工况 (Stall Mode)")
        st.write("当选择最恶劣堵转时，程序将舍弃 $1/\pi$ 正弦平滑均值，按照最大占空比进行直流连续轰炸发热计算：")
        st.latex(r"P_{cond\_stall} = D_{max} \cdot (V_0 I_{pk} + R I_{pk}^2) \quad \text{其中 } D_{max} = \frac{1+M}{2}")
        st.latex(r"P_{sw\_stall} = f_{sw} \cdot E_{adj\_total}(I_{pk})")
