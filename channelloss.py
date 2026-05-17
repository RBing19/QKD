import numpy as np
import matplotlib.pyplot as plt
from numpy import exp

import warnings
warnings.filterwarnings('ignore')
def h(x):
    # 处理x=0或x=1的边界情况
    x = np.clip(x, 1e-10, 1 - 1e-10)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
# 参数设置
eta_d = 0.9  # 探测器效率
p_dc =5e-7  # 暗计数概率
e_opt = 0.01  # 光学错误率
f = 1.16  # 误差纠正效率
q = 0.45  # 基协调因子
Y0 = 5e-7  # 实验参数
e00 = 0.5


# 计算Q和E_b的函数
def Q_eb(mu, eta_total):
    Q = 1 - (1 - p_dc) ** 2 * np.exp(-eta_total * mu)
    E_b_Q = e_opt * eta_total * mu * np.exp(-eta_total * mu) + 0.5 * p_dc
    E_b = E_b_Q / Q if Q > 0 else 0.5
    return Q, E_b

def compute_Q_Eb(mu, v1, v2, eta_total):
    Q_mu = 1 - (1 - p_dc) ** 2 * np.exp(-eta_total * mu)
    E_b_Qu = e_opt * eta_total * mu * np.exp(-eta_total * mu) + 0.5 * p_dc
    E_mu = E_b_Qu / Q_mu if Q_mu > 0 else 0.5

    Q_v1 = 1 - (1 - p_dc) ** 2 * np.exp(-eta_total * v1)
    E_v1 = (e_opt * eta_total * v1 * np.exp(-eta_total * v1) + 0.5 * p_dc) / Q_v1 if Q_v1 > 0 else 0.5

    Q_v2 = 1 - (1 - p_dc) ** 2 * np.exp(-eta_total * v2)
    E_v2 = (e_opt * eta_total * v2 * np.exp(-eta_total * v2) + 0.5 * p_dc) / Q_v2 if Q_v2 > 0 else 0.5

    return Q_mu, Q_v1, Q_v2, E_mu, E_v1, E_v2


# 计算密钥率下界
def key_rate(mu, v1,v2,eta_AE,eta_ch):
    eta_total = eta_ch * eta_d  # 使用全局eta_ch
    Q_mu, Q_v1, Q_v2, E_mu, E_v1, E_v2 = compute_Q_Eb(mu, v1, v2,eta_total)
    # 估计w20u 这里只能把多光子成分下界设为0，要计算其下界，相应的上界，上界不便于放缩为0
    t11 = v1 * Q_v2 * exp(v2) - v2 * Q_v1 * exp(v1) - (v1 * exp(v2) - v2 * exp(v1)) * Y0
    t12 = v1 * v2 * (v2 - v1) * (1 - eta_AE) ** 2 * 0.5
    w20u = t11 / t12  # w20上界  t1*代表w20的中间参数
    if w20u > 1 or w20u < 0:
        w20u = 1
    print("w20u(上界)为", w20u)
    # 估计SUM1上下界
    # 先联合计算再分配（按比例分配上下界）
    S1 = v1 ** 2 * Q_v2 * exp(v2) - v2 ** 2 * Q_v1 * exp(v1)
    A1 = v1 ** 2 * exp(v2) - v2 ** 2 * exp(v1)
    B1 = v1 * v2 * (v1 - v2)
    # 估计多光子项的上下界
    SUM1L = (S1 - A1 * Y0) / B1
    print("SUM1L", SUM1L)

    t21 = t11
    # 使用w10与w11的联合下界SUML）
    t22 = v1 * v2 * (v2 ** 2 - v1 ** 2) * (Q_mu * exp(mu) - Y0 * exp(mu) - SUM1L * mu) / mu ** 3
    t23 = v1 * v2 * (v2 - v1) * 0.5 * (mu - v1 - v2) / mu
    print("t21-t22", t21 - t22)
    print("t23", t23)
    SUM2L = (t21 - t22) / t23  # w20下界
    w20l = (SUM2L - 2 * eta_AE * (1 - eta_AE) - eta_AE ** 2) / (1 - eta_AE) ** 2
    print("w20l", w20l)
    if w20l < 0 or w20l > 1:  # or w20l>1:
        w20l = 0

    t24 = S1 - A1 * Y0
    t25 = v1 ** 2 * v2 ** 2 * (v2 - v1) * (exp(mu) * Q_mu - exp(mu) * Y0 - mu ** 2 * 0.5 * SUM2L) / mu ** 3
    t26 = v1 * v2 * (v1 - v2) * (1 + v1 * v2 / mu ** 2)
    SUM1U = (t24 - t25) / t26
    print("SUM1U", SUM1U)

    w10u = SUM1U / (1 - eta_AE)
    print("w10u1", w10u)
    # if w10u > w20u:
    #     w10u = w20u
    w11l = (SUM1L - (1 - eta_AE) * w10u) / eta_AE
    print("w11l1", w11l)
    if w11l < 0:
        w11l = 0
    w11u = SUM1U / eta_AE
    print("w11u1", w11u)
    if w11u > 1:
        w11u = 1
    w10l = (SUM1L - eta_AE * w11u) / (1 - eta_AE)
    print("w10l1", w10l)
    # w10l=tttt/(eta_AE**6*(1-eta_AE))
    if w10l < 0:
        w10l = 0

    print("w10l", w10l, "w10u", w10u)
    print("w11l", w11l, "w11u", w11u)
    Q10l = mu * exp(-mu) * (1 - eta_AE) * w10l
    Q11l = mu * exp(-mu) * eta_AE * w11l
    print("Q10l:", Q10l)
    print("Q11l:", Q11l)
    Q20l = mu ** 2 * 0.5 * exp(-mu) * (1 - eta_AE) ** 2 * w20l
    print("Q20l,", Q20l)
    if w20l < w10l:
        w20l = w10l
    tt1 = exp(-mu * eta_AE) - exp(-mu) - mu * exp(-mu) * (1 - eta_AE)
    print("tt1", tt1)
    Qsum20l = w20l * tt1
    print("Qsum20l", Qsum20l)  # Qsuml0是否不受（1-eta）过小的结果
    Qsum0l = Qsum20l + Q10l

    t41 = v1 ** 2 * exp(v2) * Q_v2 * E_v2 - v2 ** 2 * exp(v1) * Q_v1 * E_v1
    print("t41", t41)
    t42 = e00 * Y0 * (v1 ** 2 * exp(v2) - v2 ** 2 * exp(v1))
    print("t42", t42)
    t44 = v1 ** 2 * v2 ** 2 * (v2 - v1) / mu ** 3 * (Q_mu * E_mu * exp(mu) - e00 * Y0 * exp(mu))
    # print("t44",t44)
    t45 = (v1 * v2 * (v1 - v2) - v1 ** 2 * v2 ** 2 * (v2 - v1) / mu ** 2) * eta_AE * w11l
    # print("t45",t45)
    e11u = (t41 - t42 - t44) / t45
    print("e11u1(上界)为", e11u)
    if e11u > 0.5 or e11u < 0:
        e11u = 0.5
    print("e11u", e11u)

    R = q * (Y0+Qsum0l + Q11l * (1 - h(e11u)) - f * h(E_mu) * Q_mu)
    return R

channel_losses_dB = np.linspace(20, 40, 200)
eta_ch_values = 10 ** (-channel_losses_dB / 10)


key_rate_values11= []
key_rate_values12= []
key_rate_values13= []
key_rate_values21= []
key_rate_values22= []
key_rate_values22= []
key_rate_values23= []
# key_rate_values4= []

mu1=0.5
# mu2=0.5
v11=0.1
v21=0.2
v12=0.01
v22=0.02
v13=0.001
v23=0.002
eta_AE1 = 0.0005  # 根据文档，典型值为0.1
eta_AE2 = 0.001

for eta_ch_val in eta_ch_values:
    # 0.0005
    R_opt11 =key_rate(mu1, v11,v21,eta_AE1,eta_ch_val)
    key_rate_values11.append(R_opt11)
    R_opt12 = key_rate(mu1, v12, v22, eta_AE1, eta_ch_val)
    key_rate_values12.append(R_opt12)
    R_opt13 = key_rate(mu1, v13, v23, eta_AE1, eta_ch_val)
    key_rate_values13.append(R_opt13)
    #0.001
    R_opt21 = key_rate(mu1, v11, v21, eta_AE2, eta_ch_val)
    key_rate_values21.append(R_opt21)
    R_opt22 = key_rate(mu1, v12, v22, eta_AE2, eta_ch_val)
    key_rate_values22.append(R_opt22)
    R_opt23 = key_rate(mu1, v13, v23, eta_AE2, eta_ch_val)
    key_rate_values23.append(R_opt23)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 图8(a): 密钥率 vs eta_AE - y轴对数坐标，x轴普通坐标
ax1.semilogy(channel_losses_dB, key_rate_values11, 'b-', linewidth=2, label='v1=0.1,v2=0.2')
ax1.semilogy(channel_losses_dB, key_rate_values12, 'r-', linewidth=2, label='v1=0.01,v2=0.02')
ax1.semilogy(channel_losses_dB, key_rate_values13, 'k-', linewidth=2, label='v1=0.001,v2=0.002')
ax1.set_xlabel('channel_losses (dB)',fontsize=16)
ax1.set_ylabel('Secret Key Rate [per pulse]', fontsize=16)
ax1.set_title('(a) $\\eta_{AE}$=0.0005', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, which="both", ls="--", alpha=1)
ax1.tick_params(axis='both', which='major', labelsize=14)

# 图8(b): 最优mu vs eta_AE - y轴对数坐标，x轴普通坐标
ax2.semilogy(channel_losses_dB, key_rate_values21, 'b-', linewidth=2, label='v1=0.1,v2=0.2')
ax2.semilogy(channel_losses_dB, key_rate_values22, 'r-', linewidth=2, label='v1=0.01,v2=0.02')
ax2.semilogy(channel_losses_dB, key_rate_values23, 'k-', linewidth=2, label='v1=0.001,v2=0.002')
ax2.set_xlabel('channel_losses (dB)',fontsize=16)
# ax2.set_ylabel('Secret Key Rate [per pulse]', fontsize=16)
ax2.set_title('(b) $\\eta_{AE}$=0.001', fontsize=16)
ax2.grid(True, which="both", ls="--", alpha=1)
ax2.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.show()

