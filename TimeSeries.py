import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import arma_order_select_ic  # 选择阶数
from statsmodels.graphics.tsaplots import plot_acf  # 自相关图
from statsmodels.tsa.stattools import adfuller  # 平稳性检测
from statsmodels.graphics.tsaplots import plot_pacf  # 偏自相关图
from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验
from statsmodels.tsa.arima_model import ARIMA
from tsmoothie.smoother import LowessSmoother, SplineSmoother, KalmanSmoother, WindowWrapper, PolynomialSmoother


# *----------------------------- PART 1 ----------------------------*
def EDA():
    # 1. 总体趋势图
    import matplotlib
    matplotlib.rc("font", family='FangSong')
    DAYS = 240
    plt.plot(timeSeries[0:DAYS], confirm_all[0:DAYS] - recover_all[0:DAYS] - death_all[0:DAYS], label='剩余感染人数')
    plt.plot(timeSeries[0:DAYS], death_all[0:DAYS], label='死亡病例数')
    plt.plot(timeSeries[0:DAYS], recover_all[0:DAYS], label='治愈数')
    plt.legend()
    plt.show()

    # 2. 平滑处理
    # operate smoothing
    # LOWNESS
    smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
    # 滑窗法
    smoother = WindowWrapper(LowessSmoother(smooth_fraction=0.3, iterations=1), window_shape=10)
    # Spline
    smoother = SplineSmoother(n_knots=10, spline_type='natural_cubic_spline')
    # Polynomial
    smoother = PolynomialSmoother(degree=4)

    smoother.smooth(confirm_all[0:DAYS])

    x = timeSeries[0:DAYS]
    plt.plot(timeSeries[0:DAYS], smoother.smooth_data[0], linewidth=1.5, color='red')
    plt.plot(timeSeries[0:DAYS], smoother.data[0])
    low, up = smoother.get_intervals('prediction_interval')  # 'confidence_interval'
    plt.fill_between(x, low[0], up[0], alpha=0.3)
    plt.show()

    # 3. abs(二阶差分) 与 民意指数 cov() 趋势项
    # In China
    confirm_diff2 = np.diff(confirm_all[0:131], n=2)
    BaiduIndex = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5049, 6120, 5221, 6175, 4987, 6667, 5226,
                           4062, 3714, 4242, 10526, 9309, 4793, 3461, 3916, 3940, 3951, 3957, 3679, 2779, 2747, 2160,
                           1745, 1818, 1744, 1753, 1367, 1415, 1356, 1169, 1110, 1175, 1330, 1270, 1073, 795, 670, 510,
                           533, 505, 533, 489, 451, 404, 351, 400, 405, 364, 366, 370, 391, 332, 305, 376, 387, 419,
                           431,
                           344, 403, 367, 466, 450, 384, 370, 372, 309, 347, 434, 346, 369, 335, 333, 254, 270, 305,
                           326,
                           320, 310, 282, 266, 271, 261, 300, 265, 240, 225, 226, 238, 225, 213, 237, 223, 239, 193,
                           203,
                           214, 212, 216, 200, 177, 212, 211, 204, 203, 182, 184, 182, 160, 179, 170, 175, 171, 174,
                           173, 165, 169])[1:-1]

    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(timeSeries[1:130], confirm_diff2 / confirm_diff2.max(), label='确诊人数 二阶差分')
    plt.plot(timeSeries[1:130], BaiduIndex / BaiduIndex.max(), label='百度指数 疫情拐点')
    plt.legend()
    plt.axhline(y=0., color='r', linestyle='--')
    plt.xticks(rotation=60)  # , fontsize=14
    plt.title('1/22 - 5/31 实际数据与舆情拐点匹配')
    plt.show()

    # *------------ 新增确诊与新增治愈 / 死亡 -------------*
    # 4. Patterns
    # from matplotlib.pyplot import xcorr
    Start = 0
    End = DAYS
    # ccf = statools.ccf( np.diff(recover_all[Start:End]) , np.diff(confirm_all[Start:End]) )
    # print(np.argmax(ccf) )
    plt.xcorr(x=np.diff(recover_all[Start:End]).astype(float), y=np.diff(confirm_all[Start:End]).astype(float),
              maxlags=30)
    plt.title('新增治愈与新增确诊 ccf')
    plt.show()

    plt.xcorr(x=np.diff(death_all[Start:End]).astype(float), y=np.diff(confirm_all[Start:End]).astype(float),
              maxlags=30)
    plt.title('新增死亡数与新增确诊 ccf')
    plt.show()

    return


# *---------------------------- PART 2 -----------------------------*
def arima_analysis():
    import warnings
    warnings.simplefilter("ignore")

    # 定阶数有困难, 时间序列不稳定
    # 带上后期数据, 仅二阶差分即可
    # 但前期 300 天前，用二阶差分不足以拟合时间序列
    start = 30
    DAYS = 180
    x = np.array(confirm_smooth[start:start + DAYS])
    x = np.diff(x, n=3)
    # 平稳性鉴定,定差分
    output = adfuller(x)
    print(output)

    plot_acf(x).show()  # 7
    plot_pacf(x).show()  # 15

    # 定阶数
    arma_order_select_ic(x, max_ar=7, max_ma=6, ic='aic')  # 4.3.1

    confirm_diff = np.diff(confirm_smooth)  # 单日新增
    x = confirm_diff[start: start + DAYS - 1]  # 先差分一下,曲线救国
    # or directly
    # x = np.diff(confirm_smooth[start : DAYS])

    order = (6, 2, 2)  # select by AIC (6,3,2)
    model = ARIMA(x, order=order)
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    # Ljung-Box 白噪声检验
    print(acorr_ljungbox(model_fit.resid))

    # 模型拟合效果
    model_fit.plot_predict(dynamic=False)
    plt.plot()
    plt.title('每日新增病例数拟合')
    plt.show()

    # 预测
    new_start = DAYS + start
    interval = 14
    train = pd.Series(confirm_diff[start + 1:start + DAYS], index=timeSeries[start + 1:DAYS + start])
    test = pd.Series(confirm_diff[new_start:new_start + interval], index=timeSeries[new_start: new_start + interval])

    # Forecast
    fc, se, conf = model_fit.forecast(interval, alpha=0.05)  # 95% conf
    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='过去单日新增确诊', color='blue')
    plt.plot(test, label='实际值 单日新增确诊', color='red')
    plt.plot(fc_series, label='预测值 单日新增确诊', color='sandybrown')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15)
    plt.title('单日新增确诊病例在 95% C.I.下的预测值及真实值')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

    return


class SEIR_model:
    def __init__(self, population=332875137):
        self.N = population  # population

    def forcast(self, Initial_paras=None, ForcastDays=10):
        # SEIR model
        # *--- 初始值 与 参数 ---*
        N = self.N
        beta, gamma, beta_1, alpha, E0 = self.p

        if Initial_paras is None:
            # I0, R0, E0 = self.Initial_paras
            I0, R0 = self.Initial_paras
            Flag = True
        else:
            I0, R0, E0 = Initial_paras

        S = np.zeros(ForcastDays)
        E = np.zeros(ForcastDays)
        I = np.zeros(ForcastDays)
        R = np.zeros(ForcastDays)
        S[0], E[0], I[0], R[0] = N - I0 - E0 - R0, E0, I0, R0
        for i in range(1, ForcastDays):
            S[i] = S[i - 1] - (beta * I[i - 1] + beta_1 * E[i - 1]) * S[i - 1] / N
            E[i] = (1 - alpha) * E[i - 1] + (beta * I[i - 1] + beta_1 * E[i - 1]) * S[i - 1] / N
            I[i] = (1 - gamma) * I[i - 1] + alpha * E[i - 1]
            R[i] = R[i - 1] + gamma * I[i - 1]

        return S, I, R

    def get_paras(self, confirm, deathAndRecover):
        '''
        :param confirm: 现存感染者
        :param deathAndRecover:
        :return:
        '''
        # from scipy.optimize import leastsq
        from scipy.optimize import least_squares

        Initial_paras = [confirm[0], deathAndRecover[0], ]  # I0 , R0 ,
        self.Initial_paras = Initial_paras

        def func(x, p):
            I0, R0 = Initial_paras
            N = self.N
            beta, gamma, beta_1, alpha, E0 = p
            S = np.zeros(x[-1] + 1)
            I = np.zeros(x[-1] + 1)
            R = np.zeros(x[-1] + 1)
            E = np.zeros(x[-1] + 1)
            assert isinstance(x, type(np.array([1])))
            S[0], I[0], E[0], R[0] = N - I0 - R0 - E0, E0, I0, R0
            for i in range(1, x[-1] + 1):
                # S[i] = S[i - 1] - beta * S[i - 1] * I[i - 1] / N
                # I[i] = I[i - 1] + beta * S[i - 1] * I[i - 1] / N - gamma * I[i - 1]
                # R[i] = R[i - 1] + gamma * I[i - 1]
                S[i] = S[i - 1] - (beta * I[i - 1] + beta_1 * E[i - 1]) * S[i - 1] / N
                E[i] = (1 - alpha) * E[i - 1] + (beta * I[i - 1] + beta_1 * E[i - 1]) * S[i - 1] / N
                I[i] = (1 - gamma) * I[i - 1] + alpha * E[i - 1]
                R[i] = R[i - 1] + gamma * I[i - 1]
                try:
                    assert abs(S[i] + E[i] + I[i] + R[i] - (S[i - 1] + E[i - 1] + I[i - 1] + R[i - 1])) < 10
                except:
                    print(S[i] + E[i] + I[i] + R[i], (S[i - 1] + E[i - 1] + I[i - 1] + R[i - 1]))
            return np.array(I[1:])

        def residual(p, y, x):
            return y - func(x, p)

        # 获得初始值
        x = np.arange(1, len(confirm))
        y = np.array(confirm[1:]).astype('int')
        # y = np.array(deathAndRecover[1:]).astype('int')
        p0 = (0.9, 0.01, 0.8, 1 / 3, min(10, 0.1 * (confirm[1] - confirm[0])))  # beta, gamma , beta_1, alpha , E0
        p_ols = least_squares(residual, p0, bounds=(
            [0.05, 0.01, 0.02, 1 / 21, 0], [5, 1, 5, 1, max(10000, 200 * (confirm[1] - confirm[0]))]), args=(y, x))
        self.p = p_ols.x

        return p_ols.x

        # # 传统求均值法
        # N = self.N
        # I_diff = np.diff(confirm)
        # R_diff = np.diff(deathAndRecover)
        # S_diff = - (I_diff + R_diff)
        #
        # S = np.ones(len(confirm)) * N - confirm - deathAndRecover
        # beta = np.mean(  -S_diff / ( confirm[1:] * S[1:] / N  )   )
        # gamma = (R_diff / confirm[1:]).mean()
        # print(beta , gamma)
        # self.beta = beta
        # self.gamma = gamma


def SEIR_analysis():
    # 使用 SEIR模型 进行显式地 'detrend'
    interval = 30
    start = 30  # 300
    end = start + interval
    #  60 - 60 + 180  [1.67873520e-01 2.24993480e-01 7.20000802e-02 1.39529231e-01
    #  6.70233638e+05]

    # 直观说明疫苗有用！！！
    # [5.00000000e-02 3.33870018e-02 2.00000000e-02 9.88471270e-02
    #  6.88611822e+06]

    # [5.00000000e-02 3.21871536e-02 2.00000000e-02 9.85106216e-01
    #  4.16764473e+04]

    model = SEIR_model()
    p_ols = model.get_paras(confirm=confirm_smooth[start:end] - (death_smooth[start:end] + recover_smooth[start:end]),
                            deathAndRecover=death_smooth[start:end] + recover_smooth[start:end])
    print(p_ols)

    import matplotlib.pyplot as plt
    S, I, R = model.forcast(ForcastDays=interval + 14)  # 多forcast几个 截断即可
    SEIR_pred = I

    plt.plot(timeSeries[start:end], confirm_smooth[start:end] - (death_smooth[start:end] + recover_smooth[start:end]),
             label='实际值')
    plt.plot(timeSeries[start:end], I, label='拟合值')
    plt.legend()
    plt.title('7/20 – 8/19 每日剩余总感染人数')
    plt.show()

    plt.plot(timeSeries[start:end], confirm_smooth[start:end] - (death_smooth[start:end] + recover_smooth[start:end]),
             color='blue', label='历史值')
    plt.plot(timeSeries[end: end + 14],
             confirm_smooth[end: end + 14] - (death_smooth[end: end + 14] + recover_smooth[end: end + 14]), color='red',
             label='实际值')
    plt.plot(timeSeries[end: end + 14], I[-14:], label='预测值', color='sandybrown')
    plt.legend()
    plt.title('预测剩余感染人数与真实剩余感染人数')
    plt.show()


if __name__ == '__main__':
    # 数据预处理
    DAYS = 180
    # China
    confirm = pd.read_csv('time_series_covid19_confirmed_global.csv')
    death = pd.read_csv('time_series_covid19_deaths_global.csv')
    recover = pd.read_csv('time_series_covid19_recovered_global.csv')
    confirm = confirm[confirm['Country/Region'] == 'China']
    death = death[death['Country/Region'] == 'China']
    recover = recover[recover['Country/Region'] == 'China']
    confirm_all = confirm.iloc[:, 4:].sum(axis=0)
    death_all = death.iloc[:, 4:].sum(axis=0)
    recover_all = recover.iloc[:, 4:].sum(axis=0)
    timeSeries = pd.to_datetime(confirm.columns[4:])
    # USA
    confirm = pd.read_csv('time_series_covid19_confirmed_US.csv')
    death = pd.read_csv('time_series_covid19_deaths_US.csv')
    recover = pd.read_csv('time_series_covid19_recovered_global.csv')
    confirm_all = confirm.iloc[:, 11:].sum(axis=0)  # US 全部确诊数据
    death_all = death.iloc[:, 12:].sum(axis=0)
    recover_all = recover[recover['Country/Region'] == 'US'].iloc[0, 4:]
    timeSeries = pd.to_datetime(confirm.columns[11:])  # 转时间格式


    # smooth Before Forcast And Predict
    def data_smooth(x, k=3):
        d = k // 2
        smooth_x = x.copy()
        for i in range(d, len(x) - d):
            smooth_x[i] = x[(i - d): (i + d + 1)].sum() / k
        return smooth_x


    confirm_smooth = data_smooth(np.array(confirm_all))
    death_smooth = data_smooth(np.array(death_all))
    recover_smooth = data_smooth(np.array(recover_all))
