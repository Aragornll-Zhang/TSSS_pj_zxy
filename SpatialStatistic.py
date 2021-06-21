# -*- coding : utf-8-*-
import pandas as pd
import numpy as np
import datetime


def variogram(coordinates, values):
    import skgstat as skg
    # the data functions return a dict of 'sample' and 'description'
    # coordinates, values = skg.data.pancake(N=300).get('sample') # 二维数组
    V = skg.Variogram(coordinates=coordinates, values=values)
    print(V)
    fig = V.plot()
    return


def EpidemicMap(data_city, someday_str, piecewise=None):
    # 画图
    from pyecharts.charts import Map
    import pyecharts.options as opts
    from pyecharts.faker import Faker

    if piecewise is not None:
        china_city = (
            Map().add(
                "",
                data_city,
                "china-cities",
                is_map_symbol_show=False,
            ).set_series_opts(label_opts=opts.LabelOpts(is_show=True, font_size=10)).set_global_opts(
                title_opts=opts.TitleOpts(title="疫情分布市级地图"),
                visualmap_opts=opts.VisualMapOpts(
                    max_=max(data_city, key=lambda x: x[1])[1], is_piecewise=True,
                    pieces=[{"max": piecewise[0], "min": 0, "label": '0-' + str(piecewise[0]), "color": "#FF8099"},
                            {"max": piecewise[1], "min": piecewise[0] + 1,
                             "label": str(piecewise[0]) + '-' + str(piecewise[1]), "color": "#e9967a"},
                            {"max": piecewise[2], "min": piecewise[1] + 1,
                             "label": str(piecewise[1]) + "-" + str(piecewise[2]), "color": "#ff8c69"},
                            {"max": piecewise[3], "min": piecewise[2] + 1,
                             "label": str(piecewise[2]) + "-" + str(piecewise[3]), "color": "#ff6347"},
                            {"max": max(data_city, key=lambda x: x[1])[1], "min": piecewise[-1],
                             "label": ">=" + str(piecewise[-1]), "color": "#ff0000"},
                            ],
                ),
            ).render(path='市级疫情况' + someday_str + '.html')
        )
    else:
        china_city = (
            Map().add(
                "",
                data_city,
                "china-cities",
                label_opts=opts.LabelOpts(is_show=False),
            ).set_global_opts(
                title_opts=opts.TitleOpts(title="疫情分布市级地图"),
                visualmap_opts=opts.VisualMapOpts(
                    min_=min(data_city, key=lambda x: x[1])[1],
                    max_=max(data_city, key=lambda x: x[1])[1],
                    range_color=["lightskyblue", "yellow", "orangered", "red"],
                    is_piecewise=False
                ),

            ).render(path='市级疫情况' + str(someday)[5:10] + '.html')
        )

    return


def calculate_Dist(coord1, coord2):
    ''' 计算球面大圆距离
    '''
    # (114.31, 30.52) # wuhan
    # (116.46, 39.92) # beijing
    # calculate_Dist((114.31, 30.52) , (121.48, 31.22)  ) # (121.48, 31.22)
    phi_1, theta_1 = coord1
    phi_2, theta_2 = coord2
    coord1_eucili = np.array([np.sin(theta_1 / 180 * np.pi),
                              np.cos(theta_1 / 180 * np.pi) * np.cos(phi_1 / 180 * np.pi),
                              np.cos(theta_1 / 180 * np.pi) * np.sin(phi_1 / 180 * np.pi)])

    coord2_eucili = np.array([np.sin(theta_2 / 180 * np.pi),
                              np.cos(theta_2 / 180 * np.pi) * np.cos(phi_2 / 180 * np.pi),
                              np.cos(theta_2 / 180 * np.pi) * np.sin(phi_2 / 180 * np.pi)])

    return np.arccos(max(-1, min(np.sum(coord1_eucili * coord2_eucili), 1))) * 6371


def get_cityposition(city_position={}):
    with open('全国各市经纬度.txt', 'r', encoding="utf-8") as f:
        head = True
        for row in f.readlines():
            if head:
                head = False
                continue
            else:
                if len(row.split()) == 3:
                    city_list = row.split()
                    city_position[city_list[0]] = (float(city_list[1]), float(city_list[2]))
    return city_position


if __name__ == '__main__':
    # 读入数据
    df = pd.read_csv('nCov_china_0319.csv')
    df['日期'] = pd.to_datetime(df['日期'])
    DAY = datetime.timedelta(days=1)

    # 中国各市区经纬度
    city_position = get_cityposition()  # 获得全国各市经纬度

    someday = pd.Timestamp('2020-01-24')
    # someday = pd.Timestamp('2020-02-07') # 封城后第一个14天
    # someday = pd.Timestamp('2020-02-21')

    df_red = df.loc[:, ['市', '确诊', '日期', '新增确诊']]
    df_red = df_red[(df_red['市'] != '待明确地区')]
    test_df = df_red[(df_red['日期'] == someday)]

    distance = []
    citys = []
    confirm_value = []

    Interval_Confirm_Flag = True
    days = 14  # 潜伏期

    WuHan_position = city_position['武汉']
    for i in range(len(test_df)):
        city_name = test_df['市'].iloc[i]
        try:
            if Interval_Confirm_Flag:
                new_infects = float(
                    (df_red[(df_red['日期'] == someday + days * DAY) & (df_red['市'] == city_name)])['确诊']) - \
                              float((df_red[(df_red['日期'] == someday) & (df_red['市'] == city_name)])['确诊'])
            else:
                new_infects = test_df['确诊'].iloc[i]

            if not np.isnan(new_infects):  # 是否加0 or new_infects == 0
                if city_name in city_position:
                    distance.append(calculate_Dist(WuHan_position, city_position[city_name]))
                    confirm_value.append(new_infects)
                    citys.append(city_name)
                elif city_name[0:-1] in city_position:
                    distance.append(calculate_Dist(WuHan_position, city_position[city_name[0:-1]]))
                    confirm_value.append(new_infects)
                    citys.append(city_name)
        except:
            pass
    print(len(confirm_value))

    # 两个潜伏期后，基本跟武汉关系不大了
    variogram(np.array(distance), np.array(confirm_value))  # 直观上，我们用 1.25 // 2.25 号的数据来看，差异明显

    # 画地理分布图
    someday_str = '1-24to2-7'  # '2-21to3-6'
    data_city = [(citys[i], confirm_value[i]) for i in range(len(citys))]
    a = np.log1p(confirm_value)
    a.sort()
    piecewise = [0] * 4
    for i in range(4):
        piecewise[i] = int(np.exp(a[(i + 1) * (len(a) - 3) // 4]))
    print(piecewise)
    len(data_city)
    EpidemicMap(data_city, someday_str, piecewise)

    # 2/14 后判断趋势
    Hubei = {'咸宁', '黄冈', '随州', '潜江', '荆门', '荆州', '宜昌', '仙桃', '十堰',
             '鄂州', '武汉', '神农架林区', '恩施州', '孝感', '襄阳', '天门', '黄石'}

    # 在 2-7 号之后的新地理特征
    # 湖北之外 聚类 + 湖北
    coor_confirm = []
    for i in range(len(confirm_value)):
        if citys[i] in Hubei or citys[i][0:-1] in Hubei: continue
        if citys[i] in city_position:
            coordinate = city_position[citys[i]]
        else:
            coordinate = city_position[citys[i][0:-1]]
        coor_confirm.append((coordinate, confirm_value[i]))

    coor_confirm.sort(key=lambda x: -x[1])

    from sklearn.cluster import KMeans

    k = 10
    x = np.array([i for i, _ in coor_confirm[0:k]])
    n_clusters = 5  # 类簇的数量
    estimator = KMeans(n_clusters)  # 构建聚类器
    estimator.fit(x)
    print(estimator.cluster_centers_)

    x_all = x = np.array([i for i, _ in coor_confirm])
    labels = estimator.predict(x_all)

    Cluster_distance = [calculate_Dist(coor_confirm[i][0], estimator.cluster_centers_[labels[i]]) for i in
                        range(len(coor_confirm))]
    Cluster_confirms = [v for _, v in coor_confirm]

    # 加回武汉各市
    for i in range(len(citys)):
        if citys[i] in Hubei or citys[i][0:-1] in Hubei:
            if citys[i] in city_position:
                Cluster_distance.append(calculate_Dist(city_position[citys[i]], WuHan_position))
                Cluster_confirms.append(confirm_value[i])
            elif citys[i][0:-1] in city_position:
                Cluster_distance.append(calculate_Dist(city_position[citys[i][0:-1]], WuHan_position))
                Cluster_confirms.append(confirm_value[i])

    variogram(np.array(Cluster_distance), np.array(Cluster_confirms))  # 直观上，我们用 1.25 // 2.25 号的数据来看，差异明显

    data_city = [(citys[i], confirm_value[i]) for i in range(len(citys))]

    a = np.log1p(confirm_value)
    a.sort()
    piecewise = [0] * 4
    for i in range(4):
        piecewise[i] = int(np.exp(a[(i + 1) * (len(a) - 3) // 4]))
    print(piecewise)
    len(data_city)
    EpidemicMap(data_city, someday, piecewise)
