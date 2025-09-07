# financial-exercises
projects for practicing machine learning/data analysis in financial management
# **长债预测和择时的量化视角-练习**
本项目参考了平安证券研报《债市深度：长债预测和择时的量化视角》并尝试复现报告结果。通过相关性分析验证了报告所发现的影响10年期国债收益率变化方向的主要前瞻经济变量，并利用这些变量训练了二分类Logistic模型以预测10年期国债的变化方向，并评估此模型的有效性。



## **1.数据处理**
此报告通过相关性分析发现了四个影响10年期国债收益率变化方向的主要前瞻经济变量：领先3M的社融存量同比，领先2M的M1-M2同比，领先1M的IRS:FR007（一年），领先2M的二手房出售挂牌价格指数环比。因此，首先从原始数据表中提取出这几列数据并将其与10年期国债到期收益率放在一张表中。

**处理月数据**
```python=
def gen_agg_time(df, col, freq="M", to_ts=False, how="start"):
    """
    将 df[col] 聚合到指定频率的 Pandas PeriodIndex。必要时再转 Timestamp。
    参数
    ----
    df   : DataFrame
    col  : str，要聚合的列名
    freq : str，"M" 月度, "Q" 季度, "A" 年度, "W" 周度 …
    to_ts: bool，True 则 Period → Timestamp
    how  : {"start", "end"}，to_ts 时取周期首日还是末日
    """
    ser = pd.to_datetime(df[col]).dt.to_period(freq)
    if to_ts:
        return ser.dt.to_timestamp("MS" if how == "start" else freq)
    return ser
```
```python=
df_mon['m1_roll'] = df_mon['M1(货币):同比']
df_mon['m2_roll'] = df_mon['M2(货币和准货币):期末值'].rolling(window=12).apply(
    lambda x: (x[-1] / x[0] - 1) * 100
)
df_mon["m1_m2"] = df_mon['m1_roll'] - df_mon['m2_roll']

# 处理月份时间
df_mon = df_mon.reset_index()
df_mon["time"] = gen_agg_time(df_mon, "time", "M") #grouping time to the month level
df_mon = df_mon.set_index("time")

df_mon.dropna(inplace=True)
```
由于领先1M的IRS:FR007是每日一个数据，而10年期国债收益率为每月一个数据，因此对IRS:FR007取每月的平均值转化为月数据。
**处理日数据**
```python=
df_day = df_day.reset_index()
df_day["agg_mon"] = gen_agg_time(df_day, "time", "M")
df_fr007 = df_day.groupby("agg_mon")["利率互换:FR007:1年"].mean().to_frame() #daily to monthlydf_day = df_day.reset_index()
df_day["agg_mon"] = gen_agg_time(df_day, "time", "M")
df_fr007 = df_day.groupby("agg_mon")["利率互换:FR007:1年"].mean().to_frame() #daily to monthly
```
**生成准标签**
```python=
df_10y = df_day.groupby("agg_mon")["国债到期收益率:10年"].mean().to_frame() #每月国债10年到期收益率
```
**组合成一张表**
```python=
df_ms = pd.concat([df_mon, df_fr007, df_10y], axis=1)
df_ms = df_ms.loc['2015-12':'202505']    ###通过上面的temp1.csv 文件发现原始数据从2016/1-2025/3是全的
df_ms.dropna(inplace=True)
```
![相关性可视化](D:/BaiduSyncdisk/大学/机器学习和金融量化By唐老师/微信图片_20250904205404_16_22.png)

## **2.相关性可视化**
为了验证和展示四种影响因素和10年期国债收益率的相关性，绘制这四种因素与10年期国债收益率的对比折线图，并得到相关性系数。
**相关性可视化(M1-M2同比)**
```python=
fig, ax = plt.subplots(figsize=(12, 6)) 
ax = df_ms["m1_m2"].plot(
    ax=ax,
    ylabel="m1_m2",
    title="M1-M2同比与10年期国债收益率对比图",
    c="orange"
)
ax.set_ylim(-15, 20)  # 设置左侧y轴范围
df_ms["国债到期收益率:10年"].plot(
    secondary_y=True,
    ax=ax,
    ylabel="国债到期收益率:10年",
    c="blue"
)
plt.show()
```
**散点图与相关性系数**
```python=
df_ms[["m1_m2", "国债到期收益率:10年"]].plot(kind='scatter', x='m1_m2', y='国债到期收益率:10年', figsize=(10,6))

df_ms[["m1_m2", "国债到期收益率:10年"]].corr()
```
![相关性可视化](D:\BaiduSyncdisk\大学\机器学习和金融量化By唐老师\M1-M2同比与10年期国债收益率对比.png)
![相关性可视化](D:\BaiduSyncdisk\大学\机器学习和金融量化By唐老师\社会融资规模存量同比与10年期国债收益率对比.png)
![相关性可视化](D:\BaiduSyncdisk\大学\机器学习和金融量化By唐老师\微信图片_20250904223131_17_22.png)
![相关性可视化](D:\BaiduSyncdisk\大学\机器学习和金融量化By唐老师\二手住房价格指数与10年期国债收益率对比.png)
| **影响因素**                        | 相关性系数 |
| ----------------------------------- | ---------- |
| M1-M2同比（领先2M）                 | 0.381765   |
| 社会融资规模存量:期末同比（领先3M） | 0.71993    |
| 利率互换:FR0071（领先1M）           | 0.898775   |
| 二手住房价格指数环比(上月=100)（领先2M）                                 |       0.648756     |

根据以上折线图和相关系数可得出10年期国债收益率的变化趋势与四种影响因素的变化趋势都有较明显的关系，且与领先1年的利率互换:FR0071关系最显著。
## **3.模型训练**
训练二分类Logistic模型以预测10年期国债未来的变化方向。以2016年3月-2024年6月的数据为训练集，2024年7月~2025年4月的数据为验证集，验证模型预测结果。
**生成模型训练的数据集**
```python=
df_ms["社融存量同比_3M"] = df_ms["社会融资规模存量:期末同比"].shift(1)  #领先3年
df_ms["m1_m2_2M"] = df_ms["m1_m2"].shift(1) 
df_ms["二手住房价格指数环比_2M"] = (df_ms["二手住房价格指数(上月=100)"] - 100).shift(1)
df_ms["FR007_2M"] = df_ms["利率互换:FR007:1年"].shift(1)
# 构建数据集
df_ms["label"] = df_ms["国债到期收益率:10年"].pct_change()  #收益率变化
df_ms["label_final"] = df_ms["label"].apply(lambda x: 0 if x < 0 else 1)
```
**自定义验证集**
```python=
df_data = df_ms.loc["2016-03":"2024-06", ['社融存量同比_3M', 'm1_m2_2M', '二手住房价格指数环比_2M',
                                          'FR007_2M', 'label_final']]
df_data_valid = df_ms.loc["2024-07":"2025-04", ['社融存量同比_3M', 'm1_m2_2M', '二手住房价格指数环比_2M',
                                          'FR007_2M', 'label_final']]
# for col in ['社融存量同比_3M', 'm1_m2_2M', '二手住房价格指数环比_2M',
#                                              'FR007_2M']:
#     df_data[col] = (df_data[col] - df_data[col].mean())/ (df_data[col].std() + 1e-8)

num = 50  #train 50, test 50
num_val=10
train_x = df_data.iloc[:num, :4]
train_y = df_data.iloc[:num, 4:]
test_x = df_data.iloc[num:, :4]
test_y = df_data.iloc[num:, 4:]
val_x=df_data_valid.iloc[:num_val, :4]
val_y=df_data_valid.iloc[:num_val, 4:]
```
**训练与验证**
```python=
df_data = df_ms.loc["2016-03":"2024-06", ['社融存量同比_3M', 'm1_m2_2M', '二手住房价格指数环比_2M',
                                          'FR007_2M', 'label_final']]
df_data_valid = df_ms.loc["2024-07":"2025-04", ['社融存量同比_3M', 'm1_m2_2M', '二手住房价格指数环比_2M',
                                          'FR007_2M', 'label_final']]
# for col in ['社融存量同比_3M', 'm1_m2_2M', '二手住房价格指数环比_2M',
#                                              'FR007_2M']:
#     df_data[col] = (df_data[col] - df_data[col].mean())/ (df_data[col].std() + 1e-8)

num = 50  #train 50, test 50
num_val=10
train_x = df_data.iloc[:num, :4]
train_y = df_data.iloc[:num, 4:]
test_x = df_data.iloc[num:, :4]
test_y = df_data.iloc[num:, 4:]
val_x=df_data_valid.iloc[:num_val, :4]
val_y=df_data_valid.iloc[:num_val, 4:]
```
| **Class**      | Precision | Recall | F1-Score | Support |
| -------------- | --------- | ------ | -------- | ------- |
| 0              | 0.67      | 1.00   | 0.80     | 6       |
| 1              | 0.00      | 0.00   | 0.00     | 3       |
| **Accuracy**   |           |        | 0.67     | 9       |
| **Macro Avg**  | 0.33      | 0.50   | 0.40     | 9       |
| **Weighted Avg** | 0.44    | 0.67   | 0.53     | 9       |

根据最后一段代码的训练结果，模型在验证集上的准确率为0.67，说明模型对10年期国债收益率变化方向的预测有一定的区分能力，但整体表现一般。从混淆矩阵和分类报告来看，模型对“上涨”类别（标签1）的识别能力较弱，precision和recall均为0，未能正确预测任何上涨样本；而对“下跌”类别（标签0）则全部预测正确，recall达到1.00，f1-score为0.80。这可能是由于2016到2025年10年期国债收益率整体呈下行趋势，导致样本不均衡，使得模型容易偏向预测“下跌”类别。训练集和验证集数据量很少也是模型容易受样本不均衡影响的重要原因。总体而言，模型能够部分反映经济变量对国债收益率的影响，但在实际应用中还需进一步优化特征选择、样本均衡和模型参数，以提升对不同变化方向的识别能力和预测稳定性。
