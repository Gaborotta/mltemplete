# %%
from sys import displayhook
import mlflow
import matplotlib.pyplot as plt
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit


# %%
spark = (
    SparkSession.builder
    .master("local")
    .appName("Word Count")
    .getOrCreate()
)
# %%
spark

# %%
EXPERIMENT_NAME = "MlFlowのテスト"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# %%

# Loads data.
dataset = spark.read.csv("major_results_2020.csv",
                         inferSchema=True, header=True, nullValue='-')
dataset.show()

data_customer = dataset.na.drop()

# %%
data_customer.columns

data_customer = (
    data_customer
    .where(col('地域識別コード').isin(1, 2, 3))
    .where(col('外国人') != '#VALUE!')
    .withColumn('外国人', col('外国人').cast('int'))
)

data_customer.printSchema()
# %%
assemble = VectorAssembler(inputCols=[
    '総数', '男', '女', '2015年（平成27年）の人口（組替）',
    '5年間の人口増減数', '5年間の人口増減率',
    '面積（参考）', '人口密度', '平均年齢', '年齢中位数', '15歳未満14', '15～64歳15', '65歳以上16', '15歳未満17', '15～64歳18', '65歳以上19', '15歳未満20', '15～64歳21', '65歳以上22', '15歳未満23', '15～64歳24', '65歳以上25', '15歳未満26', '15～64歳27', '65歳以上28', '15歳未満29', '15～64歳30', '65歳以上31', '人口性比', '日本人', '外国人', '総世帯', '一般世帯36', '施設等の世帯', '2015年（平成27年）の世帯数（組替）', '一般世帯39', 'うち核家族世帯', '夫婦のみの世帯', '夫婦と子供から成る世帯', '男親と子供から成る世帯', '女親と子供から成る世帯', 'うち単独世帯', 'うち65歳以上の単独世帯', '夫65歳以上，妻60歳以上の夫婦のみの世帯', '3世代世帯'], outputCol='features')
assembled_data = assemble.transform(data_customer)
assembled_data.show(2)

# %%
scale = StandardScaler(inputCol='features', outputCol='standardized')
data_scale = scale.fit(assembled_data)
data_scale_output = data_scale.transform(assembled_data)
data_scale_output.show(2)

# %%
mlflow.start_run(experiment_id=experiment.experiment_id)
mlflow.set_tag("release.version", "2.2.0")


# %%
silhouette_score = []
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',
                                metricName='silhouette', distanceMeasure='squaredEuclidean')

mlflow.set_tag("ClusteringEvaluator", {
    "predictionCol": 'prediction',
    "featuresCol": 'standardized',
    "metricName": 'silhouette',
    "distanceMeasure": 'squaredEuclidean'
})

mlflow.log_param("k", list(range(2, 50)))
mlflow.log_param("maxIter", 300)

for i in range(2, 50):

    KMeans_algo = KMeans(featuresCol='standardized', k=i, maxIter=300)

    KMeans_fit = KMeans_algo.fit(data_scale_output)

    output = KMeans_fit.transform(data_scale_output)

    score = evaluator.evaluate(output)

    silhouette_score.append(score)

    print("Silhouette Score:", score)
    mlflow.log_metric("Silhouette Score", score, step=i)

# %%
# Visualizing the silhouette scores in a plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(range(2, 50), silhouette_score)
ax.set_xlabel('k')
ax.set_ylabel('cost')

mlflow.log_figure(fig, "figure.png")

# %%

# %%
mlflow.end_run()

# %%
KMeans_algo = KMeans(featuresCol='standardized', k=11)

KMeans_fit = KMeans_algo.fit(data_scale_output)
output = KMeans_fit.transform(data_scale_output)

centers = KMeans_fit.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)

# %%
output.show()
output.columns

# %%


# mlflow.log_figure(fig, "figure.html")

# %%
(
    output
    .groupBy('prediction')
    .agg(
        F.count('*'),
        *[F.avg(c).alias(c) for c in cols]
    )
    .orderBy('prediction')
).toPandas().to_csv('./res.csv', index=False, encoding='utf_8_sig')

(
    output
    .select(
        '都道府県名', '都道府県・市区町村名1', '都道府県・市区町村名2', '地域識別コード', '総数', 'prediction'
    )
    .orderBy('prediction')
).toPandas().to_csv('./res_whole.csv', index=False, encoding='utf_8_sig')
