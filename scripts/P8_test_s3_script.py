# # To run locally 
# spark-submit --packages com.amazonaws:aws-java-sdk-pom:1.10.34,org.apache.hadoop:hadoop-aws:2.7.2 s3_script.py

import os
from pyspark import SparkContext
from pyspark.sql import SparkSession

path_cred = os.path.join(os.getcwd(), "AWS/AWS_IAM_CREDENTIAL/Maryse_P8_credentials.csv")
with open(path_cred,'r') as f:
        msg = f.read()
ID = str(msg).split('\n')[1].split(',')[2]
KEY = str(msg).split('\n')[1].split(',')[3]

spark = SparkSession.builder.appName('FeatExtr').getOrCreate()
sc = spark.sparkContext.getOrCreate()

hadoopConf = sc._jsc.hadoopConfiguration()
hadoopConf.set("fs.s3a.access.key", ID)
hadoopConf.set("fs.s3a.secret.key", KEY)

from pyspark.ml.image import ImageSchema
PREFIX = 'data/SAMPLE'
bucket='ocfruitpictures'
data_path = 's3a://{}/{}'.format(bucket, PREFIX)
images_df = ImageSchema.readImages(data_path, recursive=True)
images_df.show()
print("WELL DONE !")