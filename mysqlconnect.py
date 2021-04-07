
# test connection to AWS RDS MySQL database

import pymysql # pip3 install PyMySQL
import sys
import boto3
import os

connection = pymysql.connect(
    database="videometadata",
    host="videometadata.cqcc6zskeglo.us-west-1.rds.amazonaws.com",
    user="admin",
    password="Twister123!",
    port=3306
    )

with connection:
    with connection.cursor() as cursor:
        sql = "SELECT * FROM video"
        # cursor.execute(sql, ('webmaster@python.org',))
        cursor.execute(sql)
        # result = cursor.fetchone()
        result = cursor.fetchall()
        print(result)