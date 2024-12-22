import psycopg2
import numpy as np
from R2T import R2T

def get_db_connection():
    return psycopg2.connect(
        dbname="tpch1g",
        user="postgres",
        password="Yujinqi.2002",
        host="localhost",
        port="5432"
    )
    
multi_private_query = """
SELECT supplier.s_suppkey, customer.c_custkey, l_extendedprice * (1 - l_discount) / 1000
FROM supplier, lineitem, orders, customer
WHERE supplier.s_suppkey = lineitem.l_suppkey AND lineitem.l_orderkey = orders.o_orderkey
AND orders.o_custkey = customer.c_custkey AND orders.o_orderdate >= '1997-07-01'
"""

if __name__ == "__main__":
    conn = get_db_connection()
    r2t = R2T(gs_Q=1024)
    result = r2t.query(conn, multi_private_query)
    conn.close()
    print("R2T Q:" + str(result))